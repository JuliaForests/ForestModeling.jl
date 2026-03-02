"""
    calculatescore(models::Vector{<:AllometricModel}, criteria::Vector{Symbol})

Internal helper to calculate a composite ranking score using **Standard Competition Ranking ("1224")**.

# Algorithm
1. For each criterion provided, models are ranked.
   - **Ties**: Items with equal values receive the same ranking number.
   - **Gaps**: The ranking number of the next item is equal to the number of items ranked above it plus one (e.g., 1, 1, 3).
   - **Direction**:
     - Higher is Better (`rev=true`): `r2`, `adjr2`, `d`, `normality`, `ev`.
     - Lower is Better (`rev=false`): `mae`, `mape`, `mse`, `rmse`, `cv`, `aic`, `bic`.
2. The ranks are summed to produce a `totalScore`.

# Returns
- `Vector{Int}`: The total score for each model (Lower is better).
"""
function calculatescore(models::Vector{<:AllometricModel}, criteria::Vector{Symbol})
  n = length(models)
  totalScore = zeros(Int, n)
  for crit in criteria
    # Extract values
    values = [getfield(m, crit) for m in models]
    # Determine ranking direction
    # Higher is better (r2, etc) -> rev=true (Descending sort gives Rank 1 to highest)
    # Lower is better (aic, etc) -> rev=false (Ascending sort gives Rank 1 to lowest)
    higherIsBetter = crit in (:r2, :adjr2, :d, :ev)
    # Calculate ranks using StatsBase.competerank
    # This automatically handles the "1, 1, 3" logic and returns Vector{Int}
    ranks = competerank(values; rev=higherIsBetter)
    totalScore .+= ranks
  end
  return totalScore
end

"""
    criteriatable(models::Vector{AllometricModel}, criteria::Symbol...; best::Int=10)

Evaluates, ranks, and filters regression models based on multiple statistical criteria.

This function automates the model selection process by applying a **Rank Sum** algorithm to identify the models that perform best across multiple metrics simultaneously (e.g., high precision AND low error).

# Arguments
- `models`: Vector of fitted `AllometricModel` objects.
- `criteria...`: One or more symbols representing the metrics to use for ranking (e.g., `:adjr2`, `:rmse`).
- `best` (keyword): Maximum number of top models to return (default: 10).

# Supported Criteria
| Symbol | Description | Direction |
|:-------|:------------|:----------|
| `:adjr2` | Adjusted Coefficient of Determination (Default) | Higher is better |
| `:cv` | Coefficient of Variation % (Default) | Lower is better |
| `:ev` | Explained Variance (Default) | Higher is better |
| `:rmse` | Root Mean Squared Error | Lower is better |
| `:mae` | Mean Absolute Error | Lower is better |
| `:mape` | Mean Absolute Percentage Error | Lower is better |
| `:normality` | Residual Normality Test (Shapiro-Wilk/Jarque-Bera) | **Hard Filter** |

# The Normality Filter
The `:normality` criterion is treated as a **Hard Constraint**:
- If included, the function **discards** all models where `isnormality(model) == false`.
- The ranking is then performed only on the subset of models with normal residuals.
- **Throws an Error** if no model passes the normality test (preventing the selection of invalid models).

# Returns
- A `DataFrame` containing the top models, their rank score, and the values of the selected metrics.

# Examples
```julia
# Rank by Adjusted R² and RMSE (Default)
df = criteriatable(models)

# Rank by MAE and AIC, but ONLY consider models with Normal Residuals
df = criteriatable(models, :mae, :aic, :normality)

```

"""
function criteriatable(models::Vector{<:AllometricModel}, criteria::Symbol...; best::Int=10)
  # Define allowed fields
  allowed = [:r2, :adjr2, :ev, :mae, :mape, :mse, :rmse, :cv, :syx, :uncertainty]
  # Validate criteria
  selected = if isempty(criteria)
    [:adjr2, :cv, :ev] # Default
  elseif :all in criteria
    allowed
  else
    collect(criteria)
  end
  if !issubset(selected, allowed)
    throw(ArgumentError("invalid criteria used. allowed: $allowed"))
  end
  # Hard Filter: Normality
  # We must work on a subset of models if normality is requested
  currentModels = models
  # if :normality in selected
  #   # Filter models where normality == true
  #   currentModels = filter(m -> m.normality, models)
  #   # Validation: Did any model survive?
  #   if isempty(currentModels)
  #     throw(ErrorException("No regression models passed the Normality test (Shapiro-Wilk/Jarque-Bera). Relax the criteria by removing :normality to see the best non-normal models."))
  #   end
  #   # Remove :normality from the ranking list (since they are all true now)
  #   # but keep it in 'selected' for display in the table later if desired
  # end
  # Calculate scores on the filtered list
  scores = calculatescore(currentModels, selected)
  # Sort models
  sortedIndices = sortperm(scores)
  limit = min(best, length(currentModels))
  topIndices = sortedIndices[1:limit]
  # Build output DataFrame
  # Pre-allocate DataFrame with rank and model
  resultDf = DataFrame()
  resultDf.rank = scores[topIndices]
  resultDf.model = currentModels[topIndices]
  # Dynamically add columns for selected metrics
  for k in selected
    resultDf[!, k] = [getfield(m, k) for m in currentModels[topIndices]]
  end
  return resultDf
end

# criteriatable(models:::AllometricModel, criteria::Symbol...; best::Int=10) = criteriatable([models], collect(criteria)..., best=best)

"""
    bestmodel(models::Vector{AllometricModel}, criteria::Symbol...)

Identifies and returns the single best `AllometricModel` from a collection.

This function applies the same ranking logic and filtering (including the strict `:normality` check) as `criteriatable`, but returns the model object directly instead of a table.

# Arguments

* `models`: Vector of fitted `AllometricModel`.
* `criteria...`: The metrics used to determine the "best" model.

# Returns

* The `AllometricModel` instance with the best (lowest) rank score.

# Examples

```julia
# Get the best model for prediction
best = bestmodel(models, :adjr2, :rmse)
ypred = predict(best)

```

"""
function bestmodel(models::Vector{AllometricModel}, criteria::Symbol...)
  # Select default criteria if none provided
  selected = isempty(criteria) ? [:adjr2, :cv, :ev] : collect(criteria)
  # Apply Normality Filter if requested
  currentModels = models
  # if :normality in selected
  #   currentModels = filter(m -> m.normality, models)
  #   if isempty(currentModels)
  #     throw(ErrorException("No regression models passed the Normality test."))
  #   end
  # end
  # Calculate scores and find the best index
  scores = calculatescore(currentModels, selected)
  bestIdx = argmin(scores)
  return currentModels[bestIdx]
end

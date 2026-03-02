# ==============================================================================
# DIAGNOSTICS (NORMALITY)
# ==============================================================================

"""
    isnormality(x::AbstractVector{<:Real})

Tests if the input vector `x` (residuals) comes from a Normal distribution.
Returns `true` if the null hypothesis (normality) cannot be rejected (p > 0.05).

Implementation details:
- **N < 3**: Insufficient data (defaults to true).
- **3 ≤ N ≤ 5000**: Uses **Shapiro-Wilk** (Gold standard for this range).
- **N > 5000**: Uses **Jarque-Bera** (Asymptotic test suitable for large samples).
"""
function isnormality(x::AbstractVector{<:Real})
  n = length(x)
  if n < 3
    return true
  elseif n <= 5000
    # Shapiro-Wilk is the most powerful test for this range
    # It tests: H0 = Data is Normal vs H1 = Data is NOT Normal
    return pvalue(ShapiroWilkTest(x)) > 0.05
  else
    # Shapiro-Wilk is not valid for N > 5000 in this package
    # Jarque-Bera uses Skewness and Kurtosis to test normality asymptotically
    return pvalue(JarqueBeraTest(x)) > 0.05
  end
end

"""
    isnormality(model::AllometricModel)

Helper function to test normality of the model's stored residuals.
"""
isnormality(model::AllometricModel) = model.normality

# ==============================================================================
# BASIC MODEL PROPERTIES
# ==============================================================================

"""
    nobs(model::AllometricModel)

Returns the number of observations used to fit the model.
"""
nobs(model::AllometricModel) = model.n

"""
    dof_residual(model::AllometricModel)

Returns the residual degrees of freedom (`n - p - 1`).
"""
dof_residual(model::AllometricModel) = model.ν

"""
    dof(model::AllometricModel)

Returns the degrees of freedom consumed by the model parameters (number of coefficients + 1 for intercept/variance).
"""
dof(model::AllometricModel) = length(model.β) + 1

"""
    formula(model::AllometricModel)

Returns the `StatsModels.Formula` object used to fit the model.
"""
formula(model::AllometricModel) = model.formula

# ==============================================================================
# COEFFICIENTS & INFERENCE
# ==============================================================================

"""
    coef(model::AllometricModel)

Returns the vector of estimated regression coefficients (β).
"""
coef(model::AllometricModel) = model.β

"""
    coefnames(model::AllometricModel)

Returns the names of the coefficients based on the formula terms.
"""
coefnames(model::AllometricModel) = StatsModels.coefnames(model.formula.rhs)

"""
    vcov(model::AllometricModel)

Returns the Variance-Covariance matrix of the estimated parameters.
"""
vcov(model::AllometricModel) = model.Σ

"""
    stderror(model::AllometricModel)

Returns the standard errors of the coefficients (square root of the diagonal of `vcov`).
"""
stderror(model::AllometricModel) = sqrt.(diag(model.Σ))

"""
    confint(model::AllometricModel; level::Real=0.95)

Computes confidence intervals for the coefficients at the specified confidence `level` (default 0.95), based on the t-distribution.
"""
function confint(model::AllometricModel; level::Real=0.95)
  h = 1 - (1 - level) / 2
  tdist = TDist(dof_residual(model))
  crit = quantile(tdist, h)
  se = stderror(model)
  return hcat(model.β .- crit .* se, model.β .+ crit .* se)
end

"""
    pvalue(model::AllometricModel)

Calculates the two-tailed p-values for the t-test of the coefficients (H0: β = 0).
"""
function HypothesisTests.pvalue(model::AllometricModel)
  β = coef(model)
  se = stderror(model)
  tval = β ./ se
  return 2 .* ccdf.(TDist(dof_residual(model)), abs.(tval))
end

"""
    coeftable(model::AllometricModel; level::Real=0.95)

Returns a formatted table of coefficients, standard errors, t-statistics, p-values, and confidence intervals.
"""
function coeftable(model::AllometricModel; level::Real=0.95)
  cc = coef(model)
  se = stderror(model)
  tt = cc ./ se
  # use t-distribution based on residual degrees of freedom
  dist = TDist(dof_residual(model))
  # calculate p-values (two-tailed)
  pp = 2 .* ccdf.(dist, abs.(tt))
  # calculate confidence intervals
  α = 1 - level
  crit = quantile(dist, 1 - α / 2)
  lower = cc .- crit .* se
  upper = cc .+ crit .* se
  # format level string (e.g., "95")
  levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
  # coefficient table
  StatsBase.CoefTable(
    hcat(cc, se, tt, pp, lower, upper),
    ["Coef.", "Std. Error", "t", "Pr(>|t|)", "Lower $levstr%", "Upper $levstr%"],
    coefnames(model),
    4, # p-value column index
    3  # t-statistic column index
  )
end

# ==============================================================================
# DATA, PREDICTIONS & RESPONSE
# ==============================================================================

"""
    modelmatrix(model::AllometricModel)

Returns the design matrix (X) used in the regression.
"""
modelmatrix(model::AllometricModel) = modelcols(model.formula.rhs, model.data)

"""
    response(model::AllometricModel)

Returns the response vector (y) from the model data on the original scale.
"""
response(model::AllometricModel) = model.data[1]

"""
    predictbiascorrected!(ŷ::Vector{<:Real}, xcol::Union{AbstractVector{<:Real},Nothing}, ft::FunctionTerm, σ²::Real)

In-place version. Overwrites the input vector `ŷ` with the bias-corrected predictions on the original scale.
Internal helper for `predict`.
"""
function predictbiascorrected!(ŷ::Vector{<:Real}, xcol::Union{AbstractVector{<:Real},Nothing}, ft::FunctionTerm, σ²::Real)
  fname = nameof(ft.f)
  n = length(ŷ)
  # If no transformation, do nothing (ŷ remains unchanged)
  if fname ∉ (:log, :log1p, :inv, :sqrt, :cbrt, :petterson, :naslund, :prodan)
    return
  end
  @inbounds @simd for i in 1:n
    z = ŷ[i]
    if fname == :log
      # log-normal (exact)
      ŷ[i] = exp(z + 0.5σ²)
    elseif fname == :log1p
      # We use expm1 for numerical precision
      ŷ[i] = expm1(z + 0.5σ²)
    elseif fname == :inv
      # inverse (1/y)
      # g(z) = 1/z,  ∂²g = 2/z^3
      g = 1 / z
      ∂²g = 2 / (z^3)
      ŷ[i] = g + 0.5σ² * ∂²g
    elseif fname == :sqrt
      # Y = z^2 + correction
      # g'' = 2
      ŷ[i] = (z^2) + σ²
    elseif fname == :cbrt
      # Y = z^3 + correction
      # g'' = 6z
      # corr = 0.5 * sig2 * 6z = 3 * z * sig2
      ŷ[i] = (z^3) + (3 * z * σ²)
    elseif fname == :petterson
      # inverse sqrt (1/√y)
      # g(z) = 1/z^2,  ∂²g = 6/z^4
      z² = z^2
      g = 1 / z²
      ∂²g = 6 / (z²^2)
      ŷ[i] = g + 0.5σ² * ∂²g
    elseif fname == :naslund
      # x/√y
      # g(z) = x^2/z^2,  ∂²g = 6x^2/z^4
      x = xcol[i]
      x² = x^2
      z² = z^2
      g = x² / z²
      ∂²g = (6 * x²) / (z²^2)
      ŷ[i] = g + 0.5σ² * ∂²g
    elseif fname == :prodan
      # x^2/y
      # g(z) = x^2/z,  ∂²g = 2x^2/z^3
      x = xcol[i]
      x² = x^2
      g = x² / z
      ∂²g = (2 * x²) / (z^3)
      ŷ[i] = g + 0.5σ² * ∂²g
    end
  end
end

"""
    predict(model::AllometricModel)

Generates predictions from a regression model on the original scale of the dependent variable.

This function automatically handles the back-transformation of the dependent variable (`y`) if it was transformed during model fitting (e.g., `log(y)`, `1/y`). Crucially, it applies statistical bias correction to account for the non-linear transformation of the error term (Jensen's Inequality), ensuring unbiased estimates on the original scale.

# Correction Strategies
**1. Log-Normal Correction (Exact):** ``E[y] = \\exp(\\hat{z} + \\sigma^2/2)``
**2. Delta Method Correction (Approximate):** ``E[y] \\approx g(\\hat{z}) + \\frac{1}{2}\\sigma^2 g''(\\hat{z})``

# Returns
- `Vector{Float64}`: The predicted values on the original scale.
"""
predict(model::AllometricModel) = predict(model, model.data)

"""
    fitted(model::AllometricModel)

Alias for `predict(model)`. Returns the fitted values on the original scale.
"""
fitted(model::AllometricModel) = predict(model)

"""
    predict(model::AllometricModel, data)

Generates predictions for a new dataset `data`. Applies the same bias-correction logic as the model fit.
"""
function predict(model::AllometricModel, data)
  if !Tables.istable(data)
    throw(ArgumentError("data must be a valid table"))
  else
    cols = columntable(data)
  end
  # Remove missing values and prepare the input data for the model
  cols, nonmissings = missing_omit(cols, formula(model).rhs)
  # Generate the model matrix (design matrix) from the input data
  X = modelmatrix(formula(model).rhs, cols)
  # Compute the predicted values: ŷ = X * β
  ŷ = X * coef(model)
  # Handle special cases where the left-hand side (lhs) is a function term
  ft = formula(model).lhs
  if isa(ft, FunctionTerm)
    x = length(ft.args) > 1 ? cols[ft.args[2].sym] : nothing
    # Apply the function-specific prediction logic
    predictbiascorrected!(ŷ, x, ft, model.σ²)
  end
  # Return predictions in the original data order, reinserting missing values
  StatsModels._return_predictions(
    Tables.materializer(data),
    ŷ,
    nonmissings,
    length(nonmissings)
  )
end

"""
    predict!(dest::AbstractVector{<:Real}, model::AllometricModel)

In-place prediction. Stores the result in `dest`.
"""
function predict!(dest::AbstractVector{<:Real}, model::AllometricModel)
  tmp = predict(model)
  resize!(dest, length(tmp))
  copyto!(dest, tmp)
  return dest
end

# ==============================================================================
# RESIDUALS & DEVIANCE
# ==============================================================================

"""
    residuals(model::AllometricModel)

Calculates the raw residuals on the **original scale**: ``y - \\hat{y}``.
"""
residuals(model::AllometricModel) = response(model) - predict(model)

"""
    deviance(model::AllometricModel)

Returns the Sum of Squared Errors (SSE) on the **transformed fitting scale**.
"""
deviance(model::AllometricModel) = model.sse

"""
    nulldeviance(model::AllometricModel)

Returns the Total Sum of Squares (SST) on the **transformed fitting scale**.
"""
nulldeviance(model::AllometricModel) = model.sst

# ==============================================================================
# GOODNESS OF FIT (LIKELIHOOD & R²)
# ==============================================================================

"""
    loglikelihood(model::AllometricModel)

Calculates the log-likelihood of the model assuming Gaussian errors on the transformed scale.
"""
function Distributions.loglikelihood(model::AllometricModel)
  n = nobs(model)
  # uses transformed residuals because Gaussian assumptions hold there
  rss = deviance(model)
  return -n / 2 * (log(2 * π) + 1 + log(rss / n))
end

"""
    nullloglikelihood(model::AllometricModel)

Calculates the log-likelihood of the null model (intercept only) on the transformed scale.
"""
function nullloglikelihood(model::AllometricModel)
  n = nobs(model)
  # uses transformed y for consistency with loglikelihood
  tss = nulldeviance(model)
  return -n / 2 * (log(2 * π) + 1 + log(tss / n))
end

"""
    r2(model::AllometricModel)

Returns the Coefficient of Determination (R²) of the model.
"""
r2(model::AllometricModel) = model.r2

"""
    adjr2(model::AllometricModel)

Returns the Adjusted R² of the model, accounting for the number of predictors.
"""
adjr2(model::AllometricModel) = model.adjr2

"""
    dispersion(model::AllometricModel, sqr::Bool=false)

Returns the dispersion of the model.
- If `sqr=true`, returns the Mean Squared Error (MSE).
- If `sqr=false`, returns the Root Mean Squared Error (RMSE).
"""
dispersion(model::AllometricModel, sqr::Bool=false) = sqr ? model.mse : model.rmse

"""
    cooksdistance(model::AllometricModel)

Calculates Cook's Distance for each observation to identify influential data points.
"""
function StatsBase.cooksdistance(model::AllometricModel)
  u = residuals(model)
  mse = dispersion(model, true)
  k = dof(model) - 1
  X = modelmatrix(model)
  # Efficient calculation of Diagonal of Hat Matrix (leverage)
  # XtX is required to be non-singular (no perfect collinearity)
  XtX = crossmodelmatrix(model)
  k == size(X, 2) || throw(ArgumentError("Models with collinear terms are not currently supported."))
  # hii = diag(H) = diag(X * (X'X)^-1 * X')
  hii = diag(X * inv(XtX) * X')
  D = @. u^2 * (hii / (1 - hii)^2) / (k * mse)
  return D
end

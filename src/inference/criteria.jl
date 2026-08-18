# ==============================================================================
# MODEL SELECTION — criteriaTable / criteriaSelection
# ==============================================================================
# Typed against `RegressionModel` (not `AllometricModel` specifically) so
# this ranking machinery works unchanged on any struct that mirrors
# `AllometricModel`'s tail stat fields exactly (same names), e.g.
# `TaperFit` (`taper/fitting.jl`) — `getfield(model, field)` resolves
# identically for either kind. `regressionGrouped`'s pooled comparison adds
# a `criteriaTable(::GroupedModel, ...)` *method*, defined in
# `fitting/grouped.jl` once `GroupedModel` exists, rather than anything new
# here.

const rankHigherIsBetter = (:r², :adjr², :ev, :normal)
const rankFieldMap = Dict(:r2 => :r², :adjr2 => :adjr², :ev => :ev, :mae => :mae,
  :mape => :mape, :mse => :mse, :rmse => :rmse, :cv => :cv, :normality => :normal)

"""
    calculateScore(models::Vector{<:RegressionModel}, criteria::Vector{Symbol}) -> Vector{Int}

Standard competition ranking ("1224") summed across every criterion — the
composite rank used by [`criteriaTable`](@ref)/[`criteriaSelection`](@ref).
Lower total score is better.
"""
function calculateScore(models::Vector{<:RegressionModel}, criteria::Vector{Symbol})
  total = zeros(Int, length(models))
  for crit in criteria
    field = rankFieldMap[crit]
    values = getfield.(models, field)
    total .+= competerank(values; rev=field in rankHigherIsBetter)
  end
  return total
end

"""
    criteriaTable(models::Vector{<:RegressionModel}, criteria::Symbol...; best::Int=10) -> DataFrame

Rank and filter fitted models by one or more criteria
(`:adjr2`, `:cv`, `:ev`, `:rmse`, `:mae`, `:mape`, default `(:adjr2, :cv, :ev)`).

Passing `:normality` applies a **hard filter**: models whose residuals fail
the normality test are dropped before ranking, and an error is raised if no
model survives.
"""
function criteriaTable(models::Vector{<:RegressionModel}, criteria::Symbol...; best::Int=10)
  allowed = (:r2, :adjr2, :ev, :mae, :mape, :mse, :rmse, :cv, :normality)
  selected = isempty(criteria) ? [:adjr2, :cv, :ev] : collect(criteria)
  issubset(selected, allowed) || throw(ArgumentError("allowed criteria: $allowed"))

  current = models
  if :normality in selected
    current = filter(isNormality, models)
    isempty(current) && error("no model passed the normality test — drop :normality to inspect the non-normal fits")
  end

  scores = calculateScore(current, selected)
  top = sortperm(scores)[1:min(best, length(current))]

  table = DataFrame(rank=scores[top], model=current[top])
  for k in selected
    table[!, k] = getfield.(current[top], rankFieldMap[k])
  end
  return table
end

"""
    criteriaSelection(models::Vector{<:RegressionModel}, criteria::Symbol...)

Same ranking and normality filter as [`criteriaTable`](@ref), returning only
the single best-ranked model.

```julia
best = regression(data, :h, :d) |> m -> criteriaSelection(m, :adjr2, :rmse)
predict(best)
```
"""
function criteriaSelection(models::Vector{<:RegressionModel}, criteria::Symbol...)
  selected = isempty(criteria) ? [:adjr2, :cv, :ev] : collect(criteria)
  current = models
  if :normality in selected
    current = filter(isNormality, models)
    isempty(current) && error("no model passed the normality test")
  end
  return current[argmin(calculateScore(current, selected))]
end

"""
    predict(model, newdata; interval=nothing, level=0.95)

Prediction of a model of the package for the rows of `newdata`, on the original scale and in the
unit of the response. With `interval=:confidence` (the mean) or `:prediction` (a new tree) it is a
`DataFrame` with the columns `prediction`, `lower` and `upper`.

- [`fitglm`](@ref): the prediction of `GLM.jl`; the confidence interval of a GLM comes from the
  interval of the linear predictor through the inverse link, and the prediction interval is
  available for `Normal()` with `IdentityLink()`.
- [`fitnls`](@ref): the interval by the delta method (gradient by `ForwardDiff.jl`),

```math
\\widehat{\\operatorname{Var}}(\\hat\\mu_0) = \\nabla_\\theta f(x_0)^{\\top} \\hat{\\boldsymbol\\Sigma} \\, \\nabla_\\theta f(x_0),
\\qquad \\hat\\mu_0 \\pm t_{n-p} \\sqrt{\\widehat{\\operatorname{Var}}(\\hat\\mu_0) + \\hat\\sigma^2 \\hat\\mu_0^{2\\hat k}}
```

  (the last term only in the prediction interval).
- [`fitmixed`](@ref): the random effects of the plots of the fit, the population curve for the
  others (see [`calibrate`](@ref) for new plots); no interval.

# Example
```julia
predict(model, data)
predict(model, data; interval=:prediction)
```
"""
function predict(model::UnitfulModel, newdata; interval=nothing, level=0.95)
  table = _ustrip(newdata, model.units)
  unit = model.units[model.response]
  if model.model isa MixedModel
    interval === nothing || throw(ArgumentError("Mixed models have no prediction interval."))
    # MixedModels.jl asks for the response column, which does not enter the prediction
    hasproperty(table, model.response) || (table[!, model.response] = zeros(nrow(table)))
    return predict(model.model, table; new_re_levels=:population) .* unit
  end
  interval === nothing && return predict(model.model, table) .* unit
  p = predict(model.model, table; interval, level)
  DataFrame(prediction=p.prediction .* unit, lower=p.lower .* unit, upper=p.upper .* unit)
end

function predict(model::NLSFit, newdata; interval=nothing, level=0.95)
  table = _ustrip(newdata, model.units)
  unit = model.units[model.response]
  X = collect(zip((Float64.(table[!, c]) for c in model.predictors)...))
  μ = [model.f(xi..., model.θ) for xi in X]
  interval === nothing && return μ .* unit
  interval in (:confidence, :prediction) ||
    throw(ArgumentError("interval must be :confidence or :prediction (got :$interval)."))
  t = quantile(TDist(nobs(model) - length(model.θ)), (1 + level) / 2)
  se = map(X, μ) do xi, μi
    g = ForwardDiff.gradient(θ -> model.f(xi..., θ), model.θ)
    s² = g' * model.vcov * g
    interval === :prediction && (s² += (model.σ * abs(μi)^model.k)^2)
    sqrt(s²)
  end
  DataFrame(prediction=μ .* unit, lower=(μ .- t .* se) .* unit, upper=(μ .+ t .* se) .* unit)
end

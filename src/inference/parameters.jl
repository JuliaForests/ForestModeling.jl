# ==============================================================================
# PARAMETER / STATSAPI ACCESSORS
# ==============================================================================
# Unit policy for this section: values that sit directly on the response's
# per-observation scale (`predict`, `fitted`, `response`, `residuals`) carry
# the response's `Unitful` unit, consistent with how the model was fit — see
# `units.jl`. Coefficients and every summary statistic (`rmse`, `mae`, `r2`,
# `dispersion`, ...) stay plain `Float64`, unchanged from before this package
# supported units: a regression coefficient's "unit" depends on an arbitrarily
# transformed predictor (`log(d)`, `d^2`, `1/d`, ...) and has no single clean
# physical meaning, so it is deliberately not modeled here.

nobs(model::AllometricModel) = model.n
dof_residual(model::AllometricModel) = model.ν
dof(model::AllometricModel) = length(model.β) + 1
formula(model::AllometricModel) = model.formula
coef(model::AllometricModel) = model.β
coefnames(model::AllometricModel) = StatsModels.coefnames(model.formula.rhs)
vcov(model::AllometricModel) = model.Σ
stderror(model::AllometricModel) = sqrt.(diag(model.Σ))
r2(model::AllometricModel) = model.r²
adjr2(model::AllometricModel) = model.adjr²
modelmatrix(model::AllometricModel) = modelcols(model.formula.rhs, model.data)
deviance(model::AllometricModel) = model.sse
nulldeviance(model::AllometricModel) = model.sst

"""
    response(model::AllometricModel)

Returns the response vector (y) from the model data, on the original scale
and carrying its `Unitful` unit when the model was fit on unitful data (a
plain `Vector{Float64}` otherwise).
"""
response(model::AllometricModel) = _restoreunit(model.data[1], model.units[1])

"""
    dispersion(model::AllometricModel, sqr::Bool=false)

`sqr=true` returns MSE (original scale); `sqr=false` (default) returns RMSE.
"""
dispersion(model::AllometricModel, sqr::Bool=false) = sqr ? model.mse : model.rmse

"""
    confint(model::AllometricModel; level=0.95)

t-distribution confidence intervals for the coefficients.
"""
function confint(model::AllometricModel; level::Real=0.95)
  h = 1 - (1 - level) / 2
  crit = quantile(TDist(dof_residual(model)), h)
  se = stderror(model)
  return hcat(model.β .- crit .* se, model.β .+ crit .* se)
end

"""
    pvalue(model::AllometricModel)

Two-tailed t-test p-values for `H₀: βᵢ = 0`.
"""
function HypothesisTests.pvalue(model::AllometricModel)
  t = coef(model) ./ stderror(model)
  return 2 .* ccdf.(TDist(dof_residual(model)), abs.(t))
end

"""
    coeftable(model::AllometricModel; level=0.95)

Formatted coefficient table: estimate, std. error, t, p-value, CI bounds.
"""
function coeftable(model::AllometricModel; level::Real=0.95)
  β, se = coef(model), stderror(model)
  t = β ./ se
  dist = TDist(dof_residual(model))
  p = 2 .* ccdf.(dist, abs.(t))
  crit = quantile(dist, 1 - (1 - level) / 2)
  lev = isinteger(level * 100) ? string(Int(level * 100)) : string(level * 100)
  return StatsBase.CoefTable(
    hcat(β, se, t, p, β .- crit .* se, β .+ crit .* se),
    ["Coef.", "Std. Error", "t", "Pr(>|t|)", "Lower $lev%", "Upper $lev%"],
    coefnames(model), 4, 3)
end

"""
    _predictvalues(model::AllometricModel, data) -> Vector{Float64}

Internal numeric core behind [`predict`](@ref): builds the design matrix from
`data` (already unit-matched by the caller, see [`matchunits`](@ref)) and
returns predictions on the response's original scale, bias-corrected when the
response was transformed at fit time — never unit-tagged. Anything that needs
a *plain* per-observation prediction for further arithmetic (e.g.
[`cooksdistance`](@ref), which squares the residual) goes through this
directly instead of the public, unit-aware `predict`.
"""
function _predictvalues(model::AllometricModel, data)
  cols = Tables.istable(data) ? columntable(data) : throw(ArgumentError("data must be a Tables.jl table"))
  cols, kept = missing_omit(cols, formula(model).rhs)
  X = modelmatrix(formula(model).rhs, cols)
  ŷ = X * coef(model)
  yt = formula(model).lhs
  if isa(yt, FunctionTerm)
    x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
    predictBiasCorrected!(ŷ, x, yt, model.σ²)
  end
  return StatsModels._return_predictions(Tables.materializer(data), ŷ, kept, length(kept))
end

"""
    predict(model::AllometricModel, data=model.data)

Predictions on the **original scale**, bias-corrected when the response was
transformed at fit time.

Carries the response's `Unitful` unit when the model was fit on unitful data
— a plain `Vector{Float64}` otherwise, exactly as before this package
supported units. `data` may be given in any unit compatible with what the
model was fit on (e.g. a model fit on `cm` diameters can score a table given
in `inch` or `mm`); a plain number in `data` is assumed to already be
expressed in the fit unit.

# Correction Strategies
**1. Log-Normal Correction (Exact):** ``E[y] = \\exp(\\hat{z} + \\sigma^2/2)``
**2. Delta Method Correction (Approximate):** ``E[y] \\approx g(\\hat{z}) + \\frac{1}{2}\\sigma^2 g''(\\hat{z})``
"""
function predict(model::AllometricModel, data=model.data)
  matched = data === model.data ? data : matchunits(data, model.units)
  return _restoreunit(_predictvalues(model, matched), model.units[1])
end

"""
    fitted(model::AllometricModel)

Alias for `predict(model)`. Returns the fitted values on the original scale.
"""
fitted(model::AllometricModel) = predict(model)

"""
    residuals(model::AllometricModel)

Calculates the raw residuals on the **original scale**: ``y - \\hat{y}``,
unit-tagged consistently with [`response`](@ref)/[`predict`](@ref).
"""
residuals(model::AllometricModel) = response(model) .- predict(model)

"""
    loglikelihood(model::AllometricModel)

Calculates the log-likelihood of the model assuming Gaussian errors on the transformed scale.
"""
function loglikelihood(model::AllometricModel)
  n = nobs(model)
  rss = deviance(model)
  return -n / 2 * (log(2π) + 1 + log(rss / n))
end

"""
    nullloglikelihood(model::AllometricModel)

Calculates the log-likelihood of the null model (intercept only) on the transformed scale.
"""
function nullloglikelihood(model::AllometricModel)
  n = nobs(model)
  tss = nulldeviance(model)
  return -n / 2 * (log(2π) + 1 + log(tss / n))
end

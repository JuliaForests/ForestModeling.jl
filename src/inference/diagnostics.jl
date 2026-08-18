# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

"""
    isNormality(x::AbstractVector{<:Real}) -> Bool

`true` when the null hypothesis of normality cannot be rejected (p > 0.05).
Shapiro–Wilk for n ≤ 5000 (gold standard in that range), Jarque–Bera above it
(Shapiro–Wilk is not valid for large n).
"""
function isNormality(x::AbstractVector{<:Real})
  n = length(x)
  n < 3 && return true
  return n <= 5000 ? pvalue(ShapiroWilkTest(x)) > 0.05 : pvalue(JarqueBeraTest(x)) > 0.05
end
isNormality(model::AllometricModel) = model.normal

"""
    cooksdistance(model::AllometricModel)

Cook's Distance per observation, for spotting influential points. Requires a
full-rank design (no collinear terms). Computed from the plain (unit-free)
scale internally — the residual gets squared here, so unit-tagging it first
would give a dimensionally meaningless result (e.g. `m^2` folded into a
dimensionless influence measure).
"""
function StatsBase.cooksdistance(model::AllometricModel)
  ε = model.data[1] .- _predictvalues(model, model.data)
  X = modelmatrix(model)
  k = dof(model) - 1
  k == size(X, 2) || throw(ArgumentError("collinear terms are not supported"))
  XtX = X'X
  h = diag(X * inv(XtX) * X')
  mse = model.mse
  return @. ε^2 * (h / (1 - h)^2) / (k * mse)
end

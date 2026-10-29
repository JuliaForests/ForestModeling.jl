"""
    calibrate(model, measured, newdata)

Predicts the trees of `newdata` with the random effects of their plots estimated from the trees of
`measured`, the few trees with the response measured in each plot (1 to 10 heights, say). The
fixed effects and the variances of `model` are kept; the random effect of each plot is the one
that best agrees with its measured trees, pulled toward zero by its distribution (the conditional
mode, or empirical BLUP),

```math
\\hat{\\mathbf{b}}_j = \\arg\\max_{\\mathbf{b}} \\left[\\ln p(\\mathbf{y}_j \\mid \\mathbf{b}) + \\ln p(\\mathbf{b})\\right]
```

so a plot with few measured trees stays close to the population curve and one with many follows
its own trees. The plots of `newdata` without measured trees get the population curve. The plots
need not be in the fit: calibration is how a model predicts the heights of a new plot.

It takes a linear mixed model of [`fitmixed`](@ref) with one random-effects term, whose conditional
mode has the closed form

```math
\\hat{\\mathbf{b}}_j = \\boldsymbol\\Lambda\\left(\\boldsymbol\\Lambda^{\\top}\\mathbf{Z}_j^{\\top}\\mathbf{Z}_j\\boldsymbol\\Lambda + \\mathbf{I}\\right)^{-1}
\\boldsymbol\\Lambda^{\\top}\\mathbf{Z}_j^{\\top}\\left(\\mathbf{y}_j - \\mathbf{X}_j\\hat{\\boldsymbol\\beta}\\right),
\\qquad \\boldsymbol\\Psi = \\sigma^2 \\boldsymbol\\Lambda\\boldsymbol\\Lambda^{\\top}
```

# Example
With the mixed model of [`fitmixed`](@ref) and a new plot of which two heights were measured:
```julia
measured = newplot[1:2, :]
calibrate(model, measured, newplot)
```
"""
function calibrate(model::UnitfulModel{<:LinearMixedModel}, measured, newdata)
  lmm = model.model
  length(lmm.reterms) == 1 ||
    throw(ArgumentError("Calibration takes a model with one random-effects term."))
  fixed, random = lmm.formula.rhs[1], only(filter(t -> t isa MixedModels.RandomEffectsTerm, lmm.formula.rhs))
  group = only(fnames(lmm))
  Λ = lmm.reterms[1].λ
  y = model.response
  m = _ustrip(measured, model.units)
  n = _ustrip(newdata, model.units)
  zmatrix(t) = (Z = modelcols(random.lhs, t); Z isa AbstractVector ? reshape(Z, :, 1) : Z)
  Zm, Zn = zmatrix(m), zmatrix(n)
  ŷ = modelcols(fixed, n) * coef(lmm)
  nrow(m) == 0 && return ŷ .* model.units[y]
  r = m[!, y] .- modelcols(fixed, m) * coef(lmm)
  for j in unique(n[!, group])
    rows = findall(isequal(j), m[!, group])
    isempty(rows) && continue
    Z = Zm[rows, :]
    b = Λ * ((Λ' * Z' * Z * Λ + I) \ (Λ' * Z' * r[rows]))
    targets = findall(isequal(j), n[!, group])
    ŷ[targets] .+= Zn[targets, :] * b
  end
  ŷ .* model.units[y]
end

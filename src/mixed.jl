"""
    fitmixed(formula, data)

Linear mixed model fitted by `MixedModels.jl`, with random effects for each plot, stand or site:

```math
y_{ij} = \\mathbf{x}_{ij}^{\\top}\\boldsymbol\\beta + \\mathbf{z}_{ij}^{\\top}\\mathbf{b}_j + \\varepsilon_{ij},
\\qquad \\mathbf{b}_j \\sim N(\\mathbf{0}, \\boldsymbol\\Psi), \\qquad \\varepsilon_{ij} \\sim N(0, \\sigma^2)
```

The random effects ``\\mathbf{b}_j`` shift the curve of each plot ``j`` around the population curve
``\\mathbf{x}^{\\top}\\boldsymbol\\beta``: `(1 | plot)` shifts its intercept, `(1 + inv(d) | plot)` its intercept and
slope. The response is a column, never a function of it, and the columns may be `Unitful`
quantities (see [`fitglm`](@ref)).

[`predict`](@ref) uses the random effects of the plots of the fit and the population curve for the
others; [`calibrate`](@ref) predicts the random effects of new plots from a few measured trees.

# Example
```julia
model = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), data)
```
"""
function fitmixed(formula::FormulaTerm, data)
  response = _response(formula)
  units = _units(data, StatsModels.termvars(formula))
  model = MixedModels.fit(MixedModel, formula, _ustrip(data, units); progress=false)
  UnitfulModel(model, response, units)
end

"""
    fitglm(formula, data; family=Gamma(), link=LogLink())

Generalized linear model fitted by `GLM.jl` (iteratively reweighted least squares) on the original
scale of the response. The link replaces the transformation of `y`, so the prediction needs no
back-transformation and no correction factor:

```math
y_i \\sim \\text{family}(\\mu_i, \\phi), \\qquad g(\\mu_i) = \\mathbf{x}_i^{\\top}\\boldsymbol\\beta
```

With the default `Gamma()` and `LogLink()` the error is multiplicative, its variance grows with the
square of the mean, ``\\operatorname{Var}(y) = \\phi\\mu^2``, and `@formula(v ~ log(d) + log(h))` is the
Schumacher–Hall model ``\\mu = e^{\\beta_0} d^{\\beta_1} h^{\\beta_2}``. `Normal()` with `IdentityLink()` is
the ordinary linear regression (fitted by `lm`, with prediction intervals). The predictors take any transformation (`log(d)`, `1/d`, `d^2`,
`d^2*h`) and a categorical column enters as a fixed effect: `@formula(h ~ log(d) * clone)` gives
each clone its intercept and slope. The response is a column, never a function of it.

The columns may be `Unitful` quantities: the fit uses their values in the unit of each column, and
[`predict`](@ref) and `residuals` give the response in its unit.

# Example
```julia
data = DataFrame(d=[12.0, 18.5, 25.0]u"cm", h=[14.0, 19.0, 23.5]u"m", v=[0.08, 0.24, 0.52]u"m^3")
model = fitglm(@formula(v ~ log(d) + log(h)), data)
predict(model, data)
```
"""
function fitglm(formula::FormulaTerm, data; family=Gamma(), link=LogLink())
  response = _response(formula)
  units = _units(data, StatsModels.termvars(formula))
  table = _ustrip(data, units)
  # the ordinary linear regression as a linear model, the one with prediction intervals in GLM.jl
  model = family isa Normal && link isa IdentityLink ? lm(formula, table) : glm(formula, table, family, link)
  UnitfulModel(model, response, units)
end

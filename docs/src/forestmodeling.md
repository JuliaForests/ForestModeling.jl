```@meta
CurrentModule = ForestModeling
```

# ForestModeling.jl

ForestModeling.jl fits the regression models of forest mensuration — height–diameter, volume,
biomass — with the established Julia package of each method, adding only the few steps they lack
(the units of measurement, the calibration of new plots, the validation by plot). Every model is
fitted and predicted on the **original scale of the response**.

## Why the response is not transformed

The classic way of fitting a power model is to take logarithms, ``\ln v = \beta_0 + \beta_1 \ln d +
\beta_2 \ln h``, fit a linear regression and back-transform, ``\hat v = e^{\hat\beta_0} d^{\hat\beta_1}
h^{\hat\beta_2}``. Two problems come with it:

1. ``e^{\widehat{\ln v}}`` estimates the median of ``v``, not its mean, so the volumes are underestimated
   unless a correction factor multiplies them, and the correction factors can overcompensate,
   mainly in small samples.
2. When the variance is not the one the logarithm assumes, the least squares of the logarithms
   estimate the exponents ``\beta_1, \beta_2`` themselves with a bias.

The package keeps the response as it was measured and puts in the model what the logarithm was
there for, the variance growing with the size of the tree:

- in a generalized linear model ([`fitglm`](@ref)) the **link function** gives the shape,
  ``\ln \mu = \beta_0 + \beta_1 \ln d + \beta_2 \ln h``, and the **Gamma family** the variance,
  ``\operatorname{Var}(v) = \phi\mu^2``;
- in a nonlinear regression ([`fitnls`](@ref)) a **variance function**,
  ``\operatorname{Var}(\varepsilon) = \sigma^2\mu^{2k}``, is estimated with the equation.

The prediction is then the mean on the original scale, with no back-transformation and no
correction factor, and the statistics of the models (RMSE, bias) can be compared directly.

## The methods by structure of the data

### ``y \sim x`` and ``z \sim x + y``: one equation for all the trees

| Method | Function | Package | The model |
|:-------|:---------|:--------|:----------|
| Generalized linear model | [`fitglm`](@ref) | GLM.jl | ``g(\mu) = \mathbf{x}^{\top}\boldsymbol\beta``, any transformation of the predictors |
| Nonlinear regression | [`fitnls`](@ref) | LsqFit.jl, Optim.jl | ``y = f(x; \boldsymbol\theta) + \varepsilon``, any equation |

The generalized linear model is written as a formula, so any combination of transformed predictors
can be tried at the moment it is fitted: `@formula(v ~ log(d) + log(h))` is Schumacher–Hall,
`@formula(v ~ d^2*h)` with `Normal()` is Spurr, `@formula(h ~ inv(d))` with `LogLink()` is
Curtis. The nonlinear regression takes any function: the equations of the package
([`chapmanrichards`](@ref), [`weibull`](@ref), [`prodan`](@ref), [`curtis`](@ref),
[`schumacherhall`](@ref), [`spurr`](@ref), [`power`](@ref)) or one written by the user.

### Categories: species, clones, plots, stands

| Situation | Method | Function |
|:----------|:-------|:---------|
| few categories, all known (clones) | the category as a fixed effect | [`fitglm`](@ref) with `log(d) * clone` |
| many trees in each category | one fit per category | [`fitbygroup`](@ref) |
| many plots with few trees, predictions for new plots | linear mixed model | [`fitmixed`](@ref) |

A **mixed model** gives each plot ``j`` a random effect ``u_j \sim N(0, \sigma_u^2)``, a shift of its curve
around the population curve. The plots of the fit are predicted with their own effect, a plot that
was not in the fit with the population curve, and a new plot with a few measured trees is
**calibrated** ([`calibrate`](@ref)): its random effect is estimated from those trees, pulled
toward zero when they are few,

```math
\hat u_j = \arg\max_u \left[\ln p(\mathbf{y}_j \mid u) + \ln p(u)\right].
```

This is the usual way of predicting the heights of an inventory from a few measured heights per
plot.

### For every model

| Step | Function |
|:-----|:---------|
| prediction with confidence and prediction intervals, in the unit of the response | [`predict`](@ref) |
| cross-validation by plot or stand | [`blockcv`](@ref) |
| RMSE, bias and MAPE on the original scale | [`metrics`](@ref) |
| sandwich covariance, when the variance model is in doubt | [`robustvcov`](@ref) |

The **cross-validation by block** keeps the trees of a plot together: the trees of one plot are more
alike than those of different plots, so a random split predicts each tree with its own plot in the
fit and underestimates the error of a new plot.

## Units of measurement

The columns may be `Unitful` quantities, in any units: the fit uses the values in the unit of each
column, the new data are read in those units (diameters in mm for a model fitted in cm), and the
predictions, the residuals and the RMSE come in the unit of the response. ForestModeling re-exports
ForestFoundations, so `u"cm"`, `ustrip` and the other units are available after
`using ForestModeling`.

## Not included

Methods that would be a package of their own are left out: generalized additive models (GAM), the
simultaneous fit of systems of equations (additive biomass components, taper compatible with
volume), model averaging and Bayesian models (`Turing.jl` can be used directly). The nonlinear
mixed models and the generalized linear mixed models are still under development.

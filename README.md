# ForestModeling

ForestModeling.jl fits the regression models of forest mensuration — height–diameter, volume,
biomass — with the established Julia package of each method, adding only the few steps they lack:
the units of measurement, the calibration of new plots and the validation by plot. It is part of
the JuliaForests ecosystem and re-exports ForestFoundations.

Every model is fitted and predicted on the original scale of the response: the link function of a
generalized linear model, or the variance function of a nonlinear regression, takes the place of
the logarithm of the response, so the predictions need no back-transformation and no correction
factor.

## Functions

| Data | Function | Method | Package |
|:-----|:---------|:-------|:--------|
| `y ~ x`, `z ~ x + y`, fixed categories | `fitglm` | generalized linear model (family + link) | GLM.jl |
| `y ~ x`, `z ~ x + y` | `fitnls` | nonlinear regression with a variance function | LsqFit.jl, Optim.jl |
| plots, stands | `fitmixed` | linear mixed model | MixedModels.jl |
| categories fitted apart | `fitbygroup` | one fit per species, stand or clone | DataFrames.jl |

For every model: `predict` (with confidence and prediction intervals), `calibrate` (the random
effects of a new plot from a few measured trees), `blockcv` and `metrics` (cross-validation by
plot, RMSE, bias, MAPE) and `robustvcov` (sandwich covariance). The equations `chapmanrichards`,
`weibull`, `prodan`, `curtis`, `schumacherhall`, `spurr` and `power` are plain functions for the
nonlinear fits.

## Example

```julia
using ForestModeling, DataFrames

# Schumacher–Hall, with the volume itself as the response
volume = fitglm(@formula(v ~ log(d) + log(h)), data)

# Chapman–Richards with the variance growing with the height
cr(d, θ) = chapmanrichards(d, θ; href=1.3)
height = fitnls(cr, data, :h, :d, [20.0, 0.05, 1.0])
predict(height, data; interval=:prediction)

# a random effect for each plot, calibrated in a new plot from two measured heights
mixed = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), data)
calibrate(mixed, newplot[1:2, :], newplot)

# the error of a new plot
blockcv(t -> fitglm(@formula(v ~ log(d) + log(h)), t), data, :v, :plot)
```

## License

This project is licensed under the MIT License.

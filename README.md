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
plot, RMSE, bias, MAPE), `graybilltest` (the F test of Graybill, predicted against observed),
`identitytest` (whether several clones, stands or species share one equation) and `robustvcov`
(sandwich covariance). The equations `chapmanrichards`, `weibull`, `prodan`, `curtis`,
`schumacherhall`, `spurr`, `honer`, `takata` and `power` are plain functions for the nonlinear fits.

Taper and assortments: the stem taper equations `kozak1969`, `schoepfer1966`, `ormerod1973`,
`demaerschalk1972`, `maxburkhart1976`, `kozak1988`, `kozak2004` and `bi2000`, fitted by `fitnls`,
give the diameter at a height (`taperdiameter`), the height of a diameter (`taperheight`, Roots.jl),
the volume of a section (`taperedvolume`, QuadGK.jl) and the logs of each product
(`logassortment`).

## Example

```julia
julia> using ForestModeling, DataFrames

julia> data = DataFrame(h=[10.2, 15.3, 14.8, 9.7, 16.5, 13.1, 11.6, 12.4, 14.2, 15.0]u"m",
                        d=[20.5, 25.3, 24.1, 18.7, 26.2, 22.5, 19.8, 21.0, 23.4, 24.5]u"cm");

# Curtis as a generalized linear model: the log link, the height itself as the response
julia> fitglm(@formula(h ~ inv(d)), data)
h = exp(4.075 - 33.54 * inv(d))

# the same equation as a nonlinear regression, with the variance growing with the height
julia> model = fitnls(curtis, data, :h, :d, [3.5, -10.0])
h = curtis(d; θ = [4.089, -33.89])

julia> predict(model, data[1:2, :]; interval=:prediction)
2×3 DataFrame
 Row │ prediction  lower      upper
     │ Quantity…   Quantity…  Quantity…
─────┼──────────────────────────────────
   1 │  11.4283 m  9.81524 m  13.0413 m
   2 │   15.638 m  15.1408 m  16.1353 m
```

The tutorial of the documentation fits volume, biomass and height, a mixed model by plot with its
calibration, and validates the models by plot.

## License

This project is licensed under the MIT License.

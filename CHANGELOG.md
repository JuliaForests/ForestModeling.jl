## [Unreleased]

The package was rewritten from the start. It no longer searches catalogs of transformations of the
response nor corrects the bias of their back-transformation: every model is fitted on the original
scale of the response, with the established Julia package of each method.

- `fitglm`: generalized linear models (family + link) by GLM.jl, the ordinary linear regression by `lm`.
- `fitnls`: nonlinear regression with the variance growing with a power of the mean, by
  Levenberg–Marquardt (LsqFit.jl) and maximum likelihood (Optim.jl).
- `fitmixed`: linear mixed models by MixedModels.jl.
- `fitbygroup`: one fit per category, in parallel threads.
- `predict`: on the original scale, in the unit of the response, with confidence and prediction
  intervals (delta method for the nonlinear regression).
- `calibrate`: the random effects of new plots from a few measured trees, for the linear mixed models.
- `blockcv` and `metrics`: cross-validation by plot or stand; RMSE, bias and MAPE.
- `robustvcov`: sandwich covariance.
- The equations `chapmanrichards`, `weibull`, `prodan`, `curtis`, `schumacherhall`, `spurr` and
  `power`; the height of reference of the height–diameter curves is always given (`href`).
- The columns may be `Unitful` quantities; ForestFoundations is re-exported.

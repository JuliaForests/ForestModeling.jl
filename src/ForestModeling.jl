"""
Description:

ForestModeling.jl fits the regression models of forest mensuration — height–diameter, volume,
biomass — on the original scale of the response, with the established Julia package of each
method and only the few steps they lack:

| Data | Function | Method | Package |
|:-----|:---------|:-------|:--------|
| ``y \\sim x``, ``z \\sim x + y``, fixed categorical | [`fitglm`](@ref) | generalized linear model (family + link) | GLM.jl |
| ``y \\sim x``, ``z \\sim x + y`` | [`fitnls`](@ref) | nonlinear regression with a variance function | LsqFit.jl, Optim.jl |
| plots, stands (random) | [`fitmixed`](@ref) | linear mixed model | MixedModels.jl |
| groups fitted apart | [`fitbygroup`](@ref) | one fit per species, stand, clone | DataFrames.jl |

and, for every model, [`predict`](@ref) (with intervals), [`calibrate`](@ref) (random effects of a
new plot from a few measured trees), [`blockcv`](@ref) and [`metrics`](@ref) (cross-validation by
plot, RMSE, bias, MAPE) and [`robustvcov`](@ref) (sandwich covariance). The classic equations
([`chapmanrichards`](@ref), [`weibull`](@ref), [`prodan`](@ref), [`curtis`](@ref),
[`schumacherhall`](@ref), [`spurr`](@ref), [`power`](@ref)) are plain functions used by the
nonlinear fits.

No model transforms the response: the link function (GLM) or the variance function (nonlinear)
takes the place of the logarithm, so the predictions need no back-transformation and no correction
factor. The columns may be `Unitful` quantities; the predictions come in the unit of the response.
"""
module ForestModeling

using DataFrames, Distributions, ForwardDiff, GLM, LinearAlgebra, MixedModels, Optim, Random,
      Reexport, Statistics, StatsModels
using LsqFit: AutoForwardDiff, curve_fit
using CovarianceMatrices: CovarianceMatrices
@reexport using ForestFoundations

import StatsAPI: coef, fitted, nobs, predict, residuals, response, stderror, vcov

include("units.jl")
include("equations.jl")
include("glm.jl")
include("nls.jl")
include("mixed.jl")
include("calibration.jl")
include("grouped.jl")
include("predict.jl")
include("validation.jl")
include("robust.jl")

export chapmanrichards, weibull, prodan, curtis, schumacherhall, spurr, power,
       fitglm, fitnls, fitmixed, fitbygroup, UnitfulModel, NLSFit,
       predict, calibrate, blockcv, metrics, robustvcov,
       coef, vcov, stderror, nobs, residuals,
       @formula, Gamma, Normal, Poisson, IdentityLink, LogLink, InverseLink

end

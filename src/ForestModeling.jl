"""
Description:

ForestModeling.jl is the generic allometric-regression engine behind
`ForestMensuration.jl`: a bounded-transform-catalog search over classic
regression-equation shapes, model selection by composite ranking, and
grouped/stratified-regression variants for hierarchical data, plus stem
taper model fitting (see below). It has no forestry domain knowledge beyond
that — site/age classification and every other forestry-specific
application live in `ForestMensuration.jl`, which consumes the
`AllometricModel`s this package fits.

# Stem taper models

[`TaperModel`](@ref) and its 10 concrete subtypes (`Kozak1969`,
`Schoepfer1966`, `Matte1949`, `Demaerschalk1972`, `Clutter1980`,
`MaxBurkhart1976`, `Johnson1911`, `Kozak1988`, `Kozak2004`, `Bi2000`) are the
one part of this package that *is* forestry-specific: classic published
stem-profile equations, fit with [`fit`](@ref)`(model, dbh, height, hi, di)`
against pooled stem-scaling data into a [`TaperFit`](@ref). Linear forms are
solved in closed form; nonlinear forms use `LsqFit.jl`. The forestry-facing,
`Unitful`-aware application of a `TaperFit` (diameter/height at a point,
integrated volume, log assortment) lives in `ForestMensuration.jl`.

# Units of measurement

Every fitting entry point (`fit`, `regression`, `fitRobust`,
[`regressionGrouped`](@ref)) accepts `Unitful` quantity columns and plain
numbers interchangeably. Units are stripped before fitting (`Unitful`
disallows `log`/`exp` of a dimensioned quantity, which the transform catalog
relies on) and reattached wherever a result sits directly on the response's
scale — `predict`, `fitted`, `response`, `residuals`. Fitting on plain
numbers behaves exactly as if `Unitful` did not exist: every such result
stays a plain `Float64`. Summary statistics and tables (`rmse`, `mae`,
[`criteriaTable`](@ref), [`metrics`](@ref)) are always plain numbers, on the
same numeric scale the model was fit on.
"""
module ForestModeling

using LinearAlgebra, Statistics, StatsBase, StatsModels, GLM, Distributions,
      HypothesisTests, Tables, DataFrames, Combinatorics, Printf, Optim, ForestFoundations, ADTypes, LsqFit
using ForwardDiff   # activates Optim/DifferentiationInterface's ForwardDiff extension for `fitRobust`

import StatsAPI: RegressionModel, coef, coefnames, confint, deviance, dof, dof_residual, fit, fitted,
                  loglikelihood, modelmatrix, nobs, nulldeviance, nullloglikelihood,
                  predict, r2, adjr2, residuals, response, stderror, vcov
import StatsBase: cooksdistance, competerank, CoefTable
import StatsModels: missing_omit
import Tables: columntable
import StatsModels: @formula

# `fit` is imported (not just `using`) specifically so that
# `function fit(::Type{AllometricModel}, ...)` below *extends* the shared
# StatsAPI.fit generic — the same one GLM.jl's own `fit(LinearModel, ...)`
# implements — instead of accidentally shadowing it with a new local method.

include("types.jl")
include("units.jl")
include("formula/transforms.jl")
include("formula/classicequations.jl")
include("fitting/ols.jl")
include("fitting/robust.jl")
include("inference/parameters.jl")
include("inference/diagnostics.jl")
include("inference/criteria.jl")
include("display.jl")
include("fitting/grouped.jl")
include("taper/models.jl")
include("taper/fitting.jl")

export
  @formula,
  AllometricModel, AllometricMethod, OLS, SSE, MAE, HUBER, MSLE, MAPE,
  fit, regression, fitCombinations, fitRobust, fitClassic, classicEquations,
  criteriaTable, criteriaSelection,
  predict, predictBiasCorrected!, fitted, residuals,
  coef, coefnames, coeftable, vcov, stderror, confint, pvalue,
  r2, adjr2, dispersion, cooksdistance, isNormality,
  nobs, dof, dof_residual, formula, modelmatrix, response,
  deviance, nulldeviance, loglikelihood, nullloglikelihood,
  petterson, naslund, prodan, metrics, summary,
  GroupedModel, regressionGrouped, predictBounded,
  TaperModel, TaperFit, ncoef, islinear,
  Kozak1969, Schoepfer1966, Matte1949, Demaerschalk1972, Clutter1980,
  MaxBurkhart1976, Johnson1911, Kozak1988, Kozak2004, Bi2000

end

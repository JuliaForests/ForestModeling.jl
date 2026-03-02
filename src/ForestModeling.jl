module ForestModeling

using Combinatorics, DataFrames, Unitful, StatsModels, StatsBase, Tables, LinearAlgebra, Distributions, HypothesisTests
import Tables: istable, getcolumn

using StatsAPI
import StatsBase: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof, dof_residual,
  loglikelihood, nullloglikelihood, nobs, stderror, vcov,
  residuals, predict, predict!,
  fitted, fit, model_response, response, modelmatrix, r2, r², adjr2, adjr², PValue

import StatsModels: hasintercept, missing_omit, formula, modelmatrix, @formula

export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
  loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
  fitted, fit, fit!, model_response, response, modelmatrix, r2, r², adjr2, adjr²,
  cooksdistance, hasintercept, dispersion, vif, gvif, termnames, @formula

abstract type ForestModel <: RegressionModel end
const S = Union{Symbol,String}
const β0 = InterceptTerm{true}()

struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,I<:Int64} <: RegressionModel
  formula::F
  data::N
  β::Vector{T}
  Σ::Matrix{T}
  σ²::T
  n::I
  ν::I
  p::I
  sse::T
  sst::T
  r2::T
  adjr2::T
  ev::T
  mae::T
  mape::T
  mse::T
  rmse::T
  cv::T
  syx::T
  uncertainty::T
end

import Base: show

_coefnames(model::AllometricModel) = vcat("β0", string.(StatsModels.coefnames(model.formula.rhs.terms[2:end])))

function show(io::IO, model::AllometricModel)
  β = model.β
  n = length(β)
  output = string(StatsModels.coefnames(model.formula.lhs)) * " = $(round(β[1], digits = 4))"
  for i in 2:n
    term = _coefnames(model)[i]
    product = string(round(abs(β[i]), sigdigits=4)) * " * " * term
    output *= signbit(β[i]) ? " - $(product)" : " + $(product)"
  end
  print(io, output)
end

include("fitlinearmodel.jl")
include("selectioncriteria.jl")
include("parameters.jl")

export regression,
  criteriatable,
  bestmodel,
  AllometricModel,
  ContinuousTerm,
  CategoricalTerm

end

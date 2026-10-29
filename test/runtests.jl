using ForestModeling
using CSV
using Test
using DataFrames
using GLM
using LinearAlgebra
using LsqFit
using MixedModels
using Random
using Statistics
using Unitful
import CovarianceMatrices

include("exampledata.jl")

# the values of the columns without units, for the fits of the packages of origin
plain(data) = mapcols(c -> eltype(c) <: Quantity ? ustrip.(c) : c, data)

@testset "ForestModeling.jl" begin
  include("equationstests.jl")
  include("glmtests.jl")
  include("nlstests.jl")
  include("mixedtests.jl")
  include("groupedtests.jl")
  include("validationtests.jl")
  include("assortmenttests.jl")
end

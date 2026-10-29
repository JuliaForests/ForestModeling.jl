using ForestModeling
using Test
using DataFrames
using GLM
using LinearAlgebra
using MixedModels
using Random
using Statistics
using Unitful
import CovarianceMatrices

# Simulated stands with known parameters: 30 plots of 15 trees, the heights from a Chapman–Richards
# curve whose asymptote changes from plot to plot, with an error of 5% of the mean, and the volumes
# from Schumacher–Hall with a multiplicative error
function simulated(; plots=30, trees=15, seed=42)
  rng = MersenneTwister(seed)
  n = plots * trees
  plot = repeat(1:plots, inner=trees)
  u = 3 .* randn(rng, plots)
  d = 5 .+ 35 .* rand(rng, n)
  μ = [chapmanrichards(d[i], [25 + u[plot[i]], 0.07, 1.1]; href=1.3) for i in 1:n]
  h = μ .* (1 .+ 0.05 .* randn(rng, n))
  v = [schumacherhall(d[i], h[i], [5e-5, 1.9, 1.0]) for i in 1:n] .* exp.(0.08 .* randn(rng, n))
  clone = rand(rng, ["A", "B"], n)
  DataFrame(plot=string.(plot), clone=clone, d=d .* u"cm", h=h .* u"m", v=v .* u"m^3")
end

const data = simulated()
cr(d, θ) = chapmanrichards(d, θ; href=1.3)

@testset "ForestModeling.jl" begin
  include("equationstests.jl")
  include("glmtests.jl")
  include("nlstests.jl")
  include("mixedtests.jl")
  include("groupedtests.jl")
  include("validationtests.jl")
end

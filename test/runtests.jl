using ForestModeling
using Test
using DataFrames
using Unitful
import StatsModels: ModelFrame, ContinuousTerm, AbstractTerm, FormulaTerm
using Statistics: mean
using LinearAlgebra: diag

# ==============================================================================
# SHARED FIXTURES
# ==============================================================================
# Deterministic (no RNG) so results are reproducible across Julia/platform
# versions. `D`/`H` is a small hypsometric (diameter/height) sample, the same
# shape used throughout `ForestMensuration`'s own tests.

const D = [10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 26.0, 28.0, 30.0, 32.0]
const H = [10.2, 11.5, 12.3, 14.1, 14.9, 16.5, 17.2, 18.0, 19.6, 21.2, 22.0, 23.1, 24.0, 25.0]

dataPlain() = DataFrame(d=D, h=H)
dataUnitful() = DataFrame(d=D .* u"cm", h=H .* u"m")

# a larger, grouped sample for mixed/grouped regression — 3 "species" with
# distinct slope/intercept so stratifying is actually expected to help.
const GROUP_D = repeat(10.0:3.0:37.0, 3)
const GROUP_SPECIES = repeat(["Oak", "Pine", "Cedar"], inner=10)
const GROUP_SLOPE = Dict("Oak" => 0.85, "Pine" => 0.65, "Cedar" => 0.75)
const GROUP_INTERCEPT = Dict("Oak" => 0.2, "Pine" => 0.5, "Cedar" => 0.35)
const GROUP_H = [GROUP_INTERCEPT[GROUP_SPECIES[i]] + GROUP_SLOPE[GROUP_SPECIES[i]] * log(GROUP_D[i]) + 0.03 * sin(i) for i in 1:30]

groupedDataPlain() = DataFrame(d=GROUP_D, h=GROUP_H, species=GROUP_SPECIES)
groupedDataUnitful() = DataFrame(d=GROUP_D .* u"cm", h=GROUP_H .* u"m", species=GROUP_SPECIES)

@testset "ForestModeling.jl" begin
  include("unitstests.jl")
  include("formula/transformstests.jl")
  include("fitting/olstests.jl")
  include("fitting/groupedtests.jl")
  include("inference/parameterstests.jl")
  include("tapertests.jl")
end

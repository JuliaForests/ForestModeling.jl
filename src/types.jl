# ==============================================================================
# NAMING
# ==============================================================================
# Identifiers in this package avoid "_" as a word separator and prefer Unicode
# math symbols where they read closer to the underlying statistics (β, Σ, σ²,
# ν, Δ, ŷ, ε). Two exceptions are kept on purpose: `dof_residual` and any
# GLM.jl/StatsAPI keyword argument (e.g. `dropcollinear`) belong to the wider
# Julia statistics ecosystem — renaming them would break the generic-function
# interface being extended, so they are left exactly as the ecosystem defines
# them.

const S = Union{Symbol,String}
const TermTuple = Tuple{Vararg{Tuple{AbstractTerm,Vector{Float64}}}}

"""
    const β₀ = InterceptTerm{true}()

Intercept term reused across every model built in this package.
"""
const β₀ = InterceptTerm{true}()

"""
    AllometricModel{F,N,U,T,I,B} <: RegressionModel

A fitted allometric regression model.

Uses a **lazy** design on purpose: the design matrix is not stored, only the
data needed to rebuild it (`data`, `formula`). What *is* stored eagerly, at
fit time, are the goodness-of-fit statistics on the **original scale** of the
response — this is the one thing a generic `GLM.jl` model cannot give us for
free, since it has no notion of "the response was transformed, please
back-correct". Everything else (coefficients, transformed-scale residual
variance) is delegated to `GLM.jl` under the hood; see [`fitOLS`](@ref).

# Fields
- `formula::F`: the fitted `FormulaTerm` (left-hand side may carry a transform).
- `data::N`: the `NamedTuple` of columns used to fit — kept for lazy recomputation.
- `units::U`: `NamedTuple`, same keys as `data`, each value the column's original
  `Unitful.Units` or `nothing` when that column was fit as a plain number. See
  [`stripcolumnunits`](@ref) — this is what lets [`predict`](@ref) hand back
  `Unitful` quantities when the model was fit on unitful data, and plain
  numbers (unchanged from today) otherwise.
- `β::Vector{T}`: estimated coefficients.
- `Σ::Matrix{T}`: coefficient variance–covariance matrix.
- `σ²::T`: residual variance **on the transformed (fitting) scale**.
- `n::I`, `ν::I`, `p::I`: observations, residual degrees of freedom, parameters.
- `sse::T`, `sst::T`: sum of squared errors / total, **on the original scale**.
- `r²::T`, `adjr²::T`: generalized (pseudo) R² and its adjusted form, original scale.
- `ev::T`: explained variance, original scale.
- `mae::T`, `mape::T`, `mse::T`, `rmse::T`, `cv::T`: error metrics, original scale.
- `normal::B`: `true` if residuals (transformed scale) pass a normality test.
"""
struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,U<:NamedTuple,T<:Float64,I<:Int,B<:Bool} <: RegressionModel
  formula::F
  data::N
  units::U
  β::Vector{T}
  Σ::Matrix{T}
  σ²::T
  n::I
  ν::I
  p::I
  sse::T
  sst::T
  r²::T
  adjr²::T
  ev::T
  mae::T
  mape::T
  mse::T
  rmse::T
  cv::T
  normal::B
end

"""
    @enum AllometricMethod OLS SSE MAE HUBER MSLE MAPE

The estimation criterion used by [`fitRobust`](@ref). `OLS` is closed-form
(delegated to `GLM.jl`); every other option is minimized numerically and is
meant for the cases where OLS assumptions are visibly violated (heavy
outliers, strongly relative error) — not the default path.
"""
@enum AllometricMethod OLS SSE MAE HUBER MSLE MAPE

# ==============================================================================
# TAPER MODEL FITTING
# ==============================================================================

"""
    TaperFit{M<:TaperModel,T<:Float64} <: RegressionModel

A fitted stem taper model. Carries the same tail goodness-of-fit fields as
[`AllometricModel`](@ref) (`r²`, `adjr²`, `ev`, `mae`, `mape`, `mse`, `rmse`,
`cv`, `normal`), computed by the same shared [`computeStatistics`](@ref) —
so a `Vector{TaperFit}` plugs directly into [`criteriaTable`](@ref)/
[`criteriaSelection`](@ref) unchanged, to pick the best-fitting taper form for
a dataset.

# Fields
- `model::M`: the [`TaperModel`](@ref) instance that was fit.
- `β::Vector{T}`: estimated coefficients.
- `n::Int`, `ν::Int`, `p::Int`: observations, residual degrees of freedom, parameters.
- `fitted::Vector{T}`, `residuals::Vector{T}`: on the relative-diameter (`dr = d/dbh`) scale.
- `sse::T`, `sst::T`: sum of squared errors / total, relative-diameter scale.
- `r²::T`, `adjr²::T`, `ev::T`, `mae::T`, `mape::T`, `mse::T`, `rmse::T`, `cv::T`: fit statistics.
- `normal::Bool`: `true` if residuals pass a normality test.
"""
struct TaperFit{M<:TaperModel,T<:Float64} <: RegressionModel
  model::M
  β::Vector{T}
  n::Int
  ν::Int
  p::Int
  fitted::Vector{T}
  residuals::Vector{T}
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
  normal::Bool
end

"""
    _fitnonlinear(model::TaperModel, hi, ht, dbh, dr) -> Vector{Float64}

Nonlinear least-squares fit via `LsqFit.curve_fit`, starting from
`_initialcoef(model, hi, ht, dbh, dr)`. `LsqFit` is a lightweight
Levenberg–Marquardt solver, not a modeling framework — the design (design
matrix, statistics, `TaperFit`) stays entirely hand-rolled in this package,
matching how `fitRobust` uses `Optim.jl` for exactly the same reason.
"""
function _fitnonlinear(model::TaperModel, hi, ht, dbh, dr)
  β0 = _initialcoef(model, hi, ht, dbh, dr)
  X = hcat(hi, ht, dbh)   # `LsqFit.curve_fit` health-checks xdata elementwise with `isfinite`,
                          # which rules out a Vector{Tuple}; a plain n×3 matrix works instead.
  modelfun(X, β) = [_taperratio(model, row[1], row[2], row[3], β) for row in eachrow(X)]
  result = LsqFit.curve_fit(modelfun, X, dr, β0)
  return result.param
end

"""
    fit(model::TaperModel, dbh::AbstractVector{<:Real}, height::AbstractVector{<:Real},
        hi::AbstractVector{<:Real}, di::AbstractVector{<:Real}) -> TaperFit

Fit a stem taper `model` against pooled `(hi, di)` stem-scaling observations,
each paired with the originating tree's breast-height diameter `dbh` and
total `height` — the standard multi-tree taper-fitting layout (one row per
measured stem section, `dbh`/`height` repeated within a tree).

Dispatches on [`islinear`](@ref)`(model)`: linear forms (`Kozak1969`,
`Schoepfer1966`, `Matte1949`) are solved in closed form via
`designmatrix(model, hi, height, dbh) \\ dr`; every other form is solved
nonlinearly (see [`_fitnonlinear`](@ref)).

# Arguments
- `model::TaperModel`: which taper equation to fit, e.g. `Kozak2004()`.
- `dbh::AbstractVector{<:Real}`: breast-height diameter per observation.
- `height::AbstractVector{<:Real}`: total tree height per observation.
- `hi::AbstractVector{<:Real}`: the height along the stem each `di` was measured at.
- `di::AbstractVector{<:Real}`: the measured diameter at `hi`.

All four vectors must be the same length and expressed in a consistent unit
(this package carries no forestry unit convention of its own — see the module
docstring); `dbh`/`height`/`hi`/`di` are typically cm/m/m/cm.

# Returns
A [`TaperFit`](@ref).

# Examples
```julia-repl
julia> dbh  = [20.0, 20.0, 20.0, 30.0, 30.0, 30.0];

julia> height = [18.0, 18.0, 18.0, 22.0, 22.0, 22.0];

julia> hi   = [0.3, 6.0, 14.0, 0.3, 8.0, 18.0];

julia> di   = [22.4, 15.8, 6.1, 33.6, 24.2, 8.7];

julia> ft = fit(Kozak1969(), dbh, height, hi, di);

julia> round.(coef(ft); digits=4)
3-element Vector{Float64}:
  1.0658
 -0.3129
 -0.4429
```
"""
function fit(model::TaperModel, dbh::AbstractVector{<:Real}, height::AbstractVector{<:Real},
  hi::AbstractVector{<:Real}, di::AbstractVector{<:Real})

  length(dbh) == length(height) == length(hi) == length(di) ||
    throw(DimensionMismatch("dbh, height, hi and di must all have the same length."))

  dr = di ./ dbh
  n = length(dr)
  p = ncoef(model)
  n > p || throw(ArgumentError("need more observations ($n) than coefficients ($p) to fit $(typeof(model))."))
  ν = n - p

  β = islinear(model) ?
      designmatrix(model, hi, height, dbh) \ dr :
      _fitnonlinear(model, hi, height, dbh, dr)

  fitted = [_taperratio(model, hi[i], height[i], dbh[i], β) for i in eachindex(dr)]
  residuals = dr .- fitted

  ȳ = mean(dr)
  sst = sum(abs2, dr .- ȳ)
  stats = computeStatistics(dr, fitted, residuals, n, ν, sst, ȳ)

  return TaperFit(model, β, n, ν, p, fitted, residuals, stats.sse, sst, stats.r², stats.adjr²,
    stats.ev, stats.mae, stats.mape, stats.mse, stats.rmse, stats.cv, stats.normal)
end

# ==============================================================================
# ACCESSORS
# ==============================================================================

coef(fit::TaperFit) = fit.β
nobs(fit::TaperFit) = fit.n
dof_residual(fit::TaperFit) = fit.ν
dof(fit::TaperFit) = fit.p + 1
r2(fit::TaperFit) = fit.r²
adjr2(fit::TaperFit) = fit.adjr²
deviance(fit::TaperFit) = fit.sse
nulldeviance(fit::TaperFit) = fit.sst

"""
    fitted(fit::TaperFit) -> Vector{Float64}

Fitted relative diameters (`dr = d/dbh`) at the training observations.
"""
fitted(fit::TaperFit) = fit.fitted

"""
    residuals(fit::TaperFit) -> Vector{Float64}

Training residuals on the relative-diameter (`dr = d/dbh`) scale.
"""
residuals(fit::TaperFit) = fit.residuals

"""
    dispersion(fit::TaperFit, sqr::Bool=false)

`sqr=true` returns MSE; `sqr=false` (default) returns RMSE, both on the
relative-diameter scale.
"""
dispersion(fit::TaperFit, sqr::Bool=false) = sqr ? fit.mse : fit.rmse

"""
    predict(fit::TaperFit, dbh::Real, height::Real, hi::Real) -> Real
    predict(fit::TaperFit, dbh::Real, height::Real, hi::AbstractVector{<:Real}) -> Vector

Diameter at height(s) `hi`, for a tree with the given `dbh`/total `height`,
evaluated from the fitted taper curve (`dbh * dr(hi/height, coef(fit))`) — the
absolute-scale counterpart of [`fitted`](@ref)/[`residuals`](@ref), which stay
on the relative-diameter scale the model was fit on. `dbh`, `height` and `hi`
must be expressed in the same unit the model was fit on.
"""
predict(fit::TaperFit, dbh::Real, height::Real, hi::Real) =
  dbh * _taperratio(fit.model, hi, height, dbh, fit.β)
predict(fit::TaperFit, dbh::Real, height::Real, hi::AbstractVector{<:Real}) =
  [predict(fit, dbh, height, h) for h in hi]

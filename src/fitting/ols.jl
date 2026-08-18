# ==============================================================================
# FITTING KERNEL
# ==============================================================================

"""
    fitOLS(X::AbstractMatrix, y::AbstractVector) -> (β, Σ, σ², sse)

Thin wrapper over `GLM.fit(LinearModel, X, y)`. This is the **single-fit**
path: no factorization is reused across calls, so there is no reason to
hand-roll the linear algebra — `GLM.jl` already does exactly this, tested and
maintained outside this package.
"""
function fitOLS(X::AbstractMatrix, y::AbstractVector)
  m = GLM.fit(LinearModel, X, y)
  β = coef(m)
  σ² = deviance(m) / dof_residual(m)
  Σ = vcov(m)
  return β, Σ, σ², deviance(m)
end

"""
    predictBiasCorrected!(ŷ, x, ft::FunctionTerm, σ²)

In-place. Overwrites transformed-scale predictions `ŷ` with bias-corrected
predictions on the **original scale**, given the residual variance `σ²` of
the transformed fit. Uses the exact log-normal correction for `log`, and the
second-order Delta Method for every other supported transform. `x` is only
needed for the two-argument forestry forms (`naslund`, `prodan`).

If `ft` is not one of the supported names, `ŷ` is left untouched.
"""
function predictBiasCorrected!(ŷ::Vector{<:Real}, x::Union{AbstractVector{<:Real},Nothing}, ft::FunctionTerm, σ²::Real)
  name = nameof(ft.f)
  name ∉ (:log, :inv, :sqrt, :petterson, :naslund, :prodan) && return ŷ
  @inbounds @simd for i in eachindex(ŷ)
    z = ŷ[i]
    if name == :log
      ŷ[i] = exp(z + σ² / 2)                       # exact log-normal correction
    elseif name == :inv
      ŷ[i] = inv(z) + σ² * (2 / z^3) / 2            # Δ-method, g(z)=1/z
    elseif name == :sqrt
      ŷ[i] = z^2 + σ²                               # Δ-method, g(z)=z²
    elseif name == :petterson
      z² = z^2
      ŷ[i] = inv(z²) + σ² * (6 / z²^2) / 2          # g(z)=1/z²
    elseif name == :naslund
      x² = x[i]^2
      z² = z^2
      ŷ[i] = x² / z² + σ² * (6x² / z²^2) / 2
    elseif name == :prodan
      x² = x[i]^2
      ŷ[i] = x² / z + σ² * (2x² / z^3) / 2
    end
  end
  return ŷ
end

"""
    computeStatistics(yᵣ, ŷ, ε, n, ν, sst, ȳ) -> NamedTuple

Shared post-processing step for **every** fitting entry point in this package
(`fit`, `fitCombinations`, `fitRobust`, and the pooled goodness-of-fit in
`regressionGrouped`'s `criteriaTable` alike). This
is what used to be duplicated between the single-model and combinatorial
paths — collapsing it here means there is exactly one place that defines what
"R²" or "CV%" means for a fitted model in this package.
"""
function computeStatistics(yᵣ::AbstractVector, ŷ::AbstractVector, ε::AbstractVector, n::Int, ν::Int, sst::Real, ȳ::Real)
  sse = ε ⋅ ε
  mse = sse / ν
  r² = 1 - sse / sst
  adjr² = 1 - (1 - r²) * (n - 1) / ν
  ev = 1 - var(ε) / var(yᵣ)
  mae = mean(abs, ε)
  mape = mean(abs.(ε) ./ abs.(yᵣ)) * 100
  rmse = √mse
  cv = rmse / ȳ * 100
  normal = isNormality(ε)
  return (; sse, mse, r², adjr², ev, mae, mape, rmse, cv, normal)
end

"""
    fit(::Type{AllometricModel}, formula::FormulaTerm, data; contrasts=Dict(), nonNegative::Bool=true)

Fit a single allometric model with `formula` (e.g. `@formula(log(h) ~ log(d))`).

Automatically back-transforms and bias-corrects every reported statistic to
the **original scale** of the response — see [`predictBiasCorrected!`](@ref).
Only left-hand-side transforms with a defined correction are accepted:
`log`, `inv`, `sqrt`, `petterson`, `naslund`, `prodan`.

`data` may freely mix `Unitful.Quantity` columns and plain numbers — see
[`stripcolumnunits`](@ref). The response's unit (if any) is remembered on the
returned model and reattached by [`predict`](@ref).
"""
function fit(::Type{AllometricModel}, formula::FormulaTerm, data; contrasts=Dict{Symbol,Any}(), nonNegative::Bool=true)
  data, columnUnits = stripcolumnunits(data)
  mf = ModelFrame(formula, data; model=AllometricModel, contrasts=contrasts)
  f = mf.f
  yt = f.lhs
  if isa(yt, FunctionTerm)
    name = nameof(yt.f)
    name ∉ (:log, :inv, :sqrt, :petterson, :naslund, :prodan) && throw(ArgumentError(
      "transform :$name on the response has no defined bias correction; " *
      "supported: (:log, :inv, :sqrt, :petterson, :naslund, :prodan)"))
  end

  cols = mf.data
  units = columnUnits[keys(cols)]
  yᵣ = cols[1]
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)

  X = modelmatrix(mf)
  y = modelcols(yt, cols)
  n, p = size(X)
  ν = n - p

  β, Σ, σ², _ = fitOLS(X, y)
  ŷ = X * β
  ε = y .- ŷ

  if isa(yt, FunctionTerm)
    x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
    predictBiasCorrected!(ŷ, x, yt, σ²)
    ε = yᵣ .- ŷ
  end
  nonNegative && any(<(0), ŷ) && throw(ErrorException("fitted values contain negative predictions"))

  stats = computeStatistics(yᵣ, ŷ, ε, n, ν, sst, ȳ)
  return AllometricModel(f, cols, units, β, Σ, σ², n, ν, p, stats.sse, sst, stats.r², stats.adjr²,
    stats.ev, stats.mae, stats.mape, stats.mse, stats.rmse, stats.cv, stats.normal)
end

"""
    fitCombinations(cols, units, yList, combinations, qTerms, nonNegative) -> Vector{AllometricModel}

Internal engine behind [`regression`](@ref). For each candidate `x`
combination, the design matrix is built and its Cholesky factor of `X'X` is
computed **once**, then reused to solve for every candidate `y` transform in
`yList` — this reuse-across-`y` is the one optimization `GLM.jl`'s per-call
API cannot give us (each call to `GLM.fit` would refactor `X'X` from
scratch), so it stays hand-rolled on purpose. Runs across `x` combinations
with `Threads.@threads`.
"""
function fitCombinations(cols::NamedTuple, units::NamedTuple, yList::Vector{Tuple{AbstractTerm,Vector{Float64}}},
  combinations::Vector{TermTuple}, qTerms::Vector{<:AbstractTerm}, nonNegative::Bool)

  X₀ = modelcols(β₀, cols)
  yᵣ = cols[1]
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)

  hasQ = !isempty(qTerms)
  qSum = hasQ ? sum(qTerms) : nothing
  qMatrix = hasQ ? modelmatrix(qSum, cols) : nothing

  nx, ny = length(combinations), length(yList)
  models = Matrix{Union{Missing,AllometricModel}}(undef, ny, nx)

  Threads.@threads for ix in 1:nx
    c = combinations[ix]
    X = hasQ ? hcat(X₀, map(last, c)..., qMatrix) : hcat(X₀, map(last, c)...)
    rhs = hasQ ? MatrixTerm(mapfoldl(first, +, c; init=β₀) + qSum) : MatrixTerm(mapfoldl(first, +, c; init=β₀))
    n, p = size(X)
    ν = n - p

    try
      chol = cholesky!(Symmetric(X'X))
      β = Vector{Float64}(undef, p)
      ŷ = Vector{Float64}(undef, n)

      for iy in 1:ny
        yt, y = yList[iy]
        mul!(β, X', y)
        ldiv!(chol, β)                 # reuses the same factorization for every y
        mul!(ŷ, X, β)
        ε = y .- ŷ
        σ² = (ε ⋅ ε) / ν               # transformed scale — computed once, reused for both the
                                        # bias correction below and the value stored on the model;
                                        # recomputing it from post-correction (original-scale) ε would
                                        # silently mis-scale every future predictBiasCorrected! call

        if isa(yt, FunctionTerm)
          x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
          predictBiasCorrected!(ŷ, x, yt, σ²)
          ε = yᵣ .- ŷ
        end
        if nonNegative && any(<(0), ŷ)
          models[iy, ix] = missing
          continue
        end

        Σ = rmul!(copy(inv(chol)), σ²)
        stats = computeStatistics(yᵣ, ŷ, ε, n, ν, sst, ȳ)
        models[iy, ix] = AllometricModel(FormulaTerm(yt, rhs), cols, units, copy(β), Σ, σ², n, ν, p,
          stats.sse, sst, stats.r², stats.adjr², stats.ev, stats.mae, stats.mape, stats.mse,
          stats.rmse, stats.cv, stats.normal)
      end
    catch
      models[:, ix] .= missing
    end
  end

  fitted = collect(skipmissing(vec(models)))
  isempty(fitted) && error("fitCombinations: every candidate model failed to fit")
  return fitted
end

"""
    regression(data, yName::Symbol, xNames::Symbol...; nMin=1, nMax=2, nonNegative=true, contrasts=Dict())

Fit every combination of the bounded transform catalog (see
[`expandXTerms`](@ref)/[`transformYTerms`](@ref)) between `nMin` and `nMax`
regressor terms, and return the full `Vector{AllometricModel}`. Pass the
result to [`criteriaTable`](@ref) or [`criteriaSelection`](@ref) to rank it.

Any `xNames` column that the schema infers as categorical (a `String`,
`Symbol`, or `CategoricalArray` column) is treated as a qualitative
covariate, always included alongside every continuous combination — this is
also how [`regressionGrouped`](@ref) builds its pooled "qualy" model, by
simply appending the grouping symbols to `xNames`.

# Example
```julia
models = regression(data, :h, :d; nMax=2)
best = criteriaSelection(models, :adjr2, :cv)
```
"""
function regression(data, yName::Symbol, xNames::Symbol...; nMin::Int=1, nMax::Int=2,
  nonNegative::Bool=true, contrasts=Dict{Symbol,Any}())

  nMax > 5 && throw(ArgumentError("nMax is capped at 5 — the search space is meant to stay bounded"))
  xNames = Symbol.(xNames)

  data, columnUnits = stripcolumnunits(data)
  mf = ModelFrame(term(yName) ~ sum(term.(xNames)), data; model=AllometricModel, contrasts=contrasts)
  cols = mf.data
  units = columnUnits[keys(cols)]
  yTerm = mf.f.lhs
  isa(yTerm, CategoricalTerm) && throw(ArgumentError("response must be continuous"))

  xTerms = collect(filter(t -> t isa ContinuousTerm, mf.f.rhs.terms))
  isempty(xTerms) && throw(ArgumentError("no continuous regressor supplied"))
  qTerms = collect(filter(t -> t isa CategoricalTerm, mf.f.rhs.terms))

  termGroups = Vector{Vector{Tuple{AbstractTerm,Vector{Float64}}}}()
  for x in xTerms
    subgroup = Tuple{AbstractTerm,Vector{Float64}}[]
    for t in expandXTerms(x)
      try
        col = modelcols(t, cols)
        all(isfinite, col) && push!(subgroup, (t, col))
      catch
      end
    end
    !isempty(subgroup) && push!(termGroups, subgroup)
  end
  isempty(termGroups) && error("no valid x transform could be generated")

  combinations = length(termGroups) == 1 ?
    TermTuple[Tuple(c) for c in powerset(termGroups[1], nMin, nMax)] :
    TermTuple[Tuple(g) for g in Iterators.product(termGroups...)] |> vec

  yList = Tuple{AbstractTerm,Vector{Float64}}[]
  for t in transformYTerms(yTerm, xTerms)
    try
      v = modelcols(t, cols)
      all(isfinite, v) && push!(yList, (t, v))
    catch
    end
  end

  return fitCombinations(cols, units, yList, combinations, qTerms, nonNegative)
end

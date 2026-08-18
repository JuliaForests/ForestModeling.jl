# ==============================================================================
# MIXED-EFFECTS REGRESSION (MixedModels.jl-backed)
# ==============================================================================
# Built on `MixedModels.fit(MixedModel, formula, data)` — the stable, public
# entry point — rather than hand-assembling `ReMat`/`LinearMixedModel`
# internals, which is both simpler and less exposed to breakage across
# MixedModels.jl versions. The one thing `MixedModels.jl` cannot give us for
# free (same story as `GLM.jl` in `fitting/ols.jl`) is back-transforming a
# transformed response — `predictBiasCorrected!` is reused unchanged for
# that, it only cares about `(ŷ, x, ft::FunctionTerm, σ²)`, never which
# numeric engine produced them.
#
# `MixedAllometricModel` deliberately mirrors `AllometricModel`'s *tail*
# fields (σ², n, ν, p, sse, sst, r², adjr², ev, mae, mape, mse, rmse, cv,
# normal) exactly, so `criteriaTable`/`criteriaSelection`/`calculateScore`
# (typed against `RegressionModel`) work on `Vector{MixedAllometricModel}`
# with no changes at all.

"""
    MixedAllometricModel{F,N,U,T,I,B} <: RegressionModel

A fitted mixed-effects allometric regression model — the `MixedModels.jl`
counterpart to [`AllometricModel`](@ref). Wraps a `LinearMixedModel` plus the
same original-scale goodness-of-fit statistics and unit bookkeeping as
`AllometricModel`; see its docstring for what each statistic means.

# Fields
- `model::LinearMixedModel`: the underlying `MixedModels.jl` fit.
- `formula::F`, `data::N`, `units::U`: as in [`AllometricModel`](@ref).
- `σ²::T`: residual variance **on the transformed (fitting) scale**.
- `n`, `ν`, `p`: observations, residual degrees of freedom (`MixedModels.jl`'s
  own accounting, which is not simply `n - p`), and fixed-effect parameters.
- `sse`, `sst`, `r²`, `adjr²`, `ev`, `mae`, `mape`, `mse`, `rmse`, `cv`,
  `normal`: as in [`AllometricModel`](@ref), all on the **original scale**.
"""
struct MixedAllometricModel{F<:FormulaTerm,N<:NamedTuple,U<:NamedTuple,T<:Float64,I<:Int,B<:Bool} <: RegressionModel
  model::LinearMixedModel{T}
  formula::F
  data::N
  units::U
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
    randomEffectsTerm(g::Symbol) -> RandomEffectsTerm

A simple random intercept for grouping variable `g`, i.e. the `(1 | g)`
clause of a `MixedModels.jl` formula, built the same way `@formula` itself
would (`term(1) | term(g)`) but without needing macro-literal syntax — this
package builds formulas programmatically from symbols and reused catalog
terms, so this stays a plain function call.
"""
randomEffectsTerm(g::Symbol) = term(1) | term(g)

function _validateResponseTransform(yt::AbstractTerm)
  isa(yt, FunctionTerm) || return nothing
  name = nameof(yt.f)
  name ∉ (:log, :inv, :sqrt, :petterson, :naslund, :prodan) && throw(ArgumentError(
    "transform :$name on the response has no defined bias correction; " *
    "supported: (:log, :inv, :sqrt, :petterson, :naslund, :prodan)"))
  return nothing
end

_responseSymbol(yt::AbstractTerm) = isa(yt, FunctionTerm) ? yt.args[1].sym : yt.sym

"""
    wrapMixed(m::LinearMixedModel, formula, cols, units, nonNegative) -> MixedAllometricModel

Shared post-processing behind [`fitMixed`](@ref)/[`regressionMixed`](@ref):
back-transforms and bias-corrects `m`'s fitted values to the response's
original scale (mirrors [`computeStatistics`](@ref)'s role for the OLS
path), and packages the result. `cols` must already have the response as its
first key (matching the `AllometricModel`/`model.data[1]` convention used
throughout this package).
"""
function wrapMixed(m::LinearMixedModel, formula::FormulaTerm, cols::NamedTuple, units::NamedTuple, nonNegative::Bool)
  yᵣ = cols[1]
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)

  n = nobs(m)
  ν = Int(dof_residual(m))
  p = length(fixef(m))

  ŷ = Vector{Float64}(fitted(m))
  yTransformed = Vector{Float64}(response(m))
  ε = yTransformed .- ŷ
  σ² = (ε ⋅ ε) / ν

  yt = formula.lhs
  if isa(yt, FunctionTerm)
    x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
    predictBiasCorrected!(ŷ, x, yt, σ²)
    ε = yᵣ .- ŷ
  end
  nonNegative && any(<(0), ŷ) && throw(ErrorException("fitted values contain negative predictions"))

  stats = computeStatistics(yᵣ, ŷ, ε, n, ν, sst, ȳ)
  return MixedAllometricModel(m, formula, cols, units, σ², n, ν, p, stats.sse, sst, stats.r², stats.adjr²,
    stats.ev, stats.mae, stats.mape, stats.mse, stats.rmse, stats.cv, stats.normal)
end

"""
    fitMixed(formula::FormulaTerm, data, groupNames; contrasts=Dict(), nonNegative=true)

Fit a single mixed-effects allometric model: `formula` (e.g.
`@formula(log(h) ~ 1 + log(d))`) supplies the fixed-effects part, and
`groupNames` (a `Symbol` or vector of `Symbol`s) each contribute a random
intercept — see [`randomEffectsTerm`](@ref). Bias-corrects back to the
response's original scale exactly like [`fit`](@ref); accepts and reattaches
`Unitful` units the same way too.

```julia
m = fitMixed(@formula(log(h) ~ 1 + log(d)), data, :plot)
predict(m)
```
"""
function fitMixed(formula::FormulaTerm, data, groupNames::Union{Symbol,AbstractVector{Symbol}};
  contrasts=Dict{Symbol,Any}(), nonNegative::Bool=true)

  yt = formula.lhs
  _validateResponseTransform(yt)
  yName = _responseSymbol(yt)

  data, columnUnits = stripcolumnunits(data)
  groups = groupNames isa Symbol ? Symbol[groupNames] : collect(groupNames)
  isempty(groups) && throw(ArgumentError("at least one grouping variable is required"))
  randomRhs = sum(randomEffectsTerm(g) for g in groups)
  fFull = FormulaTerm(yt, formula.rhs + randomRhs)

  m = MixedModels.fit(MixedModel, fFull, data; contrasts=contrasts)

  allCols = Tables.columntable(data)
  orderedKeys = (yName, Tuple(k for k in keys(allCols) if k != yName)...)
  cols = NamedTuple{orderedKeys}(map(k -> allCols[k], orderedKeys))
  units = columnUnits[orderedKeys]

  return wrapMixed(m, fFull, cols, units, nonNegative)
end

"""
    regressionMixed(data, yName::Symbol, xName::Symbol, groupNames; qNames=Symbol[], nMin=1, nMax=1, nonNegative=true, contrasts=Dict())

Combinatorial search over the shared [`expandXTerms`](@ref)/
[`transformYTerms`](@ref) catalog for a **single** predictor `xName`, fit as
a mixed-effects model with a random intercept per `groupNames` held fixed
across every candidate — mirrors [`regression`](@ref)'s shape, but each
candidate is fit with `MixedModels.fit!`, which is iterative rather than
closed-form. `nMax` defaults to `1` (not `2`, unlike `regression`) precisely
because of that cost — raise it deliberately, not by habit.

Returns a `Vector{MixedAllometricModel}`; rank it with
[`criteriaTable`](@ref)/[`criteriaSelection`](@ref) exactly like `regression`'s
output.
"""
function regressionMixed(data, yName::Symbol, xName::Symbol, groupNames::Union{Symbol,AbstractVector{Symbol}};
  qNames::AbstractVector{Symbol}=Symbol[], nMin::Int=1, nMax::Int=1,
  nonNegative::Bool=true, contrasts=Dict{Symbol,Any}())

  nMax > 5 && throw(ArgumentError("nMax is capped at 5 — the search space is meant to stay bounded"))
  groups = groupNames isa Symbol ? Symbol[groupNames] : collect(groupNames)
  isempty(groups) && throw(ArgumentError("at least one grouping variable is required"))

  data, columnUnits = stripcolumnunits(data)
  # `groups` is included in this ModelFrame too (not just y/x/q) purely so its rows stay
  # aligned with the rest under `missing_omit` — it is explicitly excluded from `qTerms`
  # below so it never doubles as a fixed-effect covariate.
  mf = ModelFrame(term(yName) ~ sum(term.((xName, qNames..., groups...))), data; model=AllometricModel, contrasts=contrasts)
  cols = mf.data
  units = columnUnits[keys(cols)]
  yTerm = mf.f.lhs
  isa(yTerm, CategoricalTerm) && throw(ArgumentError("response must be continuous"))

  xTerm = only(filter(t -> t isa ContinuousTerm, mf.f.rhs.terms))
  qTerms = collect(filter(t -> t isa CategoricalTerm && t.sym ∉ groups, mf.f.rhs.terms))
  randomRhs = sum(randomEffectsTerm(g) for g in groups)

  xCandidates = AbstractTerm[]
  for t in expandXTerms(xTerm)
    try
      col = modelcols(t, cols)
      all(isfinite, col) && push!(xCandidates, t)
    catch
    end
  end
  isempty(xCandidates) && error("no valid x transform could be generated")

  yCandidates = AbstractTerm[]
  for t in transformYTerms(yTerm, xCandidates)
    try
      v = modelcols(t, cols)
      all(isfinite, v) && push!(yCandidates, t)
    catch
    end
  end
  isempty(yCandidates) && error("no valid y transform could be generated")

  models = MixedAllometricModel[]
  for xset in powerset(xCandidates, nMin, min(nMax, length(xCandidates)))
    fixedRhs = mapfoldl(identity, +, xset; init=β₀)
    hasQ = !isempty(qTerms)
    hasQ && (fixedRhs += sum(qTerms))
    for yt in yCandidates
      try
        f = FormulaTerm(yt, MatrixTerm(fixedRhs) + randomRhs)
        m = MixedModels.fit(MixedModel, f, cols; contrasts=contrasts)
        push!(models, wrapMixed(m, f, cols, units, nonNegative))
      catch
      end
    end
  end
  isempty(models) && error("regressionMixed: every candidate model failed to fit")
  return models
end

# ---- accessors: delegate to MixedModels.jl's own StatsAPI methods where they
# already do exactly the right thing; use this package's own stored fields
# where the semantics need to match AllometricModel's (dof/dof_residual,
# every original-scale statistic, unit handling). Coefficient-level inference
# (confint/pvalue/coeftable) and Cook's distance are not provided for mixed
# models in this pass — mixed-model inference is a materially deeper topic
# than the fixed-effects case and out of scope here.

coef(model::MixedAllometricModel) = fixef(model.model)
coefnames(model::MixedAllometricModel) = coefnames(model.model)
vcov(model::MixedAllometricModel) = vcov(model.model)
stderror(model::MixedAllometricModel) = stderror(model.model)
modelmatrix(model::MixedAllometricModel) = modelmatrix(model.model)
nobs(model::MixedAllometricModel) = model.n
dof(model::MixedAllometricModel) = model.p + 1
dof_residual(model::MixedAllometricModel) = model.ν
formula(model::MixedAllometricModel) = model.formula
r2(model::MixedAllometricModel) = model.r²
adjr2(model::MixedAllometricModel) = model.adjr²
dispersion(model::MixedAllometricModel, sqr::Bool=false) = sqr ? model.mse : model.rmse
isNormality(model::MixedAllometricModel) = model.normal
deviance(model::MixedAllometricModel) = model.sse
nulldeviance(model::MixedAllometricModel) = model.sst

"""
    response(model::MixedAllometricModel)

Returns the response vector (y), carrying its `Unitful` unit when the model
was fit on unitful data (a plain `Vector{Float64}` otherwise).
"""
response(model::MixedAllometricModel) = _restoreunit(model.data[1], model.units[1])

"""
    predict(model::MixedAllometricModel, data=model.data)

Predictions on the response's original scale, bias-corrected exactly like
[`predict(::AllometricModel)`](@ref). Delegates to `MixedModels.jl`'s own
`predict` with `new_re_levels=:population`: for a group level seen at fit
time this is identical to the model's best (BLUP) prediction for that group
— the kwarg only changes behavior for a genuinely new group, where it
falls back to the population-average (fixed-effects-only) prediction instead
of `missing`.
"""
function predict(model::MixedAllometricModel, data=model.data)
  matched = data === model.data ? data : matchunits(data, model.units)
  ŷ = Vector{Float64}(MixedModels.predict(model.model, matched; new_re_levels=:population))
  yt = model.formula.lhs
  if isa(yt, FunctionTerm)
    cols = Tables.istable(matched) ? Tables.columntable(matched) : throw(ArgumentError("data must be a Tables.jl table"))
    x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
    predictBiasCorrected!(ŷ, x, yt, model.σ²)
  end
  return _restoreunit(ŷ, model.units[1])
end
fitted(model::MixedAllometricModel) = predict(model)
residuals(model::MixedAllometricModel) = response(model) .- predict(model)

Base.show(io::IO, model::MixedAllometricModel) = show(io, model.model)

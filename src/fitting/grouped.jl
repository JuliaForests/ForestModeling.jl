# ==============================================================================
# GROUPED / STRATIFIED REGRESSION
# ==============================================================================
# Answers a recurring forestry-biometrics question: for a grouping variable
# (species, region, stratum, ...), is it worth fitting a *separate* equation
# per group, or does one shared equation (optionally with the group as a
# covariate) already do about as well? `regressionGrouped` fits all three
# strategies so the analyst can compare them directly:
#   - `general`: one pooled equation, the group is not used at all.
#   - `qualy`:   one pooled equation with the group as an extra categorical
#                covariate — just `regression(data, y, x, groupNames...)`,
#                since the engine already auto-detects categorical columns
#                from the schema (no special-casing needed here).
#   - `grouped`: one independently-selected equation per group level, via
#                `DataFrames.groupby` (not a plotting-recipe package's
#                internals, as an earlier draft of this reached for).

"""
    GroupedModel{S<:AllometricModel}

Bundles the three regression strategies compared by [`regressionGrouped`](@ref).

# Fields
- `general::S`: pooled model, `groupNames` not used.
- `qualy::S`: pooled model with `groupNames` included as categorical covariates.
- `grouped::Dict{String,S}`: one independently best-selected model per unique
  combination of `groupNames` levels, keyed by that combination joined with `"_"`.
- `groupNames::Vector{Symbol}`: the grouping variables used.
- `xName::Symbol`: the continuous predictor searched over — recorded so
  [`predictBounded`](@ref) knows which column to range-check without
  re-deriving it from a (possibly transformed) fitted formula term.
"""
struct GroupedModel{S<:AllometricModel}
  general::S
  qualy::S
  grouped::Dict{String,S}
  groupNames::Vector{Symbol}
  xName::Symbol
end

"""
    regressionGrouped(data, yName::Symbol, xName::Symbol, groupNames; qNames=Symbol[], nMin=1, nMax=2, criteria=Symbol[], nonNegative=true, contrasts=Dict())

Fits [`GroupedModel`](@ref)'s three strategies and returns them bundled.
`criteria` (forwarded to [`criteriaSelection`](@ref) for every strategy) and
`qNames` (extra categorical covariates, alongside `groupNames`, for `qualy`
and every per-group fit) both default to the same defaults `regression`
itself uses. A group whose own subset fails to fit (usually: too few rows
for `nMax`) is skipped rather than failing the whole call — mirrors
`fitCombinations`'s "skip what fails, error only if nothing survives"
behavior.

```julia
gm = regressionGrouped(data, :h, :d, :species)
criteriaTable(gm, :adjr2, :cv)                      # general vs qualy vs grouped, pooled
criteriaTable(collect(values(gm.grouped)), :adjr2)  # per-group breakdown (existing method)
predictBounded(gm, newData)                         # range-safe prediction, see predictBounded
```
"""
function regressionGrouped(data, yName::Symbol, xName::Symbol, groupNames::Union{Symbol,AbstractVector{Symbol}};
  qNames::AbstractVector{Symbol}=Symbol[], nMin::Int=1, nMax::Int=2, criteria::AbstractVector{Symbol}=Symbol[],
  nonNegative::Bool=true, contrasts=Dict{Symbol,Any}())

  groups = groupNames isa Symbol ? Symbol[groupNames] : collect(groupNames)
  isempty(groups) && throw(ArgumentError("at least one grouping variable is required"))

  general = criteriaSelection(regression(data, yName, xName; nMin, nMax, nonNegative, contrasts), criteria...)
  qualy = criteriaSelection(regression(data, yName, xName, groups..., qNames...; nMin, nMax, nonNegative, contrasts), criteria...)

  cleanData = dropmissing(DataFrame(Tables.columntable(data)), unique([yName, xName, groups..., qNames...]))
  grouped = Dict{String,AllometricModel}()
  for (key, subframe) in pairs(groupby(cleanData, groups))
    label = join(string.(values(NamedTuple(key))), "_")
    try
      grouped[label] = criteriaSelection(regression(subframe, yName, xName, qNames...; nMin, nMax, nonNegative, contrasts), criteria...)
    catch
    end
  end
  isempty(grouped) && error("regressionGrouped: every group failed to fit")

  return GroupedModel(general, qualy, grouped, groups, xName)
end

"""
    predictBounded(model::GroupedModel, data::AbstractDataFrame)

Range-safe prediction across [`GroupedModel`](@ref)'s three strategies,
choosing a *different, broader* model per row rather than letting a
small group's own equation extrapolate into nonsensical territory
(negative heights, thousands-scale volumes). For each row:

1. If the predictor (`model.xName`) falls outside the **global** range (the
   range `model.general` was fit on), predict with `model.general`.
2. Otherwise, look up the row's own group in `model.grouped`. If no model
   exists for that group, predict with `model.general`.
3. If the group has its own model but the predictor falls outside **that
   group's own** fitted range, predict with `model.qualy` instead.
4. Only when the predictor is within the group's own fitted range does its
   individually-selected model get used.

A `missing` predictor value predicts `missing`, no fallback attempted.

Ported from a prior, independently-developed hierarchical-fallback design
(not a novel `predict` variant this package invented) — a distinct name from
`predict` is deliberate: unlike every other `predict` method in this package,
this one can silently substitute a different fitted model than the one you
might expect from the row's own group.

# Examples
```julia
gm = regressionGrouped(data, :h, :d, :species)
predictBounded(gm, newData)
```
"""
function predictBounded(model::GroupedModel, data::AbstractDataFrame)
  n = nrow(data)
  xGlobal = model.general.data[model.xName]
  xMin, xMax = extrema(xGlobal)

  out = Vector{Any}(missing, n)
  for i in 1:n
    x = data[i, model.xName]
    ismissing(x) && continue

    if x < xMin || x > xMax
      out[i] = predict(model.general, data[i:i, :])[1]
      continue
    end

    key = join(string.(collect(data[i, model.groupNames])), "_")
    if haskey(model.grouped, key)
      m = model.grouped[key]
      gMin, gMax = extrema(m.data[model.xName])
      out[i] = (x < gMin || x > gMax) ? predict(model.qualy, data[i:i, :])[1] : predict(m, data[i:i, :])[1]
    else
      out[i] = predict(model.general, data[i:i, :])[1]
    end
  end
  return identity.(out)   # narrows Vector{Any} to the tightest common type (e.g. Vector{Union{Missing,Float64}})
end

"""
    criteriaTable(model::GroupedModel, criteria::Symbol...)

Pooled goodness-of-fit comparison between the three strategies bundled in
`model` — a 3-row `DataFrame` (`General`, `Qualy`, `Grouped`) using the same
metrics and definitions as [`computeStatistics`](@ref) uses everywhere else
in this package. The `Grouped` row treats the union of every group model's
own predictions as a single pooled prediction vector against the pooled
response, with pooled residual degrees of freedom `n - Σpᵢ` (total
coefficients across every group model) — this is what makes it comparable
to `general`/`qualy` on equal footing.

For a per-group breakdown instead, call
`criteriaTable(collect(values(model.grouped)), criteria...)` — the existing
`Vector{<:RegressionModel}` method, unchanged.
"""
function criteriaTable(model::GroupedModel, criteria::Symbol...)
  selected = isempty(criteria) ? [:adjr2, :cv, :ev] : collect(criteria)
  allowed = (:r2, :adjr2, :ev, :mae, :mape, :mse, :rmse, :cv)
  issubset(selected, allowed) || throw(ArgumentError("allowed criteria for a GroupedModel comparison: $allowed"))

  groupModels = collect(values(model.grouped))
  n = sum(m -> m.n, groupModels)
  p = sum(m -> length(coef(m)), groupModels)
  ν = n - p
  ν <= 0 && throw(ArgumentError("pooled residual degrees of freedom are non-positive ($ν) — too many groups relative to observations"))

  # plain/stripped scale throughout: computeStatistics runs a normality test
  # downstream, which (like every other numeric routine in this package)
  # is not Unitful-aware — see `_predictvalues`/`cooksdistance` for the same
  # boundary rule applied elsewhere.
  yPooled = reduce(vcat, (m.data[1] for m in groupModels))
  ŷPooled = reduce(vcat, (_predictvalues(m, m.data) for m in groupModels))
  ε = yPooled .- ŷPooled
  ȳ = mean(yPooled)
  sst = sum(abs2, yPooled .- ȳ)
  grouped = computeStatistics(yPooled, ŷPooled, ε, n, ν, sst, ȳ)

  table = DataFrame(type=["General", "Qualy", "Grouped"], model=[model.general, model.qualy, missing])
  for k in selected
    field = rankFieldMap[k]
    table[!, k] = [getfield(model.general, field), getfield(model.qualy, field), grouped[field]]
  end
  return table
end

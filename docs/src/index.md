```@meta
CurrentModule = ForestModeling
```

# ForestModeling

Documentation for [ForestModeling](https://github.com/JuliaForests/ForestModeling.jl) —
the generic allometric-regression engine behind
[ForestMensuration.jl](https://github.com/JuliaForests/ForestMensuration.jl).

## Installation

```julia-repl
using Pkg
Pkg.add("ForestModeling")
```

## Quickstart

```@example quickstart
using ForestModeling, DataFrames

data = DataFrame(d=[10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0],
                  h=[10.2, 11.5, 12.3, 14.1, 14.9, 16.5, 17.2, 18.0, 19.6, 21.2])

models = regression(data, :h, :d; nMax=2)
best = criteriaSelection(models, :adjr2, :cv)
```

```@example quickstart
predict(best)
```

The examples below walk through every exported function, grouped by what it does.

## Fitting a single formula (`fit`)

```@example single
using ForestModeling, DataFrames

data = DataFrame(d=[10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 26.0, 28.0, 30.0, 32.0],
                  h=[10.2, 11.5, 12.3, 14.1, 14.9, 16.5, 17.2, 18.0, 19.6, 21.2, 22.0, 23.1, 24.0, 25.0])

m = fit(AllometricModel, @formula(log(h) ~ log(d)), data)
```

## Inspecting a fitted model

Every [`AllometricModel`](@ref) implements the standard `StatsAPI`/`StatsBase` interface,
all on the response's **original scale** (bias-corrected back-transform — see
[`predictBiasCorrected!`](@ref)) unless noted otherwise.

```@example single
coef(m), coefnames(m)
```

```@example single
stderror(m)
```

```@example single
confint(m)   # 95% CI, one row per coefficient
```

```@example single
pvalue(m)
```

```@example single
coeftable(m)
```

```@example single
r2(m), adjr2(m), dispersion(m)   # R², adjusted R², RMSE — all original scale
```

```@example single
cooksdistance(m)   # influence per observation; needs a full-rank design
```

```@example single
isNormality(m)   # Shapiro–Wilk (n≤5000) / Jarque–Bera on the transformed-scale residuals
```

```@example single
nobs(m), dof(m), dof_residual(m)
```

```@example single
formula(m)
```

```@example single
size(modelmatrix(m))
```

```@example single
response(m)
```

```@example single
deviance(m), nulldeviance(m)   # SSE / SST, original scale
```

```@example single
loglikelihood(m), nullloglikelihood(m)   # transformed scale, Gaussian errors
```

```@example single
predict(m)
```

```@example single
fitted(m) == predict(m)
```

```@example single
residuals(m)
```

```@example single
metrics(m)   # NamedTuple of every goodness-of-fit statistic — what criteriaTable is built from
```

```@example single
summary(m)
```

## Model selection (`criteriaTable` / `criteriaSelection`)

```@example single
models = regression(data, :h, :d; nMax=2)
length(models)
```

```@example single
criteriaTable(models, :adjr2, :cv; best=5)
```

```@example single
criteriaSelection(models, :adjr2, :cv)
```

Passing `:normality` applies a hard filter — models whose residuals fail the normality
test are dropped before ranking:

```@example single
criteriaTable(models, :adjr2, :normality; best=3)
```

## Robust / alternative estimation (`fitRobust`)

For the cases where OLS assumptions are visibly violated — heavy outliers, or strongly
relative (percentage-scale) error — [`fitRobust`](@ref) minimizes an alternative loss with
`Optim.jl` instead of closed-form OLS, starting from the OLS solution. `SSE` reproduces the
closed-form fit; `MAE`/`HUBER` down-weight outliers; `MSLE`/`MAPE` target relative error.

```@example single
[fitRobust(@formula(log(h) ~ log(d)), data, method) for method in (SSE, MAE, HUBER, MSLE, MAPE)]
```

```@example single
mRobust = fitRobust(@formula(log(h) ~ log(d)), data, HUBER)
predict(mRobust)
```

`fitRobust` is opt-in — [`regression`](@ref)/[`fitCombinations`](@ref) never call it, since
its optimizer makes it far more expensive to run combinatorially. The returned `Σ` is
**not** a robust/sandwich covariance matrix, so treat `confint`/`pvalue` on a non-`SSE` fit
as provisional.

## Named classic equations (`fitClassic`)

Reproduce a textbook hypsometric equation in one line, fit through the exact same engine
(and bias correction) as everything else in this package.

```@example single
collect(keys(classicEquations))
```

```@example single
fitClassic(:curtis, :h, :d, data)   # Curtis, R.O. (1967). Forest Science 13(4):365-375
```

```@example single
fitClassic(:naslund, :h, :d, data)   # Näslund
```

```@example single
mProdan2 = fitClassic(:prodan2, :h, :d, data)
predict(mProdan2) .+ 1.3   # :prodan2's response is h - 1.3 (height above breast height) — see note below
```

!!! note
    `:prodan2`'s response is `prodan(h - 1.3, d)` — Prodan's original linearization is
    stated in terms of height *above breast height* — so its `predict`/`fitted` return
    `h - 1.3`, not total height; add `1.3` back to recover `h`, as above. Every other
    entry's `predict` is already on the total-height scale.

## Units of measurement

Every fitting entry point ([`fit`](@ref), [`regression`](@ref), [`fitRobust`](@ref),
[`regressionGrouped`](@ref)) accepts `Unitful` quantity columns directly — `Units`/
`Quantity`/`@u_str` and the default-unit constants come from
[ForestFoundations.jl](https://github.com/JuliaForests/ForestFoundations.jl), not a
direct `Unitful` dependency of this package. Units are stripped before fitting —
`Unitful` disallows `log`/`exp` of a dimensioned quantity, which the transform catalog
relies on throughout — and reattached wherever a result sits on the response's own scale:
`predict`, `fitted`, `response`, `residuals`. Fitting on plain numbers behaves exactly as
if `Unitful` did not exist: every such result stays a plain `Float64`.

```@example units
using ForestModeling, DataFrames, Unitful

dataU = DataFrame(d=[10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 26.0, 28.0, 30.0, 32.0]u"cm",
                   h=[10.2, 11.5, 12.3, 14.1, 14.9, 16.5, 17.2, 18.0, 19.6, 21.2, 22.0, 23.1, 24.0, 25.0]u"m")

mUnit = fit(AllometricModel, @formula(log(h) ~ log(d)), dataU)
predict(mUnit)
```

```@example units
# predicting from data in a different (but compatible) unit converts automatically
predict(mUnit, DataFrame(d=[300.0]u"mm"))
```

```@example units
response(mUnit), residuals(mUnit)
```

`regression` and `fitRobust` accept `Unitful` columns exactly the same way — nothing about
the call changes, only `predict`'s return type does:

```@example units
bestU = criteriaSelection(regression(dataU, :h, :d; nMax=2), :adjr2, :cv)
predict(bestU)
```

```@example units
predict(fitRobust(@formula(log(h) ~ log(d)), dataU, HUBER))
```

Summary statistics and tables (`rmse`, `mae`, `criteriaTable`, `metrics`, `siteTable`) are
always plain numbers, on the same numeric scale the model was fit on — a regression
coefficient built from an arbitrarily transformed predictor (`log(d)`, `d^2`, `1/d`, ...)
has no single clean physical unit, so this package does not attempt to invent one.

## Grouped/stratified regression

[`regressionGrouped`](@ref) fits three strategies for a grouping variable so you can tell
whether stratifying is actually worth it: one pooled equation (`general`), one pooled
equation with the group as a categorical covariate (`qualy`), and one independently
selected equation per group (`grouped`).

```@example grouped
using ForestModeling, DataFrames

# 3 species, distinct slope/intercept each
d = repeat(10.0:3.0:37.0, 3)
species = repeat(["Oak", "Pine", "Cedar"], inner=10)
slope = Dict("Oak" => 0.85, "Pine" => 0.65, "Cedar" => 0.75)
intercept = Dict("Oak" => 0.2, "Pine" => 0.5, "Cedar" => 0.35)
h = [intercept[species[i]] + slope[species[i]] * log(d[i]) + 0.03 * sin(i) for i in 1:30]
gdata = DataFrame(d=d, h=h, species=species)

gm = regressionGrouped(gdata, :h, :d, :species)
gm.general
```

```@example grouped
gm.qualy
```

```@example grouped
gm.grouped   # one independently-selected model per species, keyed by group label
```

```@example grouped
criteriaTable(gm, :adjr2, :cv)   # general vs qualy vs grouped, pooled goodness-of-fit
```

```@example grouped
criteriaTable(collect(values(gm.grouped)), :adjr2)   # per-group breakdown instead
```

Here stratifying by species clearly pays for itself: `Grouped` (a separate equation per
species) and `Qualy` (one shared equation with species as a covariate) both dominate
`General` (one pooled equation).

### Range-safe prediction (`predictBounded`)

Calling [`predict`](@ref) directly on a small group's own equation risks wild
extrapolation the moment a new row's `d` falls outside that group's own fitted range.
[`predictBounded`](@ref) checks the range per row instead — out of the **global** range
→ `general`; in the global range but out of that row's **own group's** range (or no model
for that group at all) → `qualy`; only otherwise does the group's own model get used:

```@example grouped
newTrees = DataFrame(d=[15.0, 60.0, 15.0], species=["Cedar", "Cedar", "Birch"])
predictBounded(gm, newTrees)
```

## Stem taper models

The one part of this package that *is* forestry-specific: 10 classic published stem
taper (profile) equations — `Kozak1969`, `Schoepfer1966`, `Matte1949`,
`Demaerschalk1972`, `Clutter1980`, `MaxBurkhart1976`, `Johnson1911`, `Kozak1988`,
`Kozak2004`, `Bi2000` — fit against pooled stem-scaling data (each tree's `dbh`/total
`height` repeated once per measured section):

```@example taper
using ForestModeling

dbh    = [20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 25.0, 25.0, 25.0, 35.0, 35.0, 35.0]
height = [18.0, 18.0, 18.0, 22.0, 22.0, 22.0, 20.0, 20.0, 20.0, 24.0, 24.0, 24.0]
hi     = [0.3, 6.0, 14.0, 0.3, 8.0, 18.0, 0.3, 7.0, 16.0, 0.3, 9.0, 20.0]
di     = [22.4, 15.8, 6.1, 33.6, 24.2, 8.7, 27.9, 18.6, 7.4, 38.9, 27.1, 9.9]

ft = fit(Kozak1969(), dbh, height, hi, di)
coef(ft)
```

Linear forms (`Kozak1969`, `Schoepfer1966`, `Matte1949`) solve in closed form; every other
form uses `LsqFit.jl`'s Levenberg–Marquardt solver. Either way, a [`TaperFit`](@ref) carries
the same tail goodness-of-fit fields as [`AllometricModel`](@ref):

```@example taper
r2(ft), adjr2(ft), dispersion(ft)
```

```@example taper
nobs(ft), dof(ft), dof_residual(ft)
```

`predict` evaluates the fitted curve at an absolute height, scaled by `dbh` — the
`Unitful`-aware, forestry-facing wrapper (`taperdiameter`/`taperheight`/`taperedvolume`/
`logassortment`) lives in `ForestMensuration.jl`:

```@example taper
predict(ft, 25.0, 20.0, 7.0)   # diameter at 7 m, for a 25 cm dbh / 20 m tall tree
```

Fitting several models on the same data and ranking them works exactly like
`criteriaTable`/`criteriaSelection` do for `AllometricModel` — a `Vector{TaperFit}` plugs
in unchanged:

```@example taper
fits = [fit(m, dbh, height, hi, di) for m in (Kozak1969(), Schoepfer1966(), Demaerschalk1972(), Bi2000())]
criteriaTable(fits, :adjr2, :rmse)
```

## Index

```@index
```

## API Reference

```@autodocs
Modules = [ForestModeling]
```

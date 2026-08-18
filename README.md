# ForestModeling

ForestModeling.jl is the generic allometric-regression engine behind
[ForestMensuration.jl](https://github.com/JuliaForests/ForestMensuration.jl): a
bounded-transform-catalog search over classic regression-equation shapes, model
selection by composite ranking, and grouped/stratified-regression variants for
hierarchical data, plus stem taper model fitting. It has no forestry domain knowledge
beyond that — site/age classification and every other forestry-specific application
live in `ForestMensuration.jl`, which consumes the models this package fits, and every
default-unit convention comes from
[ForestFoundations.jl](https://github.com/JuliaForests/ForestFoundations.jl), shared
across the `JuliaForests` ecosystem.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaForests.github.io/ForestModeling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaForests.github.io/ForestModeling.jl/dev/)
[![Build Status](https://github.com/JuliaForests/ForestModeling.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaForests/ForestModeling.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaForests/ForestModeling.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaForests/ForestModeling.jl)

## Installation

Install the package via Julia's package manager:

```julia-repl
using Pkg
Pkg.add("ForestModeling")
```

## Overview

- **Bounded transform search (`regression`)**: fits every combination, within a search
  size you control, of a small citable catalog of predictor/response transforms (log,
  square root, inverse, and the named Curtis/Petterson/Näslund/Prodan forestry
  linearizations) and ranks the results by composite score across one or more criteria
  (`criteriaTable`/`criteriaSelection`).
- **Single fits (`fit`) and named classic equations (`fitClassic`)**: fit one formula
  directly — `@formula(log(h) ~ log(d))` — or reproduce a textbook equation
  (`:curtis`, `:henriksen`, `:petterson`, ...) in one line.
- **Automatic bias correction**: every response transform this package supports
  (`log`, `inv`, `sqrt`, and the forestry-specific `petterson`/`naslund`/`prodan`) is
  back-transformed to the original scale with the appropriate correction (exact
  log-normal, or second-order Delta method) — `predict` always returns values on the
  scale you actually measured, not the fitting scale.
- **Robust/alternative estimation (`fitRobust`)**: opt-in `SSE`/`MAE`/`HUBER`/`MSLE`/
  `MAPE` loss minimization via `Optim.jl`, for when OLS assumptions are visibly violated.
- **Grouped/stratified regression (`regressionGrouped`)**: compares one pooled equation,
  one pooled equation with the group as a covariate, and fully separate equations per
  group — so you can tell whether stratifying by species/region/stratum is actually
  worth it. **`predictBounded`** scores new data across the three, per row, falling back
  to the broader model whenever a row's predictor falls outside the range its own
  group's equation was actually fit on — so a small subgroup can't extrapolate into
  nonsensical predictions.
- **Stem taper models (`fit(::TaperModel, ...)`)**: the one forestry-specific part of this
  package — 10 classic published taper (stem-profile) equations (`Kozak1969`,
  `Schoepfer1966`, `Matte1949`, `Demaerschalk1972`, `Clutter1980`, `MaxBurkhart1976`,
  `Johnson1911`, `Kozak1988`, `Kozak2004`, `Bi2000`), fit against pooled stem-scaling data
  and rankable with the same `criteriaTable`/`criteriaSelection`.

Every entry point above accepts `Unitful` quantity columns and plain numbers
interchangeably — see [Units of measurement](#units-of-measurement) below.

## Quickstart

```julia-repl
using ForestModeling, DataFrames

julia> data = DataFrame(d=[10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0],
                         h=[10.2, 11.5, 12.3, 14.1, 14.9, 16.5, 17.2, 18.0, 19.6, 21.2]);

# search the transform catalog and rank by adjusted R² and coefficient of variation
julia> models = regression(data, :h, :d; nMax=2);

julia> best = criteriaSelection(models, :adjr2, :cv)
h ^ -1 = 0.0582 - 3.338e-5 * d ^ 2 + 4.861 * d ^ -2

julia> summary(best)
Allometric Regression Model
--------------------------------------------------
Equation: h ^ -1 = 0.0582 - 3.338e-5 * d ^ 2 + 4.861 * d ^ -2

Coefficients:
...
Goodness-of-Fit (original scale):
  R² (generalized): 0.9949
  Adjusted R²:      0.9935
  ...

julia> predict(best)
10-element Vector{Union{Missing, Float64}}:
 10.147829089308829
 11.483629283168705
  ⋮
 21.130327937569657
```

## Inspecting a fitted model

Every `AllometricModel` implements the standard `StatsAPI`/`StatsBase` interface, all on the
response's **original scale** (see [Automatic bias correction](#overview) above) except
where noted:

```julia-repl
julia> m = fit(AllometricModel, @formula(log(h) ~ log(d)), data);

julia> coef(m)
2-element Vector{Float64}:
 0.39737866774301006
 0.8233376284585446

julia> coefnames(m)
2-element Vector{String}:
 "(Intercept)"
 "log(d)"

julia> stderror(m)
2-element Vector{Float64}:
 0.05188856924563663
 0.017350732950505775

julia> confint(m)   # 95% CI, one row per coefficient
2×2 Matrix{Float64}:
 0.284323  0.510434
 0.785534  0.861142

julia> pvalue(m)
2-element Vector{Float64}:
 5.8613859277694276e-6
 5.016687515960227e-15

julia> coeftable(m)
──────────────────────────────────────────────────────────────
              Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────
(Intercept)  0.3974      0.0519   7.658    <1e-04     0.2843     0.5104
log(d)       0.8233      0.0174  47.453    <1e-10     0.7855     0.8611
──────────────────────────────────────────────────────────────

julia> r2(m), adjr2(m), dispersion(m)   # R², adjusted R², RMSE — all original scale
(0.9927226355656846, 0.9921161885294917, 0.4284443803253042)

julia> cooksdistance(m)   # influence per observation; needs a full-rank design
14-element Vector{Float64}:
 0.0243566250387583
 ⋮
 0.6918701553549762

julia> isNormality(m)   # Shapiro–Wilk (n≤5000) / Jarque–Bera on the transformed-scale residuals
true

julia> nobs(m), dof(m), dof_residual(m)
(14, 3, 12)

julia> deviance(m), nulldeviance(m)      # SSE / SST, original scale
(2.202775044388007, 302.6885714285715)

julia> loglikelihood(m), nullloglikelihood(m)   # transformed scale, Gaussian errors
(-6.919763802664194, -41.38066937668904)

julia> fitted(m) == predict(m), residuals(m) == response(m) .- predict(m)
(true, true)

julia> metrics(m)   # NamedTuple of every goodness-of-fit statistic — what criteriaTable is built from
(equation = ["log(h) = 0.3974 + 0.8233 * log(d)"], r² = [0.9927226355656846], ...)

julia> summary(m)
Allometric Regression Model
--------------------------------------------------
Equation: log(h) = 0.3974 + 0.8233 * log(d)

Coefficients:
...
Goodness-of-Fit (original scale):
  R² (generalized): 0.9927
  Adjusted R²:      0.9921
  ...
Diagnostics:
  Normality (transformed-scale residuals): pass
--------------------------------------------------
```

## Robust / alternative estimation (`fitRobust`)

For the cases where OLS assumptions are visibly violated — heavy outliers, or strongly
relative (percentage-scale) error — `fitRobust` minimizes an alternative loss with
`Optim.jl` instead of closed-form OLS, starting from the OLS solution:

```julia-repl
julia> fitRobust(@formula(log(h) ~ log(d)), data, HUBER)   # SSE, MAE, HUBER, MSLE, MAPE
log(h) = 0.3935 + 0.8245 * log(d)

julia> predict(ans)[1:3]
3-element Vector{Union{Missing, Float64}}:
 10.303914389795407
 11.503198530320947
 12.676391564976024
```

`SSE` reproduces the closed-form OLS fit exactly (`fitRobust(..., SSE) == fit(...)` up to
solver tolerance); `MAE`/`HUBER` down-weight outliers, `MSLE`/`MAPE` target relative rather
than absolute error. `fitRobust` is opt-in — `regression`/`fitCombinations` never call it,
since its optimizer makes it far more expensive to run combinatorially. The returned `Σ`
is **not** a robust/sandwich covariance matrix, so treat `confint`/`pvalue` on a non-`SSE`
fit as provisional.

## Named classic equations (`fitClassic`)

Reproduce a textbook hypsometric equation in one line, fit through the exact same engine
(and bias correction) as everything else in this package:

```julia-repl
julia> collect(keys(classicEquations))
10-element Vector{Symbol}:
 :cubic
 :curtis
 :naslund
 :petterson
 :prodan1
 :linear
 :trorey
 :henriksen
 :stoffels
 :prodan2

julia> fitClassic(:curtis, :h, :d, data)   # Curtis, R.O. (1967). Forest Science 13(4):365-375
log(h) = 3.6317 - 14.52 * d ^ -1

julia> fitClassic(:naslund, :h, :d, data)   # Näslund
naslund(h, d) = 1.886 + 0.1415 * d
```

`:prodan2`'s response is `prodan(h - 1.3, d)` — Prodan's original linearization is stated
in terms of height *above breast height* — so its `predict`/`fitted` return `h - 1.3`, not
total height; add `1.3` back to recover `h`. Every other entry's `predict` is already on
the total-height scale.

## Units of measurement

Every fitting entry point (`fit`, `regression`, `fitRobust`, `regressionGrouped`) accepts
`Unitful` quantity columns directly — `Units`/`Quantity`/`@u_str` and the default-unit
constants all come from
[ForestFoundations.jl](https://github.com/JuliaForests/ForestFoundations.jl), not a
direct `Unitful` dependency of this package. Units are stripped before fitting —
`Unitful` disallows `log`/`exp` of a dimensioned quantity, which the transform catalog
relies on throughout — and reattached wherever a result sits on the response's own
scale: `predict`, `fitted`, `response`, `residuals`. Fitting on plain numbers behaves
exactly as if `Unitful` did not exist: every such result stays a plain `Float64`.

```julia-repl
julia> using Unitful

julia> dataU = DataFrame(d=data.d .* u"cm", h=data.h .* u"m");

julia> mUnit = fit(AllometricModel, @formula(log(h) ~ log(d)), dataU);

julia> predict(mUnit)
10-element Vector{Union{Missing, Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}}:
 10.162942065328593 m
 11.406088401089173 m
  ⋮
 20.763004206358286 m

# predicting from data in a different (but compatible) unit converts automatically
julia> predict(mUnit, DataFrame(d=[300.0]u"mm"))
1-element Vector{Union{Missing, Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}}:
 25.17914038915757 m
```

`regression` and `fitRobust` accept `Unitful` columns exactly the same way — nothing about
the call changes, only `predict`'s return type does:

```julia-repl
julia> bestU = criteriaSelection(regression(dataU, :h, :d; nMax=2), :adjr2, :cv);

julia> predict(bestU)   # Vector{Quantity} — same numbers, same search, same ranking as the plain case
10-element Vector{Union{Missing, Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}}:
 10.2228717548827 m
  ⋮
 24.964158509929362 m

julia> predict(fitRobust(@formula(log(h) ~ log(d)), dataU, HUBER))[1:3]
3-element Vector{Union{Missing, Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}}:
 10.303914389795407 m
 11.503198530320947 m
 12.676391564976024 m
```

Summary statistics and tables (`rmse`, `mae`, `criteriaTable`, `metrics`, `siteTable`)
are always plain numbers, on the same numeric scale the model was fit on — a regression
coefficient built from an arbitrarily transformed predictor (`log(d)`, `d^2`, `1/d`, ...)
has no single clean physical unit, so this package does not attempt to invent one.

## Grouped/stratified regression

```julia-repl
julia> gdata = DataFrame(d=..., h=..., species=...);  # 3 species, distinct slope/intercept each

# compare "one equation for everyone" vs "one equation per species" vs "one shared
# equation with species as a covariate"
julia> gm = regressionGrouped(gdata, :h, :d, :species);

julia> gm.general, gm.qualy   # pooled model; pooled model with species as a categorical covariate
(h = 0.3539 - 0.03555 * d + 0.6602 * √d, h ^ -1 = 0.8596 + 0.05805 * √d - 0.2449 * log(d) - ...)

julia> gm.grouped   # one independently-selected model per species, keyed by group label
Dict{String, AllometricModel} with 3 entries:
  "Pine"  => h ^ -1 = 0.3927 - 3.931e-5 * d ^ 2 + 12.25 * d ^ -2
  "Cedar" => h ^ -1 = 0.7428 + 8.817e-5 * d ^ 2 - 0.08752 * √d
  "Oak"   => h ^ -1 = 0.7812 + 3.006e-5 * d ^ 2 - 0.1429 * log(d)

julia> criteriaTable(gm, :adjr2, :cv)
3×4 DataFrame
 Row │ type     model                              adjr2     cv
     │ String   Allometr…?                         Float64   Float64
─────┼────────────────────────────────────────────────────────────────
   1 │ General  h = 0.3539 - 0.03555 * d + 0.660…  0.822337  5.38691
   2 │ Qualy    h ^ -1 = 0.8596 + 0.05805 * √d -…  0.995854  0.822871
   3 │ Grouped  missing                            0.997638  0.621154

julia> criteriaTable(collect(values(gm.grouped)), :adjr2)   # per-group breakdown instead of the pooled comparison
3×3 DataFrame
 Row │ rank   model                              adjr2
     │ Int64  Allometr…                          Float64
─────┼────────────────────────────────────────────────────
   1 │     1  h ^ -1 = 0.7428 + 8.817e-5 * d ^…  0.99869
   2 │     2  h ^ -1 = 0.7812 + 3.006e-5 * d ^…  0.997155
   3 │     3  h ^ -1 = 0.3927 - 3.931e-5 * d ^…  0.996196
```

Here stratifying by species clearly pays for itself: `Grouped` (a separate equation per
species) and `Qualy` (one shared equation with species as a covariate) both dominate
`General` (one pooled equation) — the kind of comparison this function exists to make
easy.

### Range-safe prediction (`predictBounded`)

Calling `predict` directly on a small group's own equation risks wild extrapolation the
moment a new row's `d` falls outside that group's own fitted range. `predictBounded`
checks the range per row instead: out of the **global** range → `general`; in the global
range but out of that row's **own group's** range (or no model for that group at all) →
`qualy`; only otherwise does the group's own model get used:

```julia-repl
julia> newTrees = DataFrame(d=[15.0, 60.0, 15.0], species=["Cedar", "Cedar", "Birch"]);

julia> predictBounded(gm, newTrees)
3-element Vector{Union{Missing, Float64}}:
 ⋮   # Cedar, d=15 within Cedar's own range -> gm.grouped["Cedar"]
 ⋮   # Cedar, d=60 outside Cedar's range but inside the global range -> gm.qualy
 ⋮   # "Birch" has no fitted group at all -> gm.general
```

## Stem taper models

10 classic published taper (stem-profile) equations, fit against pooled stem-scaling data
— each tree's `dbh`/total `height` repeated once per measured section:

```julia-repl
julia> dbh    = [20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 25.0, 25.0, 25.0, 35.0, 35.0, 35.0];

julia> height = [18.0, 18.0, 18.0, 22.0, 22.0, 22.0, 20.0, 20.0, 20.0, 24.0, 24.0, 24.0];

julia> hi     = [0.3, 6.0, 14.0, 0.3, 8.0, 18.0, 0.3, 7.0, 16.0, 0.3, 9.0, 20.0];

julia> di     = [22.4, 15.8, 6.1, 33.6, 24.2, 8.7, 27.9, 18.6, 7.4, 38.9, 27.1, 9.9];

julia> ft = fit(Kozak1969(), dbh, height, hi, di);

julia> coef(ft)
3-element Vector{Float64}:
  1.1308095182748763
 -0.961584566854333
 -0.09155987790642069

julia> r2(ft), adjr2(ft), dispersion(ft)
(0.9968593430987872, 0.9961614193429621, 0.021887603541664042)
```

Linear forms (`Kozak1969`, `Schoepfer1966`, `Matte1949`) solve in closed form; every other
form (`Demaerschalk1972`, `Clutter1980`, `MaxBurkhart1976`, `Johnson1911`, `Kozak1988`,
`Kozak2004`, `Bi2000`) uses `LsqFit.jl`'s Levenberg–Marquardt solver. Either way, a
`TaperFit` carries the same tail goodness-of-fit fields as `AllometricModel`, so a
`Vector{TaperFit}` plugs into `criteriaTable`/`criteriaSelection` unchanged — useful for
picking the best-fitting form for a dataset:

```julia-repl
julia> fits = [fit(m, dbh, height, hi, di) for m in (Kozak1969(), Schoepfer1966(), Demaerschalk1972(), Bi2000())];

julia> criteriaTable(fits, :adjr2, :rmse)
4×4 DataFrame
 Row │ rank   model                              adjr2     rmse
     │ Int64  TaperFit…                          Float64   Float64
─────┼───────────────────────────────────────────────────────────────
   1 │     2  TaperFit{Schoepfer1966, Float64}…  0.996865  0.0197814
   2 │     4  TaperFit{Bi2000, Float64}(Bi2000…  0.996653  0.020437
   3 │     6  TaperFit{Kozak1969, Float64}(Koz…  0.996161  0.0218876
   4 │     8  TaperFit{Demaerschalk1972, Float…  0.996128  0.0219827
```

`predict(fit, dbh, height, h)` evaluates the fitted curve at an absolute height, scaled by
`dbh`:

```julia-repl
julia> predict(ft, 25.0, 20.0, 7.0)   # diameter at 7 m, for a 25 cm dbh / 20 m tall tree
19.575970870808078
```

The forestry-facing, `Unitful`-aware application of a `TaperFit` — diameter/height at a
point, integrated volume, log assortment — lives in
[ForestMensuration.jl](https://github.com/JuliaForests/ForestMensuration.jl)
(`taperdiameter`, `taperheight`, `taperedvolume`, `logassortment`).

## Keywords

Forest biometrics, allometric regression, model selection, grouped regression,
stem taper equations, forestry statistics.

## License

This project is licensed under the MIT License.

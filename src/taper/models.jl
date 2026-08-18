# ==============================================================================
# STEM TAPER MODELS
# ==============================================================================
# A stem taper model predicts the relative diameter dr = d(hi)/dbh of a tree at
# height hi as a function of relative height hr = hi/height (or, for the two
# models that use it, the complementary ratio x = (height-hi)/(height-1.3)).
# Every model here is fit against paired (hi, di) stem-scaling observations
# pooled across one or more trees, each carrying its own dbh/height — see
# `fit(::TaperModel, ...)` in `taper/fitting.jl`.
#
# Coefficients are always addressed as β[1], β[2], ... (1-based), matching the
# rest of this package. `hi`/`height`/`dbh` are plain numbers here (no
# `Unitful`): this package has no forestry unit conventions of its own (see
# the module docstring), so callers are expected to pass every length in the
# same consistent unit (typically cm for diameters, m for heights) — the
# `Unitful`-aware, dual `Real`/`Quantity` public API for evaluating a fitted
# model lives in `ForestMensuration.jl` (`taperdiameter`, `taperheight`,
# `taperedvolume`, `logassortment`), which calls back into `predict` here.

"""
    TaperModel

Abstract supertype for every parametric stem taper model in this package.
Each concrete subtype is a singleton (or carries only fixed, non-fitted
constants, e.g. a join point or inflection point) — the *fitted* coefficients
always live on the [`TaperFit`](@ref) returned by `fit`, never on the model
type itself.
"""
abstract type TaperModel end

"""
    ncoef(model::TaperModel) -> Int

Number of coefficients `fit` estimates for `model`.
"""
function ncoef end

"""
    islinear(model::TaperModel) -> Bool

`true` if `model`'s relative-diameter equation is linear in its coefficients
(fit in closed form via `designmatrix(model, ...) \\ dr`), `false` if it needs
a nonlinear least-squares solver (`LsqFit.curve_fit`, see `_fitnonlinear` in
`taper/fitting.jl`). Defaults to `false`; the three linear forms below
override it explicitly.
"""
islinear(::TaperModel) = false

"""
    _taperratio(model::TaperModel, hi, ht, dbh, β) -> Real

Relative diameter `dr = d(hi)/dbh` predicted by `model` at height `hi`, for a
tree of total height `ht` and breast-height diameter `dbh`, given coefficient
vector `β`. Internal — the public, dbh-scaled evaluator is
[`predict`](@ref)`(fit::TaperFit, dbh, height, hi)`.
"""
function _taperratio end

##############################################################################
# Kozak (1969) — quadratic, linear in β
##############################################################################

"""
    Kozak1969 <: TaperModel

Simple quadratic taper (Kozak, Munro & Smith, 1969):

```math
d_r = \\beta_1 + \\beta_2 h_r + \\beta_3 h_r^2, \\qquad h_r = h_i/H
```

3 coefficients, linear — fit in closed form.
"""
struct Kozak1969 <: TaperModel end

ncoef(::Kozak1969) = 3
islinear(::Kozak1969) = true

_taperratio(::Kozak1969, hi, ht, dbh, β) = (hr = hi / ht; β[1] + β[2] * hr + β[3] * hr^2)

designmatrix(::Kozak1969, hi, ht, dbh) = (hr = hi ./ ht; hcat(ones(length(hr)), hr, hr .^ 2))

##############################################################################
# Schöepfer (1966) — 5th-degree polynomial, linear in β
##############################################################################

"""
    Schoepfer1966 <: TaperModel

5th-degree relative-diameter polynomial (Schöepfer, 1966; mathematically the
same "Peters" model used elsewhere in this ecosystem):

```math
d_r = \\beta_1 + \\beta_2 h_r + \\beta_3 h_r^2 + \\beta_4 h_r^3 + \\beta_5 h_r^4 + \\beta_6 h_r^5,
\\qquad h_r = h_i/H
```

6 coefficients, linear — fit in closed form.
"""
struct Schoepfer1966 <: TaperModel end

ncoef(::Schoepfer1966) = 6
islinear(::Schoepfer1966) = true

_taperratio(::Schoepfer1966, hi, ht, dbh, β) =
  (hr = hi / ht; β[1] + β[2] * hr + β[3] * hr^2 + β[4] * hr^3 + β[5] * hr^4 + β[6] * hr^5)

designmatrix(::Schoepfer1966, hi, ht, dbh) =
  (hr = hi ./ ht; hcat(ones(length(hr)), hr, hr .^ 2, hr .^ 3, hr .^ 4, hr .^ 5))

##############################################################################
# Matte (1949) — cubic through the origin, linear in β
##############################################################################

"""
    Matte1949 <: TaperModel

Cubic taper through the origin (Matte, 1949), parametrized by the height
complement `x = (H-h_i)/(H-1.3)` rather than `h_r`:

```math
d_r = \\beta_1 x^2 + \\beta_2 x^3 + \\beta_3 x^4
```

3 coefficients, linear — fit in closed form (no intercept term: `x=0` at the
tip forces `d_r=0`).
"""
struct Matte1949 <: TaperModel end

ncoef(::Matte1949) = 3
islinear(::Matte1949) = true

_taperratio(::Matte1949, hi, ht, dbh, β) = (x = (ht - hi) / (ht - 1.3); β[1] * x^2 + β[2] * x^3 + β[3] * x^4)

designmatrix(::Matte1949, hi, ht, dbh) = (x = (ht .- hi) ./ (ht .- 1.3); hcat(x .^ 2, x .^ 3, x .^ 4))

##############################################################################
# Demaerschalk (1972) — power form, nonlinear
##############################################################################

"""
    Demaerschalk1972 <: TaperModel

Power-form taper (Demaerschalk, 1972):

```math
d_r = \\beta_1 (1-h_r)^{\\beta_2}, \\qquad h_r = h_i/H
```

2 coefficients, nonlinear — fit via `LsqFit.curve_fit`.
"""
struct Demaerschalk1972 <: TaperModel end

ncoef(::Demaerschalk1972) = 2

_taperratio(::Demaerschalk1972, hi, ht, dbh, β) = (hr = hi / ht; β[1] * (1 - hr)^β[2])

function _initialcoef(::Demaerschalk1972, hi, ht, dbh, dr)
  hr = hi ./ ht
  valid = (dr .> 0) .& (hr .< 1)
  X = hcat(ones(count(valid)), log.(1 .- hr[valid]))
  b = X \ log.(dr[valid])
  return [exp(b[1]), b[2]]
end

##############################################################################
# Clutter (1980) — power form under a square root, nonlinear
##############################################################################

"""
    Clutter1980 <: TaperModel

Power-form taper under a square root (Clutter, 1980):

```math
d_r = \\sqrt{\\max(\\beta_1 (1-h_r)^{\\beta_2},\\, 0)}, \\qquad h_r = h_i/H
```

2 coefficients, nonlinear — fit via `LsqFit.curve_fit`. The `max(\\cdot, 0)`
guard keeps the argument of the square root non-negative for `h_r` near 1.
"""
struct Clutter1980 <: TaperModel end

ncoef(::Clutter1980) = 2

_taperratio(::Clutter1980, hi, ht, dbh, β) = (hr = hi / ht; sqrt(max(β[1] * (1 - hr)^β[2], 0.0)))

function _initialcoef(::Clutter1980, hi, ht, dbh, dr)
  hr = hi ./ ht
  valid = (dr .> 0) .& (hr .< 1)
  X = hcat(ones(count(valid)), log.(1 .- hr[valid]))
  b = X \ log.(dr[valid] .^ 2)
  return [exp(b[1]), b[2]]
end

##############################################################################
# Max & Burkhart (1976) — segmented quadratic, nonlinear
##############################################################################

"""
    MaxBurkhart1976(; α1=0.11, α2=0.73) <: TaperModel

Segmented (quadratic-quadratic-quadratic) taper (Max & Burkhart, 1976), with
fixed join points `α1` (lower) and `α2` (upper) along `z = 1-h_r`:

```math
d_r = \\sqrt{\\max(\\beta_1 z + \\beta_2 z^2 + \\beta_3(\\alpha_1-z)^2\\mathbb{1}[z<\\alpha_1] +
      \\beta_4(\\alpha_2-z)^2\\mathbb{1}[z<\\alpha_2],\\, 0)}
```

4 coefficients, nonlinear — fit via `LsqFit.curve_fit`. `α1`/`α2` default to
the commonly-published starting values (0.11, 0.73) and can be overridden.
"""
struct MaxBurkhart1976 <: TaperModel
  α1::Float64
  α2::Float64
end

MaxBurkhart1976(; α1::Real=0.11, α2::Real=0.73) = MaxBurkhart1976(Float64(α1), Float64(α2))

ncoef(::MaxBurkhart1976) = 4

function _taperratio(model::MaxBurkhart1976, hi, ht, dbh, β)
  hr = hi / ht
  z = 1 - hr
  y = β[1] * z + β[2] * z^2 +
      β[3] * (model.α1 - z)^2 * (z < model.α1) +
      β[4] * (model.α2 - z)^2 * (z < model.α2)
  sqrt(max(y, 0.0))
end

function _initialcoef(model::MaxBurkhart1976, hi, ht, dbh, dr)
  hr = hi ./ ht
  z = 1 .- hr
  X = hcat(z, z .^ 2, (model.α1 .- z) .^ 2 .* (z .< model.α1), (model.α2 .- z) .^ 2 .* (z .< model.α2))
  X \ (dr .^ 2)
end

##############################################################################
# Johnson (1911) — logarithmic, nonlinear
##############################################################################

"""
    Johnson1911 <: TaperModel

Logarithmic taper (Johnson, 1911), parametrized by the height complement
`x = (H-h_i)/(H-1.3)`:

```math
d_r = \\beta_1 \\ln\\!\\left(\\frac{\\beta_2 + x}{\\beta_3}\\right)
```

3 coefficients, nonlinear — fit via `LsqFit.curve_fit`.
"""
struct Johnson1911 <: TaperModel end

ncoef(::Johnson1911) = 3

_taperratio(::Johnson1911, hi, ht, dbh, β) = (x = (ht - hi) / (ht - 1.3); β[1] * log((β[2] + x) / β[3]))

function _initialcoef(::Johnson1911, hi, ht, dbh, dr)
  x = (ht .- hi) ./ (ht .- 1.3)
  [1.0, max(mean(x), 1.0), 1.0]
end

##############################################################################
# Kozak (1988) — variable-exponent, nonlinear
##############################################################################

"""
    Kozak1988(; p=0.25) <: TaperModel

Variable-exponent taper (Kozak, 1988), with a fixed inflection point `p`:

```math
X = \\frac{1-\\sqrt{h_r}}{1-\\sqrt{p}}, \\qquad
d_r = X^{\\,\\beta_1+\\beta_2 h_r+\\beta_3 h_r^2+\\beta_4 h_r^3}
```

4 coefficients, nonlinear — fit via `LsqFit.curve_fit`.
"""
struct Kozak1988 <: TaperModel
  p::Float64
end

Kozak1988(; p::Real=0.25) = Kozak1988(Float64(p))

ncoef(::Kozak1988) = 4

function _taperratio(model::Kozak1988, hi, ht, dbh, β)
  hr = hi / ht
  x = (1 - sqrt(hr)) / (1 - sqrt(model.p))
  exponent = β[1] + β[2] * hr + β[3] * hr^2 + β[4] * hr^3
  x^exponent
end

_initialcoef(::Kozak1988, hi, ht, dbh, dr) = [1.0, 0.0, 0.0, 0.0]

##############################################################################
# Kozak (2004) — variable-exponent with dbh/height covariates, nonlinear
##############################################################################

"""
    Kozak2004(; p=0.25) <: TaperModel

Variable-exponent taper with dbh/height covariates (Kozak, 2004), the richer,
9-coefficient published form (as used by `fptools`/`timbeR`, not the
6-coefficient simplification found in early drafts of this feature):

```math
X = \\frac{1-h_r^{1/3}}{1-p^{1/3}}, \\qquad
d = \\beta_1 dbh^{\\beta_2} H^{\\beta_3} X^{\\,\\beta_4 h_r^4+\\beta_5 e^{-dbh/H}+\\beta_6 X^{0.1}+\\beta_7/dbh+\\beta_8 H^{1-h_r^{1/3}}+\\beta_9 X}
```

(`d_r = d/dbh`, i.e. the `dbh^{\\beta_2}` term above appears as `dbh^{\\beta_2-1}`
once divided through). 9 coefficients, nonlinear — fit via `LsqFit.curve_fit`.
"""
struct Kozak2004 <: TaperModel
  p::Float64
end

Kozak2004(; p::Real=0.25) = Kozak2004(Float64(p))

ncoef(::Kozak2004) = 9

function _taperratio(model::Kozak2004, hi, ht, dbh, β)
  hr = hi / ht
  x = (1 - hr^(1 / 3)) / (1 - model.p^(1 / 3))
  exponent = β[4] * hr^4 + β[5] * exp(-dbh / ht) + β[6] * x^0.1 +
             β[7] / dbh + β[8] * ht^(1 - hr^(1 / 3)) + β[9] * x
  β[1] * dbh^(β[2] - 1) * ht^β[3] * x^exponent
end

_initialcoef(::Kozak2004, hi, ht, dbh, dr) = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

##############################################################################
# Bi (2000) — trigonometric/logarithmic, nonlinear
##############################################################################

"""
    Bi2000 <: TaperModel

Trigonometric taper (Bi, 2000), the published form (as used by
`fptools`/`timbeR`, not the simplified linear-Fourier approximation found in
early drafts of this feature):

```math
d_r = \\left(\\frac{\\ln\\sin(\\tfrac{\\pi}{2}h_r)}{\\ln\\sin(\\tfrac{\\pi}{2}\\cdot\\tfrac{1.3}{H})}\\right)^{
      \\beta_1+\\beta_2\\sin(\\tfrac{\\pi}{2}h_r)+\\beta_3\\sin(\\tfrac{3\\pi}{2}h_r)+
      \\beta_4\\sin(\\tfrac{\\pi}{2}h_r)/h_r+\\beta_5 dbh+\\beta_6 h_r\\sqrt{dbh}+\\beta_7 h_r\\sqrt{H}}
```

7 coefficients, nonlinear — fit via `LsqFit.curve_fit`. Undefined at `h_i=0`
(ground level); stem-scaling data should start above the stump.
"""
struct Bi2000 <: TaperModel end

ncoef(::Bi2000) = 7

function _taperratio(::Bi2000, hi, ht, dbh, β)
  hr = hi / ht
  term1 = log(sin((π / 2) * hr)) / log(sin((π / 2) * (1.3 / ht)))
  term2 = β[1] + β[2] * sin((π / 2) * hr) + β[3] * sin((3π / 2) * hr) +
          β[4] * sin((π / 2) * hr) / hr + β[5] * dbh + β[6] * hr * sqrt(dbh) + β[7] * hr * sqrt(ht)
  term1^term2
end

_initialcoef(::Bi2000, hi, ht, dbh, dr) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

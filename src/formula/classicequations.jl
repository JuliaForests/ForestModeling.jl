# ==============================================================================
# CLASSIC EQUATIONS — reference / validation fixtures, not a maintained feature
# ==============================================================================
# These are NOT wrapped in dedicated types or macros — each entry is a single
# formula-building closure plus its citation, run through the exact same
# `fit(AllometricModel, ...)` used for everything else in this package. Their
# job is to (a) let a user reproduce a textbook equation in one line, and
# (b) serve as regression tests: does our engine reproduce known coefficients?
# Extending this list costs one line; it is deliberately not exhaustive.
# Because `fitClassic` delegates to `fit`, unit handling (see `units.jl`) is
# inherited for free — nothing here needs to know about `Unitful` at all.

"""
    classicEquations::Dict{Symbol,NamedTuple}

Each entry has a `formula(h, d) -> FormulaTerm` closure and a `citation`.

Currently included: `:curtis`, `:henriksen`, `:stoffels`, `:trorey`,
`:prodan1`, `:prodan2`, `:petterson`, `:naslund`, `:linear`, `:cubic`.

!!! note
    `:prodan2`'s response is `prodan(h - 1.3, d)` — Prodan's original
    linearization is stated in terms of height *above breast height*, so
    `predict`/`fitted` on that model return `h - 1.3`, not total height; add
    `1.3` back to recover `h`. Every other entry's `predict` is already on
    the total-height scale.

Every response transform below is built from the same named
[`petterson`](@ref)/[`naslund`](@ref)/[`prodan`](@ref) functions the
transform catalog uses (not the equivalent raw `^`/`/` expression) — bias
correction ([`predictBiasCorrected!`](@ref)) recognizes a response transform
by its function *name*, so writing e.g. `h^-1` instead of `petterson(h)`
would fit without error but then throw when `predict` tries to back-transform
it.
"""
const classicEquations = Dict{Symbol,NamedTuple}(
  :linear => (formula=(h, d) -> @eval(@formula($h ~ 1 + $d)),
    citation="generic linear reference form"),
  :cubic => (formula=(h, d) -> @eval(@formula($h ~ 1 + $d + $d^2 + $d^3)),
    citation="generic cubic reference form"),
  :trorey => (formula=(h, d) -> @eval(@formula($h ~ 1 + $d + $d^2)),
    citation="Trorey (1932)"),
  :curtis => (formula=(h, d) -> @eval(@formula(log($h) ~ 1 + $d^-1)),
    citation="Curtis, R.O. (1967). Forest Science 13(4):365-375"),
  :henriksen => (formula=(h, d) -> @eval(@formula($h ~ 1 + log($d))),
    citation="Henriksen (1950)"),
  :stoffels => (formula=(h, d) -> @eval(@formula(log($h) ~ 1 + log($d))),
    citation="Stoffels & Van Soest"),
  :prodan1 => (formula=(h, d) -> @eval(@formula(prodan($h, $d) ~ 1 + $d + $d^2)),
    citation="Prodan (linearized form I)"),
  :prodan2 => (formula=(h, d) -> @eval(@formula(prodan($h - 1.3, $d) ~ 1 + $d + $d^2)),
    citation="Prodan (linearized form II) — predict returns h - 1.3, see note above"),
  :petterson => (formula=(h, d) -> @eval(@formula(petterson($h) ~ 1 + $d^-1)),
    citation="Petterson"),
  :naslund => (formula=(h, d) -> @eval(@formula(naslund($h, $d) ~ 1 + $d)),
    citation="Näslund"),
)

"""
    fitClassic(name::Symbol, h::Symbol, d::Symbol, data)

Fit a named classic equation from [`classicEquations`](@ref) through the same
engine as every other model in this package — useful for a one-line
comparison against a textbook coefficient, or as a regression test.

```julia
m = fitClassic(:curtis, :h, :d, data)
predict(m)
```
"""
function fitClassic(name::Symbol, h::Symbol, d::Symbol, data)
  haskey(classicEquations, name) || throw(ArgumentError("unknown classic equation :$name — see keys(classicEquations)"))
  f = classicEquations[name].formula(h, d)
  return fit(AllometricModel, f, data)
end

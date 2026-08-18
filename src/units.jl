# ==============================================================================
# UNIT HANDLING AT THE FITTING BOUNDARY
# ==============================================================================
# Unitful.jl deliberately disallows `log`/`exp` (and similar) on a dimensioned
# `Quantity` — log of a physical quantity is not well-defined independent of
# unit choice — so the transform catalog in `formula/transforms.jl` (log, 1/x,
# √x, ...) cannot run on `Quantity` columns directly. Every fitting entry
# point therefore strips units *before* building a `ModelFrame`, runs the
# existing plain-`Float64` numeric core completely unchanged, and only
# reattaches units at the edges: once on the model (so `predict` knows what to
# hand back later) and again whenever a function produces a value on the
# response's original scale.
#
# The `nonmissingtype`/`T <: Quantity` check below mirrors
# `ForestFoundations.removeunits` exactly, including its reliance on
# `Unitful.ustrip` handling `missing` elementwise — the same assumption that
# code already makes for `DataFrame` columns.

_colunit(v::AbstractVector) = (T = nonmissingtype(eltype(v)); T <: Quantity ? unit(T) : nothing)
_colunit(::Any) = nothing

_stripcol(v::AbstractVector) = nonmissingtype(eltype(v)) <: Quantity ? ustrip.(v) : v
_stripcol(v) = v

"""
    stripcolumnunits(data) -> (cols::NamedTuple, units::NamedTuple)

Normalizes any `Tables.jl`-compatible `data` into a column `NamedTuple`,
stripping the unit off every `Unitful.Quantity` column so the rest of the
fitting machinery only ever sees plain numeric/categorical columns — exactly
as it did before this package supported units at all. `units` carries the
same keys as `cols`, one `Unitful.Units` (or `nothing` for a column that
carried none) per column — meant to be stored on the fitted model and
consulted later by [`matchunits`](@ref) and `_restoreunit`.
"""
function stripcolumnunits(data)
  cols = Tables.columntable(data)
  ks = keys(cols)
  units = NamedTuple{ks}(map(_colunit, values(cols)))
  stripped = NamedTuple{ks}(map(_stripcol, values(cols)))
  return stripped, units
end

function _matchcol(v::AbstractVector, u::Units)
  T = nonmissingtype(eltype(v))
  return T <: Quantity ? ustrip.(uconvert.(u, v)) : v
end
_matchcol(v::AbstractVector, ::Nothing) = _stripcol(v)
_matchcol(v, ::Any) = v

"""
    matchunits(data, units::NamedTuple) -> NamedTuple

Normalizes new prediction/scoring data the same way [`stripcolumnunits`](@ref)
does, except a `Quantity` column is first `uconvert`ed to the unit the model
was **fitted** in — so a model trained on `cm` can score a table given in
`inch` or `mm`. A column supplied as a plain number is assumed to already be
expressed in the fit unit, mirroring the rest of the `JuliaForests` org's
"bare numbers assume the documented default" idiom (e.g.
`ForestFoundations.withunit`). Columns absent from `units`, or recorded
with a `nothing` unit (the model was fit unitless), pass through untouched.
"""
function matchunits(data, units::NamedTuple)
  cols = Tables.columntable(data)
  ks = keys(cols)
  matched = map(ks) do k
    haskey(units, k) ? _matchcol(cols[k], units[k]) : cols[k]
  end
  return NamedTuple{ks}(matched)
end

"""
    _restoreunit(v, ::Nothing) -> v
    _restoreunit(v, u::Units) -> v .* u

Reattaches a stored unit to a plain-numeric result. A `nothing` unit (the
model was fit without units) is a no-op, so callers that never touch
`Unitful` get back exactly the plain numbers they always have.
"""
_restoreunit(v, ::Nothing) = v
_restoreunit(v, u::Units) = v .* u

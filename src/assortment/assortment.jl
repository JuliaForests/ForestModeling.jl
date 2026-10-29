# The application of a fitted taper model to one tree. Every function works on the values of the
# lengths in the units of the fit and returns them with units; the model is a `fitnls` of a taper
# equation with the predictors (h, D, H), in this order.

# The units of the fit and the curve d(h) of the tree, on the values without units
function _stem(model::NLSFit, D, H)
  length(model.predictors) == 3 ||
    throw(ArgumentError("A taper model is fitted with the predictors (h, D, H); it has $(model.predictors)."))
  units = (h=model.units[model.predictors[1]], D=model.units[model.predictors[2]],
           H=model.units[model.predictors[3]], d=model.units[model.response])
  all(u -> 1.0 * u isa Length, units) ||
    throw(ArgumentError("The taper model needs the lengths with units (cm, m, inch, ft)."))
  Dv, Hv = ustrip(units.D, D), ustrip(units.H, H)
  d(h) = model.f(h, Dv, Hv, model.θ)
  units, d, ustrip(units.h, H)
end

"""
    taperdiameter(model, D, H, h)

Diameter at the height(s) `h` of a tree of diameter at breast height `D` and total height `H`, by the
fitted taper `model` (a [`fitnls`](@ref) of a taper equation with the predictors `(:h, :D, :H)`).

# Example
```julia
taperdiameter(taper, 25.0u"cm", 28.0u"m", [2.0, 10.0, 20.0]u"m")
```
"""
function taperdiameter(model::NLSFit, D, H, h)
  units, d, _ = _stem(model, D, H)
  d.(ustrip.(units.h, h)) .* units.d
end

"""
    taperheight(model, D, H, d)

Height at which the stem of a tree of diameter at breast height `D` and total height `H` narrows to
the diameter `d`, the inverse of [`taperdiameter`](@ref): the root of ``\\hat d(h) - d`` between
the base and the top of the stem, found by bisection (`Roots.jl`), since the taper decreases
with the height. A diameter the stem does not reach is a `DomainError`.

# Example
```julia
taperheight(taper, 25.0u"cm", 28.0u"m", 10.0u"cm")   # the height of a top diameter of 10 cm
```
"""
function taperheight(model::NLSFit, D, H, d)
  units, curve, top = _stem(model, D, H)
  target = ustrip(units.d, d)
  lo, hi = top * 1e-6, top * (1 - 1e-6)
  (curve(lo) - target) * (curve(hi) - target) <= 0 ||
    throw(DomainError(d, "The stem narrows from $(curve(lo) * units.d) to $(curve(hi) * units.d); it does not reach $d."))
  find_zero(h -> curve(h) - target, (lo, hi), Bisection()) * units.h
end

"""
    taperedvolume(model, D, H, h1=0, h2=H)

Volume of the stem between the heights `h1` and `h2` (the whole stem by default) of a tree of
diameter at breast height `D` and total height `H`, integrating the sectional area of the fitted
taper (`QuadGK.jl`),

```math
v = \\int_{h_1}^{h_2} \\frac{\\pi}{4} \\hat d(h)^2 \\, dh
```

in m³ (ft³ for heights in an imperial unit). It is the fitted counterpart of `cubage` of
ForestMensuration.jl, which integrates the measured diameters.

# Example
```julia
taperedvolume(taper, 25.0u"cm", 28.0u"m")                       # the whole stem
taperedvolume(taper, 25.0u"cm", 28.0u"m", 0.1u"m", 12.0u"m")    # a section
```
"""
function taperedvolume(model::NLSFit, D, H, h1=zero(H), h2=H)
  units, curve, top = _stem(model, D, H)
  a, b = ustrip(units.h, h1), ustrip(units.h, h2)
  0 <= a < b <= top || throw(DomainError((h1, h2), "The section must satisfy 0 <= h1 < h2 <= H."))
  _volume(curve, a, b, units)
end

# the volume between the heights a and b (values in the unit of the heights)
_volume(curve, a, b, units) =
  inreferencevolume(first(quadgk(h -> quartπ * max(curve(h), 0)^2, a, b)) * units.d^2 * units.h, units.h)

"""
    logassortment(model, D, H, products; stump=0, fixedlength=false)

Cuts the stem of a tree of diameter at breast height `D` and total height `H` into logs of the
`products`, by the fitted taper `model`, and returns a row per product with the number of logs `n`
and their volume `v`.

`products` has a row per product, from the most to the least valuable, with the columns `name`,
`sed` (the smallest diameter of the small end of a log), `minlength` and `maxlength` (the shortest
and the longest log) and `kerf` (the length lost in each cut), all with units. From the `stump`
up, each product takes logs of the longest length allowed while the stem is not thinner than its
`sed` and a log of at least `minlength` still fits; the rest of the stem goes to the next product.
The volume of each log is [`taperedvolume`](@ref) between its ends. With `fixedlength=true` every
log has the longest length (`maxlength`) and the shorter one that would end a product is not cut,
as in the Python package fptools.

It is a greedy priority cut, not an optimization of the value of the stem: the algorithm of
`poly5_logs` of the R package timbeR (2.0.1), whose numbers of logs and volumes it reproduces with
the same coefficients. The `bi_logs` and `kozak_logs` of timbeR read the loss of each cut from the
column of the longest log, so their logs differ from these by that loss.

# Example
```julia
products = DataFrame(name=["sawlog", "pulpwood"], sed=[18.0, 8.0]u"cm",
                     minlength=[2.5, 2.0]u"m", maxlength=[4.0, 6.0]u"m", kerf=[0.005, 0.005]u"m")
logassortment(taper, 30.0u"cm", 28.0u"m", products; stump=0.1u"m")
```
"""
function logassortment(model::NLSFit, D, H, products; stump=zero(H), fixedlength=false)
  issubset([:name, :sed, :minlength, :maxlength, :kerf], propertynames(products)) ||
    throw(ArgumentError("products must have the columns name, sed, minlength, maxlength and kerf."))
  units, curve, top = _stem(model, D, H)
  h0 = ustrip(units.h, stump)
  0 <= h0 < top || throw(DomainError(stump, "The stump must satisfy 0 <= stump < H."))
  n = zeros(Int, nrow(products))
  v = fill(zero(inreferencevolume(1.0 * units.d^2 * units.h, units.h)), nrow(products))
  for (i, p) in enumerate(eachrow(products))
    sed = ustrip(units.d, p.sed)
    minlength, maxlength, kerf = ustrip.(units.h, (p.minlength, p.maxlength, p.kerf))
    # the height where the stem narrows to the sed: none of it when the base is thinner, all of it
    # when the top is thicker
    lo, hi = top * 1e-6, top * (1 - 1e-6)
    limit = curve(lo) < sed ? lo : curve(hi) >= sed ? top : find_zero(h -> curve(h) - sed, (lo, hi), Bisection())
    shortest = fixedlength ? maxlength : minlength
    while h0 + shortest <= limit
      l = min(maxlength, limit - h0)
      v[i] += _volume(curve, h0, h0 + l, units)
      n[i] += 1
      h0 += l + kerf
    end
  end
  DataFrame(name=products.name, n=n, v=v)
end

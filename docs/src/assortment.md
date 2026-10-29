# Taper and Assortments

A **taper equation** gives the diameter ``d`` at the height ``h`` of a tree from its diameter at
breast height ``D`` and its total height ``H``. Fitted to the stem profiles of felled trees, it
describes the stem of any tree of the stand: the diameter at any height, the height of any
diameter, the volume of any section, and the logs of each product the stem yields — the
**assortments** (sortimentos).

## The data

The stem profiles of 50 trees, with the diameter ``d`` with bark measured at the heights ``h`` (995
diameters), and the ``D`` and ``H`` of each tree, read from a local file (`data/cubage.csv`, not
in the repository). The units are in the header of each column, as `parseunits` of
ForestFoundations reads them:

```@example taper
using ForestModeling, DataFrames, CSV

cubage = parseunits(CSV.read(joinpath(pkgdir(ForestModeling), "data", "cubage.csv"), DataFrame))
first(cubage, 5)
```

## The equations

Every equation is a plain function `f(h, D, H, θ)` of the package, with ``z = h/H`` the relative
height:

| Function | Equation | Parameters |
|:---------|:---------|:-----------|
| [`kozak1969`](@ref) | ``(d/D)^2 = \beta_0 + \beta_1 z + \beta_2 z^2`` | 3 |
| [`schoepfer1966`](@ref) | ``d/D = \beta_0 + \beta_1 z + \dots + \beta_5 z^5`` | 6 |
| [`ormerod1973`](@ref) | ``d = D\,[(H - h)/(H - h_{ref})]^{\beta}`` | 1 |
| [`demaerschalk1972`](@ref) | ``d = \beta_0 D^{\beta_1} (H - h)^{\beta_2} H^{\beta_3}`` | 4 |
| [`maxburkhart1976`](@ref) | three quadratics in ``z`` joined at ``a_1`` and ``a_2`` | 4 |
| [`kozak1988`](@ref) | variable exponent, inflection point ``p`` | 8 |
| [`kozak2004`](@ref) | variable exponent with ``D/H`` | 9 |
| [`bi2000`](@ref) | trigonometric variable form | 7 |

The constants that are not fitted — the height of reference `href`, the join points `a1` and `a2`,
the inflection point `p` — are keywords, given in a short function that leaves `f(h, D, H, θ)`.

## Fitting

A taper equation is a nonlinear regression of ``d`` on ``(h, D, H)``, so [`fitnls`](@ref) fits it, with
the predictors in this order. The variance of the diameter grows with the diameter, which
`fitnls` estimates (`variance=:power`). Kozak (2004):

```@example taper
k04(h, D, H, θ) = kozak2004(h, D, H, θ; href=1.3)
taper = fitnls(k04, cubage, :d, (:h, :D, :H), [1.0, 0.97, 0.03, 0.49, -0.87, 0.50, 3.88, 0.03, -0.19])
```

All the equations on the same data, compared on the original scale:

```@example taper
starts = [
  ("kozak1969", (h, D, H, θ) -> kozak1969(h, D, H, θ), [1.2, -2.2, 1.1], :constant),
  ("schoepfer1966", (h, D, H, θ) -> schoepfer1966(h, D, H, θ), [1.2, -3.0, 10.0, -20.0, 18.0, -6.0], :power),
  ("ormerod1973", (h, D, H, θ) -> ormerod1973(h, D, H, θ; href=1.3), [0.8], :power),
  ("demaerschalk1972", (h, D, H, θ) -> demaerschalk1972(h, D, H, θ), [1.0, 1.0, 0.8, -0.8], :power),
  ("maxburkhart1976", (h, D, H, θ) -> maxburkhart1976(h, D, H, θ; a1=0.75, a2=0.1), [-3.0, 1.5, -1.5, 50.0], :power),
  ("kozak1988", (h, D, H, θ) -> kozak1988(h, D, H, θ; p=0.25), [1.0, 1.0, 1.0, 0.5, -0.1, 0.5, -0.5, 0.1], :power),
  ("kozak2004", k04, [1.0, 0.97, 0.03, 0.49, -0.87, 0.50, 3.88, 0.03, -0.19], :power),
  ("bi2000", (h, D, H, θ) -> bi2000(h, D, H, θ; href=1.3), [1.8, -0.2, -0.04, -0.9, -0.0006, 0.07, -0.14], :power),
]
comparison = map(starts) do (name, f, θ0, variance)
  model = fitnls(f, cubage, :d, (:h, :D, :H), θ0; variance)
  (; equation=name, parameters=length(θ0), metrics(cubage.d, predict(model, cubage))...)
end
DataFrame(comparison)
```

The variable-exponent equations (Kozak, Bi) describe these stems best. Two remarks:

- the quadratic of Kozak (1969) turns negative near the top of some stems, where the diameter is
  taken as zero; the variance cannot grow with a mean of zero, so it is fitted with
  `variance=:constant`;
- with Kozak (1988) the optimizer of the variance function warns that it did not converge; the
  estimates above still describe the stems almost as well as Kozak (2004).

[`blockcv`](@ref) with the block `:tree` compares the equations on trees that were not in the fit.

## The stem of a tree

For a tree of ``D`` = 25 cm and ``H`` = 28 m, the diameters along the stem
([`taperdiameter`](@ref)), the height where the stem narrows to 10 cm ([`taperheight`](@ref)), and the
volume of the whole stem and of its first 12 m ([`taperedvolume`](@ref)):

```@example taper
D, H = 25.0u"cm", 28.0u"m"
taperdiameter(taper, D, H, [0.1, 1.3, 10.0, 20.0, 27.0]u"m")
```

```@example taper
taperheight(taper, D, H, 10.0u"cm")
```

```@example taper
(stem=taperedvolume(taper, D, H), first12m=taperedvolume(taper, D, H, 0.0u"m", 12.0u"m"))
```

The commercial volume, up to a top diameter of 8 cm, is the volume up to the height of that
diameter:

```@example taper
taperedvolume(taper, D, H, 0.1u"m", taperheight(taper, D, H, 8.0u"cm"))
```

and the stem of the mean tree of the stand is the taper of the mean ``D`` and ``H``, ``\bar D`` and
``\bar H``, of its trees:

```@example taper
trees = unique(cubage[!, [:tree, :D, :H]])
D̄, H̄ = sum(trees.D) / nrow(trees), sum(trees.H) / nrow(trees)
taperdiameter(taper, D̄, H̄, [0.1, 1.3, 5.0, 10.0, 15.0, 20.0]u"m")
```

The volume integrates the sectional area of the fitted taper,
``v = \int_{h_1}^{h_2} \frac{\pi}{4} \hat d(h)^2 \, dh``; it is close to the volume of the measured
diameters by Smalian (`cubage` of ForestMensuration.jl):

```@example taper
tree1 = cubage[cubage.tree .== 1, :]
smalian = sum(quartπ .* (tree1.d[1:end-1] .^ 2 .+ tree1.d[2:end] .^ 2) ./ 2 .* diff(tree1.h))
(smalian=uconvert(u"m^3", smalian), taper=taperedvolume(taper, tree1.D[1], tree1.H[1], tree1.h[1], tree1.h[end]))
```

## Assortments

The products, from the most to the least valuable: the smallest diameter of the small end of a log
(`sed`), the shortest and the longest log, and the length lost in each cut (`kerf`):

```@example taper
products = DataFrame(name=["sawlog", "pulpwood"], sed=[18.0, 8.0]u"cm", minlength=[2.5, 2.0]u"m",
                     maxlength=[4.0, 6.0]u"m", kerf=[0.005, 0.005]u"m")
```

[`logassortment`](@ref) cuts the stem from the stump up: the logs of the first product while the stem
is not thinner than its `sed`, then the next product on the rest of the stem:

```@example taper
logassortment(taper, 30.0u"cm", 28.0u"m", products; stump=0.1u"m")
```

With `fixedlength=true` every log has the longest length, and the shorter log that would end a
product is left to the next one:

```@example taper
logassortment(taper, 30.0u"cm", 28.0u"m", products; stump=0.1u"m", fixedlength=true)
```

For every tree of a stand, `combine` the assortment of each one:

```@example taper
stand = combine(groupby(trees, :tree)) do t
  logassortment(taper, only(t.D), only(t.H), products; stump=0.1u"m")
end
combine(groupby(stand, :name), :n => sum => :n, :v => sum => :v)
```

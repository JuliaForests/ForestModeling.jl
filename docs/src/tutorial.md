# Getting Started

The examples use three data sets of the examples of ForestMensuration.jl, with the units of each
column: the heights and diameters of 55 trees in four plots, the volumes of 91 trees and the
biomass of 46 trees.

```@example tutorial
using ForestModeling, DataFrames, Random

heights = DataFrame(
  plot=["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
        "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
        "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
        "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D"],
  h=[20.9, 19.6, 13.2, 23.3, 19.2, 16.2, 8.3, 19.7, 11.0, 24.0, 25.8, 28.2, 24.2, 26.2, 28.3,
     14.4, 14.9, 15.6, 8.2, 22.1, 16.7, 22.3, 19.5, 15.9, 16.7, 24.5, 21.7, 23.8,
     20.8, 17.7, 19.3, 16.7, 22.2, 18.6, 6.9, 22.3, 8.7, 22.1, 21.0, 23.5,
     19.5, 19.7, 18.2, 13.9, 12.3, 14.5, 12.3, 18.6, 18.0, 17.4, 24.3, 22.8, 23.2, 23.5, 25.2]u"m",
  d=[31.5, 30.0, 26.5, 31.0, 29.0, 26.5, 14.5, 28.8, 19.0, 31.5, 32.5, 33.8, 32.5, 33.3, 36.0,
     24.0, 28.0, 23.0, 15.5, 31.0, 27.0, 29.0, 28.0, 26.0, 29.0, 30.0, 29.0, 30.5,
     25.0, 26.8, 27.5, 26.0, 26.0, 25.8, 10.8, 27.0, 16.5, 26.5, 27.0, 26.3,
     26.0, 25.5, 25.0, 23.5, 22.0, 23.0, 23.0, 26.0, 25.5, 27.5, 26.5, 26.5, 27.8, 26.0, 27.0]u"cm",
)

volume = DataFrame(
  v=[0.153963, 0.151394, 0.178328, 0.221116, 0.225294, 0.183733, 0.210808, 0.200058, 0.183102, 0.231159,
     0.272580, 0.224460, 0.322517, 0.246280, 0.264114, 0.308264, 0.326833, 0.316227, 0.331639, 0.354205,
     0.421524, 0.331393, 0.421420, 0.462966, 0.427326, 0.482238, 0.345795, 0.476053, 0.556150, 0.441768,
     0.543030, 0.449364, 0.450939, 0.520234, 0.422021, 0.410333, 0.598158, 0.763291, 0.610406, 0.596743,
     0.667189, 0.653583, 0.626541, 0.638496, 0.683411, 0.740718, 0.710917, 0.737058, 0.714613, 0.749651,
     0.786985, 0.781447, 0.727167, 0.960877, 0.768727, 0.690346, 0.971356, 0.989444, 0.809371, 0.945937,
     0.892852, 1.074173, 1.176289, 1.003187, 1.010025, 1.054886, 1.030369, 1.176427, 1.021023, 1.098403,
     1.050754, 1.070469, 1.118549, 1.559863, 1.083926, 1.322375, 1.271627, 1.057683, 1.262937, 1.496877,
     1.398170, 1.317415, 1.653541, 1.313996, 1.611356, 1.399721, 1.300030, 1.480614, 1.583871, 1.454267,
     1.616712]u"m^3",
  h=[18.4, 18.9, 20.2, 20.4, 22.5, 19.2, 21.8, 20.0, 19.2, 22.3, 23.0, 19.0, 23.4, 19.1, 20.2, 21.9,
     22.2, 21.8, 23.4, 23.2, 22.4, 21.5, 23.5, 23.5, 24.4, 22.9, 21.2, 22.7, 22.6, 22.4, 23.6, 20.7,
     22.5, 23.7, 21.0, 22.0, 24.0, 25.6, 23.0, 24.0, 23.3, 23.2, 23.1, 23.0, 23.4, 26.0, 24.1, 23.8,
     24.6, 24.2, 24.5, 25.3, 22.1, 25.2, 25.0, 22.6, 24.2, 26.0, 23.5, 25.3, 24.8, 25.1, 25.0, 23.8,
     23.0, 24.4, 25.0, 25.1, 24.9, 24.8, 24.2, 23.7, 25.2, 29.2, 23.7, 25.1, 24.9, 24.2, 25.1, 26.7,
     25.5, 25.5, 26.2, 23.9, 25.0, 26.5, 26.9, 25.2, 26.0, 26.5, 25.2]u"m",
  d=[16.9, 17.2, 17.5, 17.8, 17.8, 17.8, 17.8, 18.5, 18.5, 18.8, 19.1, 19.7, 19.7, 19.7, 20.1, 20.1,
     21.0, 21.0, 21.3, 22.0, 22.0, 22.3, 22.6, 22.6, 23.6, 23.6, 23.6, 23.6, 24.2, 24.5, 24.5, 24.5,
     25.1, 25.5, 25.5, 26.1, 26.4, 27.1, 27.1, 27.4, 27.4, 27.7, 28.0, 28.0, 28.3, 28.6, 29.3, 29.6,
     29.9, 30.2, 30.2, 31.2, 31.2, 31.5, 31.5, 31.5, 32.1, 32.1, 32.5, 33.1, 33.4, 33.7, 33.7, 34.1,
     34.4, 35.0, 35.0, 35.0, 35.0, 35.7, 36.3, 36.3, 36.6, 37.2, 37.2, 37.9, 37.9, 38.2, 38.2, 38.8,
     38.8, 39.5, 39.8, 39.8, 40.1, 40.1, 40.1, 41.1, 41.4, 41.7, 41.7]u"cm",
)

biomass = DataFrame(
  d=[6.5, 8.0, 8.5, 8.5, 9.5, 9.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.2, 10.5, 10.5, 10.5, 11.0,
     11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.5, 12.5, 13.0, 13.0, 13.0, 13.0, 13.0, 13.5,
     13.5, 14.0, 14.0, 14.0, 14.5, 14.5, 14.5, 14.5, 14.5, 15.0, 15.0, 15.5, 16.0]u"cm",
  h=[11.5, 15.2, 14.8, 17.0, 16.9, 19.8, 17.7, 19.0, 18.2, 15.8, 17.5, 17.8, 16.7, 18.0, 17.9, 16.0,
     18.8, 17.1, 19.3, 19.5, 18.8, 20.0, 19.1, 16.8, 17.9, 19.7, 19.0, 19.9, 20.5, 20.1, 16.1, 19.3,
     20.2, 19.2, 21.1, 21.2, 21.8, 18.9, 20.6, 20.6, 21.8, 22.2, 19.4, 20.5, 21.1, 21.3]u"m",
  b=[5.832, 18.825, 16.065, 15.657, 23.476, 29.376, 25.247, 31.962, 31.954, 27.356, 25.163, 29.865,
     23.391, 32.887, 38.719, 31.059, 31.689, 26.379, 36.608, 44.182, 37.616, 40.209, 45.496, 39.182,
     40.036, 58.883, 56.035, 64.647, 65.173, 59.035, 33.082, 50.726, 53.361, 56.979, 77.001, 84.832,
     77.141, 65.280, 65.389, 65.240, 76.359, 69.406, 71.757, 81.424, 100.448, 89.240]u"kg",
)
nothing # hide
```

## Volume

Schumacher–Hall as a generalized linear model: the log link gives the power shape and the Gamma
family the variance growing with the volume. The response is the volume itself, not its logarithm,
and the model is shown as its equation on the original scale:

```@example tutorial
sh = fitglm(@formula(v ~ log(d) + log(h)), volume)
```

The prediction is the mean volume, in m³, with no back-transformation; with
`interval=:confidence`, the interval of the mean:

```@example tutorial
predict(sh, first(volume, 3); interval=:confidence)
```

The same equation as a nonlinear regression, [`schumacherhall`](@ref), with the variance growing
with a power `k` of the mean estimated from the data instead of fixed by the Gamma family:

```@example tutorial
shnls = fitnls(schumacherhall, volume, :v, (:d, :h), [1e-4, 2.0, 1.0])
```

Both on the original scale, so their errors are compared directly:

```@example tutorial
(glm=metrics(volume.v, predict(sh, volume)), nls=metrics(volume.v, predict(shnls, volume)))
```

## Biomass

The power model of the diameter, ``b = aD^{\beta}``, and the one with the height, in kg:

```@example tutorial
fitnls(power, biomass, :b, :d, [0.1, 2.4])
```

```@example tutorial
fitglm(@formula(b ~ log(d) + log(h)), biomass)
```

## Height

Curtis, ``h = e^{\beta_0 + \beta_1 / d}``, as a generalized linear model of the inverse of the diameter
with the log link:

```@example tutorial
fitglm(@formula(h ~ inv(d)), heights)
```

and as a nonlinear regression, with the prediction interval of a new tree:

```@example tutorial
curtisnls = fitnls(curtis, heights, :h, :d, [3.5, -10.0])
predict(curtisnls, first(heights, 3); interval=:prediction)
```

A curve with an asymptote, as [`chapmanrichards`](@ref), needs diameters large enough for the
height to level off; with the diameters of these plots (10.8 to 36 cm) its asymptote is not
determined, and the two parameters of Curtis are the better choice.

## Other classic equations

The equations linear in their coefficients, or linear after the logarithm, are formulas of
[`fitglm`](@ref): the ordinary linear regression with `Normal()` and `IdentityLink()`, the power
forms with the log link of the Gamma default, and a linear form with a variance growing with the
mean with `Gamma()` and `IdentityLink()`. In a formula, `d^2 & h` is the product ``d^2 h`` and `0 +`
removes the intercept:

| Equation | Form | Formula and arguments of `fitglm` |
|:---------|:-----|:----------------------------------|
| Stoffels | ``h = \beta_0 d^{\beta_1}`` | `@formula(h ~ log(d))` |
| Henricksen | ``h = \beta_0 + \beta_1 \ln d`` | `@formula(h ~ log(d)); family=Normal(), link=IdentityLink()` |
| parabolic | ``h = \beta_0 + \beta_1 d + \beta_2 d^2`` | `@formula(h ~ d + d^2); family=Normal(), link=IdentityLink()` |
| Spurr | ``v = \beta_0 + \beta_1 d^2 h`` | `@formula(v ~ d^2 & h); family=Normal(), link=IdentityLink()` |
| Stoate | ``v = \beta_0 + \beta_1 d^2 + \beta_2 d^2 h + \beta_3 h`` | `@formula(v ~ d^2 + d^2 & h + h); family=Normal(), link=IdentityLink()` |
| Näslund | ``v = \beta_1 d^2 + \beta_2 d^2 h + \beta_3 d h^2 + \beta_4 h^2`` | `@formula(v ~ 0 + d^2 + d^2 & h + d & h^2 + h^2); family=Normal(), link=IdentityLink()` |
| Meyer | ``v = \beta_0 + \beta_1 d + \beta_2 d^2 + \beta_3 d h + \beta_4 d^2 h + \beta_5 h`` | `@formula(v ~ d + d^2 + d & h + d^2 & h + h); family=Normal(), link=IdentityLink()` |

Stoffels and Henricksen:

```@example tutorial
(stoffels=fitglm(@formula(h ~ log(d)), heights),
 henricksen=fitglm(@formula(h ~ log(d)), heights; family=Normal(), link=IdentityLink()))
```

Stoate, with the variance growing with the volume:

```@example tutorial
fitglm(@formula(v ~ d^2 + d^2 & h + h), volume; link=IdentityLink())
```

The rational equations of Honer (1965) and Takata (1958), ``v = d^2/(\beta_0 + \beta_1/h)`` and
``v = d^2 h/(\beta_0 + \beta_1 d)``, are not linear in any scale, so [`honer`](@ref) and
[`takata`](@ref) are fitted by [`fitnls`](@ref):

```@example tutorial
(honer=fitnls(honer, volume, :v, (:d, :h), [400.0, 20000.0]),
 takata=fitnls(takata, volume, :v, (:d, :h), [20000.0, 400.0]))
```

## Plots: a mixed model and its calibration

A mixed model gives each plot a random intercept, a shift of its curve around the population
curve:

```@example tutorial
mixed = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), heights)
```

In a new plot where only two heights were measured, [`calibrate`](@ref) estimates the random effect of
the plot from those two trees and predicts all of them; without calibration the plot gets the
population curve. Here the plots differ little (a standard deviation of 0.5 m between plots against
2.8 m between trees), so the calibration moves the curve little:

```@example tutorial
newplot = transform(heights[heights.plot .== "A", :], :plot => (p -> fill("new", length(p))) => :plot)
measured = newplot[1:2, :]
DataFrame(h=newplot.h, population=predict(mixed, newplot), calibrated=calibrate(mixed, measured, newplot))
```

## Validation by plot

[`blockcv`](@ref) predicts the plots of each fold with the model fitted without them; with four plots
and `k = 4`, each plot is predicted by the other three:

```@example tutorial
curtisfit(t) = fitnls(curtis, t, :h, :d, [3.5, -10.0])
mixedfit(t) = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), t)
(curtis=blockcv(curtisfit, heights, :h, :plot; k=4), mixed=blockcv(mixedfit, heights, :h, :plot; k=4))
```

## One model per category

[`fitbygroup`](@ref) fits a model for each level of a column, in parallel threads; each cell shows the
equation of its plot:

```@example tutorial
fitbygroup(curtisfit, heights, :plot)
```

Whether the plots need a curve each, or one curve serves all of them, is the test of the identity
of models, [`identitytest`](@ref): the likelihood ratio of the curves fitted apart against the
one fitted to all the trees. A small `pvalue` rejects one curve for all the plots:

```@example tutorial
identitytest(curtisfit, heights, :plot)
```

## Predicted against observed

The F test of Graybill, [`graybilltest`](@ref), regresses the predicted volumes on the observed ones
and tests that the line is the identity, ``\beta_0 = 0`` and ``\beta_1 = 1``. Schumacher–Hall does
not reject it; the same predictions 10% higher are rejected:

```@example tutorial
(sh=graybilltest(volume.v, predict(sh, volume)), biased=graybilltest(volume.v, 1.1 .* predict(sh, volume)))
```

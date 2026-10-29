# Getting Started

The examples use simulated stands of known parameters, so the estimates can be compared with the
truth: 30 plots of 15 trees, the heights from a Chapman–Richards curve whose asymptote changes
from plot to plot (a standard deviation of 3 m around 25 m) with an error of 5% of the height, and
the volumes from Schumacher–Hall, ``v = 5 \times 10^{-5} d^{1.9} h^{1.0}``, with a multiplicative error.

```@example tutorial
using ForestModeling, DataFrames, Random

rng = MersenneTwister(42)
plots, trees = 30, 15
plot = repeat(1:plots, inner=trees)
u = 3 .* randn(rng, plots)
d = 5 .+ 35 .* rand(rng, plots * trees)
μ = [chapmanrichards(d[i], [25 + u[plot[i]], 0.07, 1.1]; href=1.3) for i in eachindex(d)]
h = μ .* (1 .+ 0.05 .* randn(rng, length(d)))
v = schumacherhall.(d, h, Ref([5e-5, 1.9, 1.0])) .* exp.(0.08 .* randn(rng, length(d)))

data = DataFrame(plot=string.(plot), d=d .* u"cm", h=h .* u"m", v=v .* u"m^3")
first(data, 5)
```

## Volume: a generalized linear model

Schumacher–Hall as a generalized linear model: the log link gives the power shape and the Gamma
family the variance growing with the volume. The response is the volume itself, not its logarithm.

```@example tutorial
volume = fitglm(@formula(v ~ log(d) + log(h)), data)
```

The exponents are close to the true 1.9 and 1.0. The prediction is the mean volume, in m³, with no
back-transformation; with `interval=:confidence`, the interval of the mean:

```@example tutorial
predict(volume, first(data, 3); interval=:confidence)
```

## Height: a nonlinear regression

Chapman–Richards with the variance of the error growing with a power ``k`` of the height. The
diameters enter in cm and the heights in m, so `href` is in m:

```@example tutorial
cr(d, θ) = chapmanrichards(d, θ; href=1.3)
height = fitnls(cr, data, :h, :d, [20.0, 0.05, 1.0])
```

The prediction interval of a new tree is wider than the confidence interval of the mean:

```@example tutorial
predict(height, first(data, 3); interval=:prediction)
```

The same function fits an equation of two predictors, Schumacher–Hall in its nonlinear form:

```@example tutorial
fitnls(schumacherhall, data, :v, (:d, :h), [1e-4, 2.0, 1.0])
```

## Plots: a mixed model and its calibration

The asymptote changes from plot to plot. A mixed model with a random intercept for each plot
captures it:

```@example tutorial
mixed = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), data)
```

In a new plot where only two heights were measured, [`calibrate`](@ref) estimates the random effect of
the plot from those two trees and predicts all of them; without calibration the plot gets the
population curve:

```@example tutorial
newplot = transform(data[data.plot .== "1", :], :plot => (p -> fill("new", length(p))) => :plot)
measured = newplot[1:2, :]
DataFrame(h=newplot.h, population=predict(mixed, newplot), calibrated=calibrate(mixed, measured, newplot))
```

## Validation by plot

[`blockcv`](@ref) predicts the plots of each fold with the model fitted without them. The error of
the mixed model out of its plots is larger than the error in its plots, where it uses their random
effects:

```@example tutorial
fitheight(t) = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), t)
(fit=metrics(data.h, predict(mixed, data)), cv=blockcv(fitheight, data, :h, :plot; rng=MersenneTwister(1)))
```

## One model per category

[`fitbygroup`](@ref) fits a model for each level of a column, in parallel threads:

```@example tutorial
data.stand = ifelse.(parse.(Int, data.plot) .<= 15, "A", "B")
fitbygroup(t -> fitglm(@formula(v ~ log(d) + log(h)), t), data, :stand)
```

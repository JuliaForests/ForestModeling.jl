"""
    blockcv(fit, data, y, block; k=5, rng=Random.default_rng())

Block cross-validation: the levels of the column `block` (plot, stand) are shuffled and split into
`k` folds, the rows of each fold are predicted by the model fitted without them, and the
[`metrics`](@ref) of these out-of-fold predictions of the column `y` are returned. `fit` takes a
`DataFrame` and returns a model of the package.

```math
\\text{RMSE}_{CV} = \\sqrt{\\frac{1}{n}\\sum_{f=1}^{k}\\sum_{i \\in B_f}\\left(y_i - \\hat y_i^{(-B_f)}\\right)^2}
```

Splitting by block, not by tree, keeps the trees of a plot together: the trees of one plot are more
alike than those of different plots, and a random split would predict each tree with its own plot
in the fit, an error smaller than that of a new plot. A mixed model predicts the plots left out
with the population curve, as it would a new plot without calibration.

# Example
With `heights`, the table of the tutorial, each of its four plots predicted by the other three:
```julia
blockcv(t -> fitnls(curtis, t, :h, :d, [3.5, -10.0]), heights, :h, :plot; k=4)
```
"""
function blockcv(fit, data, y::Symbol, block::Symbol; k::Integer=5, rng=Random.default_rng())
  levels = unique(data[!, block])
  length(levels) >= k ||
    throw(ArgumentError("There are $(length(levels)) levels of $block for k = $k folds."))
  fold = Dict(zip(shuffle(rng, levels), mod1.(eachindex(levels), k)))
  folds = [fold[b] for b in data[!, block]]
  ŷ = Vector{Any}(undef, nrow(data))
  for f in 1:k
    test = folds .== f
    ŷ[test] = predict(fit(data[.!test, :]), data[test, :])
  end
  metrics(data[!, y], identity.(ŷ))
end

"""
    metrics(y, ŷ)

Error statistics of the predictions `ŷ` of `y`, on the original scale and in the unit of `y`: the
root mean squared error, the bias and the mean absolute percentage error (%),

```math
\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_i (y_i - \\hat y_i)^2}, \\qquad
\\text{bias} = \\frac{1}{n}\\sum_i (y_i - \\hat y_i), \\qquad
\\text{MAPE} = \\frac{100}{n}\\sum_i \\left|\\frac{y_i - \\hat y_i}{y_i}\\right|
```

A positive bias is a model that underestimates.

# Example
```jldoctest
julia> metrics([10.0, 12.0, 15.0], [11.0, 12.0, 13.5])
(rmse = 1.0408329997330663, bias = 0.16666666666666666, mape = 6.666666666666667)
```
"""
function metrics(y, ŷ)
  checklengths(:y => y, :ŷ => ŷ)
  e = y .- ŷ
  n = length(e)
  (rmse=sqrt(sum(abs2, e) / n), bias=sum(e) / n, mape=100 * sum(NoUnits.(abs.(e ./ y))) / n)
end

"""
    graybilltest(y, ŷ)

F test of Graybill (1976) that the predictions `ŷ` equal the observed `y`: the regression of the
predicted on the observed values,

```math
\\hat y_i = \\beta_0 + \\beta_1 y_i + \\varepsilon_i, \\qquad H_0: (\\beta_0, \\beta_1) = (0, 1)
```

```math
F = \\frac{(\\hat{\\boldsymbol\\beta} - \\boldsymbol\\theta)^{\\top} \\mathbf{X}^{\\top}\\mathbf{X}
(\\hat{\\boldsymbol\\beta} - \\boldsymbol\\theta)}{2 s^2} \\sim F_{2,\\, n-2},
\\qquad \\boldsymbol\\theta = (0, 1)
```

with ``\\mathbf X = [\\mathbf 1 \\; \\mathbf y]`` and ``s^2`` the residual variance of the regression,
as in Leite & Oliveira (2002). A small `pvalue` rejects that the predictions equal the observed
values: a model biased or with a slope other than 1. Returns `β0` (in the unit of `y`), `β1`, `F`
and `pvalue`.

# Example
With `volume`, the table of the tutorial:
```julia
graybilltest(volume.v, predict(fitglm(@formula(v ~ log(d) + log(h)), volume), volume))
```
"""
function graybilltest(y, ŷ)
  checklengths(:y => y, :ŷ => ŷ)
  u = unit(first(y))
  Y, Ŷ = ustrip.(u, y), ustrip.(u, ŷ)
  n = length(Y)
  X = [ones(n) Y]
  β = X \ Ŷ
  s² = sum(abs2, Ŷ - X * β) / (n - 2)
  δ = β - [0, 1]
  F = δ' * (X' * X) * δ / (2s²)
  (β0=β[1] * u, β1=β[2], F=F, pvalue=ccdf(FDist(2, n - 2), F))
end

"""
    identitytest(fit, data, group)

Test of the identity of models (Regazzi, 1993, 2003): whether the levels of the column(s) `group`
(clones, stands, species) can share one equation. `fit` takes a `DataFrame` and returns a model of
the package; the complete model fits each group apart ([`fitbygroup`](@ref)) and the reduced one
fits all the data together, compared by the likelihood ratio,

```math
\\lambda = 2\\left(\\sum_{g=1}^{G} \\ell_g - \\ell_{r}\\right) \\sim \\chi^2_{\\nu}, \\qquad
\\nu = \\sum_{g=1}^{G} p_g - p_r
```

with ``p`` the number of parameters of each model, those of the variance included. A small
`pvalue` rejects one equation for all the groups. The complete model also gives each group its
own variance, so the test compares the whole models, the curve and its spread. Returns `lr`
(``\\lambda``), `df` (``\\nu``) and `pvalue`.

A categorical column in the formula of [`fitglm`](@ref) tests a part of the equation instead: with
`@formula(h ~ inv(d) + clone)` each clone has its intercept and the same slope.

# Example
With `heights`, the table of the tutorial, whether the four plots share one Curtis curve:
```julia
identitytest(t -> fitnls(curtis, t, :h, :d, [3.5, -10.0]), heights, :plot)
```
"""
function identitytest(fit, data, group)
  reduced = fit(data)
  complete = fitbygroup(fit, data, group).model
  λ = 2 * (sum(loglikelihood, complete) - loglikelihood(reduced))
  ν = sum(dof, complete) - dof(reduced)
  (lr=λ, df=ν, pvalue=ccdf(Chisq(ν), λ))
end

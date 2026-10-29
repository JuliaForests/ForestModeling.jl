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
```julia
blockcv(t -> fitglm(@formula(v ~ log(d) + log(h)), t), data, :v, :plot)
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

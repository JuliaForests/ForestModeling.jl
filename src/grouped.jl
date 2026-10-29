"""
    fitbygroup(fit, data, group)

Fits a model separately for each level of the column(s) `group` (species, stand, clone), each one in
its own thread. `fit` takes the rows of one group and returns its model,

```math
\\hat{\\boldsymbol\\theta}_g = \\arg\\max_{\\boldsymbol\\theta} \\ell(\\boldsymbol\\theta; \\mathbf{y}_g), \\qquad g = 1, \\dots, G
```

and the result is a `DataFrame` with a row per group: its level(s) and its model in the column
`model`. A group gets its own curve with no information from the others; with few trees per group,
or groups that were not sampled, a mixed model ([`fitmixed`](@ref)) is the alternative.

# Example
With `heights`, the table of the tutorial, a Curtis curve for each plot:
```julia
models = fitbygroup(t -> fitnls(curtis, t, :h, :d, [3.5, -10.0]), heights, :plot)
predict(only(models[models.plot .== "A", :model]), newdata)
```
"""
function fitbygroup(fit, data, group)
  groups = groupby(data, group)
  tasks = [Threads.@spawn fit(g) for g in groups]
  result = DataFrame([NamedTuple(k) for k in keys(groups)])
  result.model = fetch.(tasks)
  result
end

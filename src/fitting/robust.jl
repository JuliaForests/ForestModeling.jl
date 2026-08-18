# ==============================================================================
# ROBUST / ALTERNATIVE ESTIMATION CRITERIA (Optim.jl-backed, opt-in)
# ==============================================================================

"""
    lossKernel(yᵣ, ŷ, method::AllometricMethod) -> Float64

Loss value minimized by [`fitRobust`](@ref)'s numerical optimizer. `OLS`
itself never reaches this function — it is always the closed-form
[`fitOLS`](@ref) path.
"""
function lossKernel(yᵣ::AbstractVector{<:Real}, ŷ::AbstractVector{<:Real}, method::AllometricMethod)
  if method == SSE
    return sum(abs2, yᵣ .- ŷ)
  elseif method == MAE
    return sum(abs, yᵣ .- ŷ)
  elseif method == MSLE
    (any(v -> v <= eps(), ŷ) || any(v -> v <= 0, yᵣ)) && return Inf
    return sum(abs2, log.(yᵣ) .- log.(ŷ))
  elseif method == MAPE
    any(v -> abs(v) < eps(), yᵣ) && return Inf
    return sum(abs, (yᵣ .- ŷ) ./ yᵣ)
  elseif method == HUBER
    ε = yᵣ .- ŷ
    δ = 1.345 * median(abs.(ε))
    δ < 1e-9 && return sum(abs2, ε)
    loss = 0.0
    @inbounds for r in ε
      a = abs(r)
      loss += a <= δ ? 0.5r^2 : δ * (a - 0.5δ)
    end
    return loss
  else
    error("OLS is not routed through the numerical optimizer")
  end
end

"""
    fitRobust(formula::FormulaTerm, data, method::AllometricMethod; contrasts=Dict(), nonNegative=true)

Fit a single model by minimizing [`lossKernel`](@ref) with `Optim.jl` instead
of closed-form OLS — for the cases where OLS assumptions are visibly
violated (heavy outliers → `HUBER`/`MAE`; strongly relative error →
`MSLE`/`MAPE`). Opt-in: `regression`/`fitCombinations` never call this.

Starting values come from the closed-form OLS fit on the same design, which
is almost always a good enough starting point for these loss surfaces.

!!! warning
    The returned `Σ` is **not** a robust/sandwich covariance matrix — treat
    `confint`/`pvalue` on a `HUBER`/`MAE`/`MSLE`/`MAPE` fit as provisional
    until that is revisited.
"""
function fitRobust(formula::FormulaTerm, data, method::AllometricMethod; contrasts=Dict{Symbol,Any}(), nonNegative::Bool=true)
  method == OLS && return fit(AllometricModel, formula, data; contrasts, nonNegative)

  data, columnUnits = stripcolumnunits(data)
  mf = ModelFrame(formula, data; model=AllometricModel, contrasts=contrasts)
  f = mf.f
  yt = f.lhs
  if isa(yt, FunctionTerm)
    name = nameof(yt.f)
    name ∉ (:log, :inv, :sqrt, :petterson, :naslund, :prodan) && throw(ArgumentError(
      "transform :$name on the response has no defined bias correction; " *
      "supported: (:log, :inv, :sqrt, :petterson, :naslund, :prodan)"))
  end

  cols = mf.data
  units = columnUnits[keys(cols)]
  yᵣ = cols[1]
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)
  X = modelmatrix(mf)
  y = modelcols(yt, cols)                             # transformed response — the loss below must be
                                                        # minimized on the same scale `fit`/`fitOLS` use,
                                                        # not the raw response, or the bias correction
                                                        # further down would be back-transforming values
                                                        # that were never on the transformed scale to begin with
  n, p = size(X)
  ν = n - p

  β0, _, _, _ = fitOLS(X, y)                          # OLS start point
  objective(β) = lossKernel(y, X * β, method)
  result = optimize(objective, β0, LBFGS(); autodiff=AutoForwardDiff())
  β = Optim.minimizer(result)

  ŷ = X * β
  ε = y .- ŷ
  σ² = (ε ⋅ ε) / ν                                    # transformed scale — see the note in fitCombinations

  if isa(yt, FunctionTerm)
    x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
    predictBiasCorrected!(ŷ, x, yt, σ²)
    ε = yᵣ .- ŷ
  end
  nonNegative && any(<(0), ŷ) && throw(ErrorException("fitted values contain negative predictions"))

  Σ = σ² .* inv(X'X)                                  # sandwich-free covariance; sanity check before inference
  stats = computeStatistics(yᵣ, ŷ, ε, n, ν, sst, ȳ)
  return AllometricModel(f, cols, units, β, Σ, σ², n, ν, p, stats.sse, sst, stats.r², stats.adjr²,
    stats.ev, stats.mae, stats.mape, stats.mse, stats.rmse, stats.cv, stats.normal)
end

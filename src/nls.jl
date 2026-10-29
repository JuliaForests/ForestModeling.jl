"""
    NLSFit

Result of [`fitnls`](@ref): the equation `f`, the estimates of its parameters `θ`, of the variance
function (`σ`, `k`) and of the covariance of `θ`, the log-likelihood, and the data of the fit
without units, with the unit of each column.
"""
struct NLSFit{F}
  f::F
  θ::Vector{Float64}
  σ::Float64
  k::Float64
  vcov::Matrix{Float64}
  loglik::Float64
  p::Vector{Float64}
  response::Symbol
  predictors::Vector{Symbol}
  units::Dict{Symbol,Any}
  y::Vector{Float64}
  x::Vector
end

# The terms of the negative log-likelihood of each observation, with the parameters
# p = (θ..., log σ) and the standard deviation of the error σ, or p = (θ..., log σ, k) and σ|μ|^k;
# the constant variance never takes the logarithm of μ, which may be 0 (the tip of a stem)
function _nllterms(f, p, q, x, y)
  θ = p[1:q]
  logσ = p[q+1]
  power = length(p) > q + 1
  map(x, y) do xi, yi
    μ = f(xi..., θ)
    logs = power ? logσ + p[q+2] * log(abs(μ)) : logσ
    logs + (yi - μ)^2 / (2 * exp(2logs)) + log(2π) / 2
  end
end

"""
    fitnls(f, data, y, x, θ0; variance=:power)

Nonlinear regression of the column `y` on the column(s) `x` fitted by maximum likelihood. The
standard deviation of the error grows with a power of the mean (`variance=:power`) or is constant
(`variance=:constant`, the ordinary nonlinear least squares):

```math
y_i = f(x_i; \\boldsymbol\\theta) + \\varepsilon_i, \\qquad
\\varepsilon_i \\sim N\\!\\left(0, \\sigma^2 \\mu_i^{2k}\\right), \\qquad \\mu_i = f(x_i; \\boldsymbol\\theta)
```

```math
\\ell(\\boldsymbol\\theta, \\sigma, k) = -\\sum_{i=1}^{n}\\left[\\ln\\!\\left(\\sigma \\mu_i^{k}\\right)
+ \\frac{\\left(y_i - \\mu_i\\right)^2}{2\\sigma^2 \\mu_i^{2k}}\\right] - \\frac{n}{2}\\ln 2\\pi
```

The heteroscedasticity is in the model, so the prediction stays on the original scale. The fit has
two steps: the least squares of the constant variance (``k = 0``) by Levenberg–Marquardt
(`LsqFit.jl`), robust to the initial values, and from them the maximum of ``\\ell`` with ``k`` by
L-BFGS (`Optim.jl`). The gradients come from `ForwardDiff.jl` and the covariance of the estimates
is the inverse of the Hessian of ``-\\ell``.

`f(x..., θ)` is any function of the predictors and of the vector of parameters: an equation of the
package ([`chapmanrichards`](@ref), [`schumacherhall`](@ref), ...) or one written by the user. `x`
is a column (`:d`) or a tuple of columns (`(:d, :h)`). `θ0` are the initial values of `θ`. The
columns may be `Unitful` quantities: `f` receives their values in the unit of each column, so
`href` is given in the unit of the response.

# Example
With `heights` and `volume`, the tables of the tutorial:
```julia
fitnls(curtis, heights, :h, :d, [3.5, -10.0])                     # h = curtis(d; θ = [4.266, -34.76])
fitnls(schumacherhall, volume, :v, (:d, :h), [1e-4, 2.0, 1.0])    # v = schumacherhall(d, h; θ = [6.008e-6, 1.982, 1.57])
cr(d, θ) = chapmanrichards(d, θ; href=1.3)                        # a curve with an asymptote, heights in m,
fitnls(cr, data, :h, :d, [25.0, 0.05, 1.0])                       # for diameters where the height levels off
```
"""
function fitnls(f, data, y::Symbol, x, θ0; variance::Symbol=:power)
  variance in (:power, :constant) ||
    throw(ArgumentError("variance must be :power or :constant (got :$variance)."))
  predictors = x isa Symbol ? [x] : collect(Symbol, x)
  units = _units(data, [y; predictors])
  table = _ustrip(data, units)
  Y = Float64.(table[!, y])
  X = collect(zip((Float64.(table[!, c]) for c in predictors)...))
  q = length(θ0)
  nll(p) = sum(_nllterms(f, p, q, X, Y))
  # the least squares of the constant variance, then the power of the variance from them
  ls = curve_fit((rows, θ) -> [f(X[i]..., θ) for i in rows], eachindex(Y), Y, float.(θ0); autodiff=AutoForwardDiff())
  ls.converged || @warn "The least squares did not converge; try other initial values θ0."
  p = [ls.param; log(sqrt(mean(abs2, ls.resid)))]
  if variance === :power
    result = optimize(nll, (G, p) -> ForwardDiff.gradient!(G, nll, p), [p; 0.0], LBFGS())
    Optim.converged(result) || @warn "The variance function did not converge."
    p = Optim.minimizer(result)
  end
  Σ = inv(ForwardDiff.hessian(nll, p))
  NLSFit(f, p[1:q], exp(p[q+1]), variance === :power ? p[q+2] : 0.0, Σ[1:q, 1:q], -nll(p), p,
         y, predictors, units, Y, X)
end

coef(m::NLSFit) = m.θ
vcov(m::NLSFit) = m.vcov
stderror(m::NLSFit) = sqrt.(diag(m.vcov))
nobs(m::NLSFit) = length(m.y)
residuals(m::NLSFit) = (m.y .- [m.f(xi..., m.θ) for xi in m.x]) .* m.units[m.response]
loglikelihood(m::NLSFit) = m.loglik
dof(m::NLSFit) = length(m.p)

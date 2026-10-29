"""
    robustvcov(model)

Sandwich (heteroskedasticity-consistent) covariance of the parameters, valid when the variance of
the error is not well modeled:

```math
\\hat{\\boldsymbol\\Sigma}_{HC} = \\mathbf{A}^{-1}\\mathbf{B}\\,\\mathbf{A}^{-1}, \\qquad
\\mathbf{A} = \\sum_i \\nabla^2 \\ell_i, \\qquad \\mathbf{B} = \\sum_i \\nabla \\ell_i \\nabla \\ell_i^{\\top}
```

where ``\\ell_i`` is the term of the observation ``i`` in the negative log-likelihood. With a
correct variance model it is close to `vcov(model)`; when they differ, the standard errors of
`vcov` are not reliable.

- [`fitglm`](@ref): the HC3 estimator of `CovarianceMatrices.jl`.
- [`fitnls`](@ref): ``\\mathbf{A}`` and ``\\mathbf{B}`` by `ForwardDiff.jl`, for the parameters `θ`.

# Example
```julia
sqrt.(diag(robustvcov(model)))   # robust standard errors
```
"""
robustvcov(model::UnitfulModel{<:StatsModels.TableRegressionModel}) =
  CovarianceMatrices.vcov(CovarianceMatrices.HC3(), model.model)

function robustvcov(model::NLSFit)
  q = length(model.θ)
  terms(p) = _nllterms(model.f, p, q, model.x, model.y)
  A = ForwardDiff.hessian(p -> sum(terms(p)), model.p)
  J = ForwardDiff.jacobian(terms, model.p)
  Ainv = inv(A)
  (Ainv*(J'*J)*Ainv)[1:q, 1:q]
end

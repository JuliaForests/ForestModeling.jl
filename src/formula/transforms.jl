# ==============================================================================
# TRANSFORM CATALOG (bounded, citable — not a combinatorial free-for-all)
# ==============================================================================
# Every transform below has a named citation in classical forestry biometrics
# (Curtis 1967; Prodan; Petterson; Näslund; Scolforo 2005 — Biometria
# Florestal). Kept intentionally small: this is the search space for
# `fitCombinations`/`regression`, so every candidate curve shape should be
# defensible on its own, not just "one more thing the optimizer could try".

"""
    petterson(y) = 1 / √y

Linearizing transform for the Petterson hypsometric form: `1/√h = β₀ + β₁/d`.
"""
petterson(y::Real) = inv(√y)

"""
    naslund(y, x) = x / √y

Linearizing transform for the Näslund hypsometric form: `d/√h = β₀ + β₁·d`.
"""
naslund(y::Real, x::Real) = x / √y

"""
    prodan(y, x) = x² / y

Linearizing transform for the Prodan hypsometric form: `d²/h = β₀ + β₁·d + β₂·d²`.
"""
prodan(y::Real, x::Real) = x^2 / y

"""
    expandXTerms(x::AbstractTerm) -> Vector{AbstractTerm}

Bounded set of independent-variable transforms: identity, quadratic, square
root, natural log, and inverse (first and second order). These five cover the
shape families (polynomial, root, logarithmic, asymptotic) used across the
classic named hypsometric/volumetric equations in Brazilian and Scandinavian
forestry literature.
"""
expandXTerms(x::AbstractTerm) = AbstractTerm[
  x,
  FunctionTerm(z -> z^2, [x], :($(x)^2)),
  FunctionTerm(sqrt, [x], :(√$(x))),
  FunctionTerm(log, [x], :(log($(x)))),
  FunctionTerm(inv, [x], :($(x)^-1)),
  FunctionTerm(z -> inv(z^2), [x], :($(x)^-2)),
]

"""
    transformYTerms(y::AbstractTerm, xTerms::Vector{<:AbstractTerm}) -> Vector{AbstractTerm}

Bounded set of dependent-variable transforms: identity, log, inverse, square
root, plus the three named forestry linearizations (`petterson`, `naslund`,
`prodan`, the last two built against every available `x` term).
"""
function transformYTerms(y::AbstractTerm, xTerms::Vector{<:AbstractTerm})
  yList = AbstractTerm[
    y,
    FunctionTerm(log, [y], :(log($y))),
    FunctionTerm(inv, [y], :($y^-1)),
    FunctionTerm(sqrt, [y], :(√$y)),
    FunctionTerm(petterson, [y], :(√($y)^-1)),
  ]
  for x in xTerms
    push!(yList, FunctionTerm(naslund, [y, x], :($x / √($y))))
    push!(yList, FunctionTerm(prodan, [y, x], :(($x)^2 / $y)))
  end
  return yList
end

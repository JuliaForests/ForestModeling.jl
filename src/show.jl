# The fitted equation in one line: the coefficients with 4 significant digits, each with its term,
# inside the inverse of the link
function _equation(response, β, names, inverse)
  terms = map(β, names) do b, name
    value = string(round(abs(b); sigdigits=4))
    name == "(Intercept)" ? value : value * " * " * name
  end
  η = (signbit(β[1]) ? "-" : "") * terms[1] *
      join((signbit(b) ? " - " : " + ") * t for (b, t) in zip(β[2:end], terms[2:end]))
  string(response, " = ", inverse(η))
end

_inverse(::IdentityLink) = identity
_inverse(::LogLink) = η -> "exp($η)"
_inverse(::InverseLink) = η -> "1 / ($η)"
_inverse(link) = η -> "$(nameof(typeof(link)))⁻¹($η)"

function _equation(m::UnitfulModel)
  model = m.model
  if model isa MixedModel
    random = filter(t -> t isa MixedModels.RandomEffectsTerm, collect(model.formula.rhs))
    return _equation(m.response, coef(model), coefnames(model), identity) * join(" + " .* string.(random))
  end
  link = model.model isa GeneralizedLinearModel ? GLM.Link(model.model) : IdentityLink()
  _equation(m.response, coef(model), coefnames(model), _inverse(link))
end

# the equation alone, the form of a cell of a table
Base.show(io::IO, m::UnitfulModel) = print(io, _equation(m))

# the equation, the table of the package of the fit (the coefficients of a GLM, the whole summary of
# a mixed model) and the unit of the response
function Base.show(io::IO, ::MIME"text/plain", m::UnitfulModel)
  println(io, _equation(m), "\n")
  show(io, MIME("text/plain"), m.model isa MixedModel ? m.model : coeftable(m.model))
  print(io, "\n\nResponse: ", m.response, " (", m.units[m.response], "), n = ", round(Int, nobs(m)))
end

# the name of the equation, or f for an anonymous function
function _name(f)
  name = string(nameof(f))
  startswith(name, "#") ? "f" : name
end

Base.show(io::IO, m::NLSFit) =
  print(io, m.response, " = ", _name(m.f), "(", join(m.predictors, ", "), "; θ = ",
        round.(m.θ; sigdigits=4), ")")

function Base.show(io::IO, ::MIME"text/plain", m::NLSFit)
  println(io, m, "\n")
  println(io, "Nonlinear regression of ", m.response, " (", m.units[m.response], ") on ",
          join(m.predictors, ", "), " (n = ", nobs(m), ")")
  for (i, (θ, s)) in enumerate(zip(m.θ, stderror(m)))
    println(io, "  θ[", i, "] = ", θ, "  (standard error ", s, ")")
  end
  print(io, "  σ = ", m.σ, ", k = ", m.k, ", log-likelihood = ", m.loglik)
end

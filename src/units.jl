# The unit of each numeric column among `columns` (`NoUnits` for plain numbers); the other
# columns (a plot, a clone) are left out
function _units(data, columns)
  units = Dict{Symbol,Any}()
  for c in columns
    x = first(skipmissing(data[!, c]))
    x isa Number && (units[c] = x isa Quantity ? unit(x) : NoUnits)
  end
  units
end

# The columns of `data` with the units of `units` removed, the values in those units (a
# diameter in mm is read in the cm of the fit); the other columns are kept as they are
function _ustrip(data, units)
  table = DataFrame(data; copycols=false)
  for (c, u) in units
    hasproperty(table, c) && (table[!, c] = ustrip.(u, table[!, c]))
  end
  table
end

# The response of a formula, which must be a column: the link function, not a
# transformation of the response, gives the relation on the original scale
function _response(formula::FormulaTerm)
  formula.lhs isa Term || throw(ArgumentError(
    "The response must be a column, not a function of it ($(formula.lhs)); use the link function instead."))
  formula.lhs.sym
end

"""
    UnitfulModel

A model of `GLM.jl` or `MixedModels.jl` with the units of the columns it was fitted with: the
predictions and the residuals come in the unit of the response. The fitted model of the package is
the field `model`, for everything else it offers (`coeftable`, `loglikelihood`, ...).
"""
struct UnitfulModel{M}
  model::M
  response::Symbol
  units::Dict{Symbol,Any}
end

coef(m::UnitfulModel) = coef(m.model)
vcov(m::UnitfulModel) = vcov(m.model)
stderror(m::UnitfulModel) = stderror(m.model)
nobs(m::UnitfulModel) = nobs(m.model)
residuals(m::UnitfulModel) = (response(m.model) .- fitted(m.model)) .* m.units[m.response]

function Base.show(io::IO, m::UnitfulModel)
  show(io, MIME("text/plain"), m.model)
  print(io, "\n\nResponse: ", m.response, " (", m.units[m.response], ")")
end

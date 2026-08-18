# ==============================================================================
# DISPLAY
# ==============================================================================

termLabels(model::AllometricModel) = vcat("β0", string.(StatsModels.coefnames(model.formula.rhs.terms[2:end])))

function Base.show(io::IO, model::AllometricModel)
  β = model.β
  labels = termLabels(model)
  out = string(StatsModels.coefnames(model.formula.lhs)) * " = $(round(β[1], digits=4))"
  for i in 2:length(β)
    term = " $(round(abs(β[i]), sigdigits=4)) * $(labels[i])"
    out *= signbit(β[i]) ? " -" * term : " +" * term
  end
  print(io, out)
end

"""
    summary(io, model::AllometricModel)
    summary(model::AllometricModel)

Full statistical report: equation, coefficient table, goodness-of-fit
metrics (original scale), and the normality diagnostic.
"""
function Base.summary(io::IO, model::AllometricModel)
  println(io, "Allometric Regression Model")
  println(io, "-"^50)
  print(io, "Equation: "); show(io, model); println(io, "\n")
  println(io, "Coefficients:"); show(io, coeftable(model)); println(io, "\n")
  println(io, "Goodness-of-Fit (original scale):")
  @printf(io, "  R² (generalized): %.4f\n", model.r²)
  @printf(io, "  Adjusted R²:      %.4f\n", model.adjr²)
  @printf(io, "  Explained Var.:   %.4f\n", model.ev)
  @printf(io, "  RMSE:             %.4f\n", model.rmse)
  @printf(io, "  CV%%:              %.2f%%\n", model.cv)
  @printf(io, "  MAE:              %.4f\n", model.mae)
  @printf(io, "  MAPE:             %.2f%%\n", model.mape)
  println(io, "\nDiagnostics:")
  println(io, "  Normality (transformed-scale residuals): ", model.normal ? "pass" : "fail")
  println(io, "-"^50)
end
Base.summary(model::AllometricModel) = summary(stdout, model)

"""
    metrics(model::AllometricModel)
    metrics(models::Vector{<:AllometricModel})

Column table (`NamedTuple` of vectors) of the equation and every
goodness-of-fit statistic — the shape `criteriaTable` builds its `DataFrame`
from.
"""
function metrics(model::AllometricModel)
  return (equation=[sprint(show, model)], r²=[model.r²], adjr²=[model.adjr²], ev=[model.ev],
    rmse=[model.rmse], cv=[model.cv], mse=[model.mse], mae=[model.mae], mape=[model.mape],
    sse=[model.sse], sst=[model.sst], n=[model.n], p=[model.p], dof=[model.ν], normal=[model.normal])
end
function metrics(models::Vector{<:AllometricModel})
  return (equation=sprint.(show, models), r²=getfield.(models, :r²), adjr²=getfield.(models, :adjr²),
    ev=getfield.(models, :ev), rmse=getfield.(models, :rmse), cv=getfield.(models, :cv),
    mse=getfield.(models, :mse), mae=getfield.(models, :mae), mape=getfield.(models, :mape),
    sse=getfield.(models, :sse), sst=getfield.(models, :sst), n=getfield.(models, :n),
    p=getfield.(models, :p), dof=getfield.(models, :ν), normal=getfield.(models, :normal))
end

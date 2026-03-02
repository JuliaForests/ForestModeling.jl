using StatsModels
using BenchmarkTools
using GLM
using Optim
# using Combinatorics
using Distributions
using DataFrames
using StatsBase
import StatsModels: TableRegressionModel
using CSV
import Base: show

function show(io::IO, ::MIME"text/plain", v::Vector{<:TableRegressionModel})
  n = length(v)
  println(io, "$n-element Vector{TableRegressionModel}:")

  isConsole = get(io, :limit, false)

  for i in 1:n
    if isConsole && i > 5 && i < n - 4
      if i == 6
        println(io, "  ⋮")
      end
      continue
    end

    m = v[i]
    formulaStr = string(m.mf.f)
    linkObj = GLM.Link(m.model)
    linkName = string(nameof(typeof(linkObj)))

    if linkObj isa PowerLink
      lambdaVal = round(linkObj.λ, digits=4)
      print(io, "  [$i] $formulaStr | Link: $linkName(λ = $lambdaVal)\n")
    else
      print(io, "  [$i] $formulaStr | Link: $linkName\n")
    end
  end
end

# McFadden pseudo-R²
function GLM.r2(model::TableRegressionModel)
  nullDev = nulldeviance(model.model)
  resDev = deviance(model.model)
  return 1 - (resDev / nullDev)
end

# Adjusted McFadden pseudo-R²
function GLM.adjr2(model::TableRegressionModel)
  n = nobs(model.model)
  p = dof(model.model)
  r2Val = GLM.r2(model)
  return 1 - (1 - r2Val) * ((n - 1) / (n - p))
end

const β0 = InterceptTerm{true}()

# Static functions to prevent closure allocation
pow2(x::Float64) = x^2
pow3(x::Float64) = x^3
logPow2(x::Float64) = log(x)^2
logPow3(x::Float64) = log(x)^3
invPow2(x::Float64) = inv(x^2)
invPow3(x::Float64) = inv(x^3)
crossMul(x::Float64, y::Float64) = x * y
crossSq1(x::Float64, y::Float64) = (x^2) * y
crossSq2(x::Float64, y::Float64) = x * (y^2)
crossDiv1(x::Float64, y::Float64) = x / y
crossDiv2(x::Float64, y::Float64) = y / x
crossLog(x::Float64, y::Float64) = log(x) * log(y)

function prepareModelData(mf::ModelFrame)
  yTerm = mf.f.lhs
  yTerm isa CategoricalTerm && throw(ArgumentError("dependent variable must be continuous"))
  cols = mf.data
  nObs = length(cols[1])
  contTerms = filter(t -> t isa ContinuousTerm, mf.f.rhs.terms)
  isempty(contTerms) && throw(ArgumentError("no continuous independent variables provided"))
  catTerms = filter(t -> t isa CategoricalTerm, mf.f.rhs.terms)
  nVars = length(contTerms)
  nUnivariate = 12 * nVars
  nInteractions = (nVars * (nVars - 1) ÷ 2) * 5
  totalEstimate = nUnivariate + nInteractions
  # build all terms in logical groups while avoiding manual idx increments
  allTerms = Vector{AbstractTerm}()
  sizehint!(allTerms, totalEstimate)
  termGroups = Vector{UnitRange{Int}}()
  # univariate groups
  for t in contTerms
    start = length(allTerms) + 1
    push!(allTerms, t)
    push!(allTerms, FunctionTerm(pow2, (t,), Expr(:call, :^, t, 2)))
    push!(allTerms, FunctionTerm(pow3, (t,), Expr(:call, :^, t, 3)))
    push!(allTerms, FunctionTerm(sqrt, (t,), Expr(:call, :sqrt, t)))
    push!(allTerms, FunctionTerm(cbrt, (t,), Expr(:call, :cbrt, t)))
    push!(allTerms, FunctionTerm(log, (t,), Expr(:call, :log, t)))
    push!(allTerms, FunctionTerm(logPow2, (t,), Expr(:call, :^, Expr(:call, :log, t), 2)))
    push!(allTerms, FunctionTerm(logPow3, (t,), Expr(:call, :^, Expr(:call, :log, t), 3)))
    push!(allTerms, FunctionTerm(log1p, (t,), Expr(:call, :log1p, t)))
    push!(allTerms, FunctionTerm(inv, (t,), Expr(:call, :^, t, -1)))
    push!(allTerms, FunctionTerm(invPow2, (t,), Expr(:call, :^, t, -2)))
    push!(allTerms, FunctionTerm(invPow3, (t,), Expr(:call, :^, t, -3)))
    push!(termGroups, start:length(allTerms))
  end
  # interaction groups
  for i in 1:nVars-1
    t1 = contTerms[i]
    for j in i+1:nVars
      t2 = contTerms[j]
      start = length(allTerms) + 1
      push!(allTerms, FunctionTerm(crossMul, (t1, t2), Expr(:call, :*, t1, t2)))
      push!(allTerms, FunctionTerm(crossSq1, (t1, t2), Expr(:call, :*, Expr(:call, :^, t1, 2), t2)))
      push!(allTerms, FunctionTerm(crossSq2, (t1, t2), Expr(:call, :*, t1, Expr(:call, :^, t2, 2))))
      push!(allTerms, FunctionTerm(crossDiv1, (t1, t2), Expr(:call, :/, t1, t2)))
      push!(allTerms, FunctionTerm(crossDiv2, (t1, t2), Expr(:call, :/, t2, t1)))
      push!(allTerms, FunctionTerm(crossLog, (t1, t2), Expr(:call, :*, Expr(:call, :log, t1), Expr(:call, :log, t2))))
      push!(termGroups, start:length(allTerms))
    end
  end
  # evaluate terms, keep only valid columns and remap groups to new indices
  validXTerms = AbstractTerm[]
  xCols = Vector{Vector{Float64}}()
  validTermGroups = Vector{Vector{Int}}()
  for group in termGroups
    groupNew = Int[]
    for idx in group
      t = allTerms[idx]
      try
        col = modelcols(t, cols)
        all(isfinite, col) || continue
        push!(xCols, col)
        push!(validXTerms, t)
        push!(groupNew, length(validXTerms))
      catch
        # skip invalid transform
      end
    end
    if !isempty(groupNew)
      push!(validTermGroups, groupNew)
    end
  end
  isempty(xCols) && throw(ArgumentError("no valid continuous transformations found"))
  nCols = length(xCols)
  # Pre-allocate matrix with an extra column for the intercept
  xPool = Matrix{Float64}(undef, nObs, nCols + 1)
  # Fill the intercept at column 1
  fill!(view(xPool, :, 1), 1.0)
  # Fill the transformations starting at column 2
  @inbounds for j in 1:nCols
    copyto!(view(xPool, :, j + 1), xCols[j])
  end
  qMatrix = isempty(catTerms) ? Matrix{Float64}(undef, nObs, 0) : StatsModels.modelmatrix(sum(catTerms), cols)
  return (xPool, validXTerms, catTerms, qMatrix, cols, validTermGroups)
end

# crossLog(x1, x2) = log(x1) * log(x2)

# function prepareModelData(mf::ModelFrame)
#   yTerm = mf.f.lhs
#   yTerm isa CategoricalTerm && throw(ArgumentError("dependent variable must be continuous"))

#   cols = mf.data
#   nObs = length(cols[1])

#   contTerms = filter(t -> t isa ContinuousTerm, mf.f.rhs.terms)
#   isempty(contTerms) && throw(ArgumentError("no continuous independent variables provided"))

#   catTerms = filter(t -> t isa CategoricalTerm, mf.f.rhs.terms)

#   nVars = length(contTerms)
#   nUnivariate = 5 * nVars
#   nInteractions = (nVars * (nVars - 1) ÷ 2) * 4
#   totalEstimate = nUnivariate + nInteractions

#   allTerms = Vector{AbstractTerm}()
#   sizehint!(allTerms, totalEstimate)

#   termGroups = Vector{UnitRange{Int}}()

#   for t in contTerms
#     start = length(allTerms) + 1

#     push!(allTerms, t)
#     push!(allTerms, FunctionTerm(pow2, (t,), Expr(:call, :^, t, 2)))
#     push!(allTerms, FunctionTerm(sqrt, (t,), Expr(:call, :sqrt, t)))
#     push!(allTerms, FunctionTerm(log, (t,), Expr(:call, :log, t)))
#     push!(allTerms, FunctionTerm(log1p, (t,), Expr(:call, :log1p, t)))

#     push!(termGroups, start:length(allTerms))
#   end

#   for i in 1:nVars-1
#     t1 = contTerms[i]
#     for j in i+1:nVars
#       t2 = contTerms[j]
#       start = length(allTerms) + 1

#       push!(allTerms, FunctionTerm(crossMul, (t1, t2), Expr(:call, :*, t1, t2)))
#       push!(allTerms, FunctionTerm(crossSq1, (t1, t2), Expr(:call, :*, Expr(:call, :^, t1, 2), t2)))
#       push!(allTerms, FunctionTerm(crossSq2, (t1, t2), Expr(:call, :*, t1, Expr(:call, :^, t2, 2))))
#       push!(allTerms, FunctionTerm(crossLog, (t1, t2), Expr(:call, :*, Expr(:call, :log, t1), Expr(:call, :log, t2))))

#       push!(termGroups, start:length(allTerms))
#     end
#   end

#   validXTerms = AbstractTerm[]
#   xCols = Vector{Vector{Float64}}()
#   validTermGroups = Vector{Vector{Int}}()

#   for group in termGroups
#     groupNew = Int[]
#     for idx in group
#       t = allTerms[idx]
#       try
#         col = modelcols(t, cols)
#         all(isfinite, col) || continue
#         push!(xCols, col)
#         push!(validXTerms, t)
#         push!(groupNew, length(validXTerms))
#       catch
#       end
#     end
#     if !isempty(groupNew)
#       push!(validTermGroups, groupNew)
#     end
#   end

#   isempty(xCols) && throw(ArgumentError("no valid continuous transformations found"))

#   nCols = length(xCols)
#   xPool = Matrix{Float64}(undef, nObs, nCols + 1)

#   fill!(view(xPool, :, 1), 1.0)

#   @inbounds for j in 1:nCols
#     copyto!(view(xPool, :, j + 1), xCols[j])
#   end

#   qMatrix = isempty(catTerms) ? Matrix{Float64}(undef, nObs, 0) : StatsModels.modelmatrix(sum(catTerms), cols)

#   return (xPool, validXTerms, catTerms, qMatrix, cols, validTermGroups)
# end

###########
yName = "Btot"

xNames = ["dbh", "heig"]

contrasts = Dict{Symbol,Any}()

modeltype = GeneralizedLinearModel
dist = Normal()

############


ySym = Symbol(yName)
xSyms = collect(Symbol.(xNames))

# create formula
f = term(ySym) ~ sum(term.(xSyms))

# build model frame
mf = ModelFrame(f, data; model=modeltype, contrasts=contrasts)


pm = prepareModelData(mf)
# julia> @btime prepareModelData($mf)
#   22.500 μs (355 allocations: 37.01 KiB)

# data.dbh = -data.dbh

# mf = ModelFrame(f, data; model=modeltype, contrasts=contrasts)

# pm = prepareModelData(mf)
# julia> @btime prepareModelData($mf)
#   40.600 μs (374 allocations: 36.15 KiB)

xPool = pm[1]
validXTerms = pm[2]
validTermGroups = pm[6]

yTarget = response(mf)
yTerm = mf.f.lhs
distTarget = Normal()

nGroups = length(validTermGroups)

if nGroups == 1
  comboIter = collect(powerset(validTermGroups[1], 1, 2))
else
  comboIter = [collect(c) for c in Iterators.product(validTermGroups...)]
end

nCombos = length(comboIter)

# preallocate maximum possible size (3 links per combination)
maxModels = nCombos * 3
fittedModels = Vector{TableRegressionModel}(undef, maxModels)

# atomic counter ensures threads claim unique index safely
validCount = Atomic{Int}(1)

Threads.@threads for i in 1:nCombos
  combo = comboIter[i]

  colsToKeep = Int[1]
  for idx in combo
    push!(colsToKeep, idx + 1)
  end

  xMat = xPool[:, colsToKeep]

  comboTerms = AbstractTerm[InterceptTerm{true}()]
  for idx in combo
    push!(comboTerms, validXTerms[idx])
  end

  fNew = FormulaTerm(yTerm, Tuple(comboTerms))
  mfNew = ModelFrame(fNew, mf.schema, mf.data, GeneralizedLinearModel)

  assignVec = collect(1:length(colsToKeep))
  mmNew = ModelMatrix(xMat, assignVec)

  # identity link adjustment
  try
    rawIdentity = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, IdentityLink())
    idx = atomic_add!(validCount, 1)
    fittedModels[idx] = TableRegressionModel(rawIdentity, mfNew, mmNew)
  catch
  end

  # log link adjustment
  try
    rawLog = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, LogLink())
    idx = atomic_add!(validCount, 1)
    fittedModels[idx] = TableRegressionModel(rawLog, mfNew, mmNew)
  catch
  end

  # power link adjustment
  try
    objective = lambda -> begin
      try
        tempModel = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(lambda))
        return bic(tempModel)
      catch
        return Inf
      end
    end

    optRes = optimize(objective, -5.0, 1.0, Brent())
    bestLambda = Optim.minimizer(optRes)

    rawPower = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(bestLambda))
    idx = atomic_add!(validCount, 1)
    fittedModels[idx] = TableRegressionModel(rawPower, mfNew, mmNew)
  catch
  end
end



# shrink array to exact number of successfully fitted models
resize!(fittedModels, validCount[] - 1)

# sort fittedModels by bic values
# sort!(fittedModels, by=m -> bic(m.model))



# sort fittedModels by bic values
sort!(fittedModels, by=m -> adjr2(m), rev=true)


nCombos = length(comboIter)

# preallocate maximum possible size (3 links per combination)
maxModels = nCombos * 3

fittedModels = Vector{TableRegressionModel}(undef, maxModels)

Threads.@threads for i in 1:nCombos
  combo = comboIter[i]

  colsToKeep = Int[1]
  for idx in combo
    push!(colsToKeep, idx + 1)
  end

  xMat = xPool[:, colsToKeep]

  comboTerms = AbstractTerm[InterceptTerm{true}()]
  for idx in combo
    push!(comboTerms, validXTerms[idx])
  end

  fNew = FormulaTerm(yTerm, Tuple(comboTerms))
  mfNew = ModelFrame(fNew, mf.schema, mf.data, GeneralizedLinearModel)

  assignVec = collect(1:length(colsToKeep))
  mmNew = ModelMatrix(xMat, assignVec)

  # fit identity link
  try
    rawIdentity = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, IdentityLink())
    idxStored = atomic_add!(validCount, 1)
    fittedModels[idxStored] = TableRegressionModel(rawIdentity, mfNew, mmNew)
  catch
    # ignore failed identity fits
  end

  # fit log link
  try
    rawLog = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, LogLink())
    idxStored = atomic_add!(validCount, 1)
    fittedModels[idxStored] = TableRegressionModel(rawLog, mfNew, mmNew)
  catch
    # ignore failed log fits
  end

  # fit power link with penalized criterion (BIC by default)
  try
    # choose criterion: useBIC = true for stronger complexity penalty
    useBIC = true
    lb = -5.0
    ub = 2.0
    tol_bound = 1e-6

    objective = λ -> begin
      try
        mtmp = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(λ))
        return useBIC ? bic(mtmp) : aic(mtmp)
      catch
        return Inf
      end
    end

    optRes = optimize(objective, lb, ub, Brent())
    bestLambda = Optim.minimizer(optRes)

    # if optimizer hits the boundary, clamp to bounds
    if abs(bestLambda - lb) < tol_bound
      bestLambda = lb
    elseif abs(bestLambda - ub) < tol_bound
      bestLambda = ub
    end

    rawPower = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(bestLambda))
    idxStored = atomic_add!(validCount, 1)
    fittedModels[idxStored] = TableRegressionModel(rawPower, mfNew, mmNew)
  catch
    # ignore failures in power-link search/fit
  end
end


sort!(fittedModels, by=m -> adjr2(m), rev=true)

sort!(fittedModels, by=m -> Link(m.model).λ)


fittedModels






# preallocate maximum possible size (3 links per combination, though PowerLink runs less often)
maxModels = nCombos * 3

fittedModels = Vector{TableRegressionModel}(undef, maxModels)
validCount = Atomic{Int}(1)

Threads.@threads for i in 1:nCombos
  combo = comboIter[i]

  colsToKeep = Int[1]
  for idx in combo
    push!(colsToKeep, idx + 1)
  end

  xMat = xPool[:, colsToKeep]

  comboTerms = AbstractTerm[InterceptTerm{true}()]
  for idx in combo
    push!(comboTerms, validXTerms[idx])
  end

  fNew = FormulaTerm(yTerm, Tuple(comboTerms))
  mfNew = ModelFrame(fNew, mf.schema, mf.data, GeneralizedLinearModel)

  assignVec = collect(1:length(colsToKeep))
  mmNew = ModelMatrix(xMat, assignVec)

  # fit identity link (runs for all combinations)
  try
    rawIdentity = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, IdentityLink())
    idxStored = atomic_add!(validCount, 1)
    fittedModels[idxStored] = TableRegressionModel(rawIdentity, mfNew, mmNew)
  catch
    # ignore failed identity fits
  end

  # fit log link (runs for all combinations)
  try
    rawLog = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, LogLink())
    idxStored = atomic_add!(validCount, 1)
    fittedModels[idxStored] = TableRegressionModel(rawLog, mfNew, mmNew)
  catch
    # ignore failed log fits
  end

  # fit power link (STRICTLY for single-variable models to preserve biological meaning)
  if length(combo) == 1
    try
      # choose criterion: useBIC = true for stronger complexity penalty
      useBIC = true
      lb = -5.0
      ub = 2.0
      tol_bound = 1e-6

      objective = λ -> begin
        try
          mtmp = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(λ))
          return useBIC ? bic(mtmp) : aic(mtmp)
        catch
          return Inf
        end
      end

      optRes = optimize(objective, lb, ub, Brent())
      bestLambda = Optim.minimizer(optRes)

      # if optimizer hits the boundary, clamp to bounds
      if abs(bestLambda - lb) < tol_bound
        bestLambda = lb
      elseif abs(bestLambda - ub) < tol_bound
        bestLambda = ub
      end

      rawPower = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(bestLambda))
      idxStored = atomic_add!(validCount, 1)
      fittedModels[idxStored] = TableRegressionModel(rawPower, mfNew, mmNew)
    catch
      # ignore failures in power-link search/fit
    end
  end
end

# shrink array to exact number of successfully fitted models
resize!(fittedModels, validCount[] - 1)








import Base: show

# Custom struct to hold the fitted model
struct FittedBiometricModel
  model::TableRegressionModel
  link_name::String
  lambda::Union{Float64,Nothing}
end

# Show method for a single model
function show(io::IO, bm::FittedBiometricModel)
  link_str = bm.link_name
  if bm.link_name == "PowerLink" && bm.lambda !== nothing
    link_str *= "(λ = $(round(bm.lambda, digits=4)))"
  end

  formula_str = string(bm.model.mf.f)
  print(io, "FittedBiometricModel: $formula_str | Link: $link_str")
end

# Show method for a Vector of our custom models
function show(io::IO, ::MIME"text/plain", v::Vector{FittedBiometricModel})
  n = length(v)
  println(io, "$n-element Vector{FittedBiometricModel}:")

  if n <= 10
    for i in 1:n
      print(io, "  [$i] ")
      show(io, v[i])
      println(io)
    end
  else
    for i in 1:5
      print(io, "  [$i] ")
      show(io, v[i])
      println(io)
    end
    println(io, "  ⋮")
    for i in (n-4):n
      print(io, "  [$i] ")
      show(io, v[i])
      println(io)
    end
  end
end

# ... previous setup code ...

function test()
  maxModels = nCombos * 3
  # Update the array type
  fittedModels = Vector{FittedBiometricModel}(undef, maxModels)

  validCount = Atomic{Int}(1)

  Threads.@threads for i in 1:nCombos
    combo = comboIter[i]

    colsToKeep = Int[1]
    for idx in combo
      push!(colsToKeep, idx + 1)
    end

    xMat = xPool[:, colsToKeep]

    comboTerms = AbstractTerm[InterceptTerm{true}()]
    for idx in combo
      push!(comboTerms, validXTerms[idx])
    end

    fNew = FormulaTerm(yTerm, Tuple(comboTerms))
    mfNew = ModelFrame(fNew, mf.schema, mf.data, GeneralizedLinearModel)

    assignVec = collect(1:length(colsToKeep))
    mmNew = ModelMatrix(xMat, assignVec)

    # identity link adjustment
    try
      rawIdentity = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, IdentityLink())
      idx = atomic_add!(validCount, 1)
      fittedModels[idx] = FittedBiometricModel(TableRegressionModel(rawIdentity, mfNew, mmNew), "IdentityLink", nothing)
    catch
    end

    # log link adjustment
    try
      rawLog = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, LogLink())
      idx = atomic_add!(validCount, 1)
      fittedModels[idx] = FittedBiometricModel(TableRegressionModel(rawLog, mfNew, mmNew), "LogLink", nothing)
    catch
    end

    # power link adjustment
    try
      objective = lambda -> begin
        try
          tempModel = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(lambda))
          return deviance(tempModel)
        catch
          return Inf
        end
      end

      optRes = optimize(objective, -2.0, 1.0, Brent())
      bestLambda = Optim.minimizer(optRes)

      rawPower = fit(GeneralizedLinearModel, xMat, yTarget, distTarget, PowerLink(bestLambda))
      idx = atomic_add!(validCount, 1)
      fittedModels[idx] = FittedBiometricModel(TableRegressionModel(rawPower, mfNew, mmNew), "PowerLink", bestLambda)
    catch
    end
  end

  resize!(fittedModels, validCount[] - 1)
  return fittedModels
end

# 227.258 ms (1834736 allocations: 128.10 MiB)


# atual
using DataFrames
using StatsModels
using Combinatorics
using LinearAlgebra
using Distributions
using Statistics
using Tables
using CSV
using BenchmarkTools
# --- structure definition ---

struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,I<:Int64,B<:Bool} <: RegressionModel
  formula::F
  data::N
  β::Vector{T}
  Σ::Matrix{T}
  σ²::T
  n::I
  ν::I
  p::I
  sse::T
  sst::T
  r2::T
  adjr2::T
  ev::T
  mae::T
  mape::T
  mse::T
  rmse::T
  cv::T
  normality::B
end

import Base: show, summary

_coefnames(model::AllometricModel) = vcat("β0", string.(StatsModels.coefnames(model.formula.rhs.terms[2:end])))

# Display the equation of the fitted linear model.
function show(io::IO, model::AllometricModel)
  β = model.β
  n = length(β)
  output = string(StatsModels.coefnames(model.formula.lhs)) * " = $(round(β[1], digits = 4))"

  for i in 2:n
    term = _coefnames(model)[i]
    product = string(round(abs(β[i]), sigdigits=4)) * " * " * term
    output *= signbit(β[i]) ? " - $(product)" : " + $(product)"
  end

  print(io, output)
end

const S = Union{Symbol,String}

const β0 = InterceptTerm{true}()

function xTerms(t::AbstractTerm)
  return AbstractTerm[
    t,
    FunctionTerm(x -> x^2, [t], :($(t)^2)),
    FunctionTerm(x -> x^3, [t], :($(t)^3)),
    FunctionTerm(sqrt, [t], :(√$(t))),
    FunctionTerm(cbrt, [t], :(∛$(t))),
    FunctionTerm(log, [t], :(log($(t)))),
    FunctionTerm(x -> log(x)^2, [t], :(log($(t))^2)),
    FunctionTerm(x -> log(x)^3, [t], :(log($(t))^3)),
    FunctionTerm(log1p, [t], :(log1p($(t)))),
    FunctionTerm(inv, [t], :($(t)^-1)),
    FunctionTerm(x -> inv(x^2), [t], :($(t)^-2)),
    FunctionTerm(x -> inv(x^3), [t], :($(t)^-3))
  ]
end

function yTerms(t::AbstractTerm)
  return AbstractTerm[
    t,
    FunctionTerm(log, [t], :(log($(t)))),
    FunctionTerm(log1p, [t], :(log1p($(t)))),
    FunctionTerm(sqrt, [t], :(√$(t))),
    FunctionTerm(cbrt, [t], :(∛$(t)))
  ]
end

function predict!(ŷ::Vector{<:Real}, xcol::Union{AbstractVector{<:Real},Nothing}, ft::FunctionTerm, σ²::Real)
  fname = nameof(ft.f)
  n = length(ŷ)
  # If no transformation, do nothing (ŷ remains unchanged)
  if fname ∉ (:log, :log1p, :inv, :sqrt, :cbrt, :petterson, :naslund, :prodan)
    return
  end
  @inbounds @simd for i in 1:n
    z = ŷ[i]
    if fname == :log
      # log-normal (exact)
      ŷ[i] = exp(z + 0.5σ²)
    elseif fname == :log1p
      # We use expm1 for numerical precision
      ŷ[i] = expm1(z + 0.5σ²)
    elseif fname == :inv
      # inverse (1/y)
      # g(z) = 1/z,  ∂²g = 2/z^3
      g = 1 / z
      ∂²g = 2 / (z^3)
      ŷ[i] = g + 0.5σ² * ∂²g
    elseif fname == :sqrt
      # Y = z^2 + correction
      # g'' = 2
      ŷ[i] = (z^2) + σ²
    elseif fname == :cbrt
      # Y = z^3 + correction
      # g'' = 6z
      # corr = 0.5 * sig2 * 6z = 3 * z * sig2
      ŷ[i] = (z^3) + (3 * z * σ²)
    elseif fname == :petterson
      # inverse sqrt (1/√y)
      # g(z) = 1/z^2,  ∂²g = 6/z^4
      z² = z^2
      g = 1 / z²
      ∂²g = 6 / (z²^2)
      ŷ[i] = g + 0.5σ² * ∂²g
    elseif fname == :naslund
      # x/√y
      # g(z) = x^2/z^2,  ∂²g = 6x^2/z^4
      x = xcol[i]
      x² = x^2
      z² = z^2
      g = x² / z²
      ∂²g = (6 * x²) / (z²^2)
      ŷ[i] = g + 0.5σ² * ∂²g
    elseif fname == :prodan
      # x^2/y
      # g(z) = x^2/z,  ∂²g = 2x^2/z^3
      x = xcol[i]
      x² = x^2
      g = x² / z
      ∂²g = (2 * x²) / (z^3)
      ŷ[i] = g + 0.5σ² * ∂²g
    end
  end
end

function prepareModelData(df::DataFrame, yname::Symbol, xnames::Vector{Symbol}, contrasts=Dict{Symbol,Any}())
  # create formula with all variables
  f = term(yname) ~ sum(term.(xnames))
  # build model frame once
  mf = ModelFrame(f, df; contrasts=contrasts)
  # extract typed terms
  yTerm = mf.f.lhs
  if yTerm isa CategoricalTerm
    throw(ArgumentError("dependent variable must be continuous"))
  end
  # extract rhs terms
  allTerms = mf.f.rhs.terms
  # separate continuous and categorical terms
  conTerms = filter(t -> t isa ContinuousTerm, allTerms)
  if isempty(conTerms)
    throw(ArgumentError("no continuous independent variables provided"))
  end
  catTerms = filter(t -> t isa CategoricalTerm, allTerms) |> collect
  # create named tuple for fast column access
  cols = mf.data
  nObs = length(cols[1])
  # expand all continuous terms
  xCandidates = mapreduce(xTerms, vcat, conTerms)
  # initialize accumulator with intercept column
  validColumns = Vector{Float64}[fill(1.0, nObs)]
  validXTerms = AbstractTerm[]
  # iterate over x candidates
  for t in xCandidates
    try
      colData = modelcols(t, cols)
      # check for domain errors
      if all(isfinite, colData)
        push!(validColumns, colData)
        push!(validXTerms, t)
      end
    catch
      # ignore invalid transformations
    end
  end
  if length(validColumns) == 1
    throw(ArgumentError("no valid continuous transformations found"))
  end
  # combine vector of vectors into a single matrix
  xPool = reduce(hcat, validColumns)
  # build categorical matrix
  if isempty(catTerms)
    qMatrix = Matrix{Float64}(undef, nObs, 0)
  else
    # modelmatrix handles one hot encoding and contrasts automatically
    qMatrix = modelmatrix(sum(catTerms), df)
  end
  # prepare y variants
  yCandidates = yTerms(yTerm)
  # strict typing for y variants container
  yVariants = Vector{Tuple{AbstractTerm,Vector{Float64}}}()
  for t in yCandidates
    try
      yVec = modelcols(t, cols)

      # check for domain errors
      if all(isfinite, yVec)
        push!(yVariants, (t, yVec))
      end
    catch
      # ignore invalid y transformations
    end
  end
  if isempty(yVariants)
    throw(ArgumentError("no valid dependent variable transformations found"))
  end
  return (xPool, validXTerms, catTerms, qMatrix, yVariants, cols)
end

function fitCombinations(
  xPool::Matrix{Float64},
  validXTerms::Vector{<:AbstractTerm},
  catTerms::Vector{<:AbstractTerm},
  qMatrix::Matrix{Float64},
  yVariants::Vector{Tuple{AbstractTerm,Vector{Float64}}},
  cols::NamedTuple;
  nmin::Int=2,
  nmax::Int=2,
  nonnegative::Bool=true)
  # validate range limit
  nFeatures = length(validXTerms)
  if nmax > nFeatures
    nmax = nFeatures
  end
  # generate all combinations indices upfront
  combos = Vector{Vector{Int}}()
  sizehint!(combos, 1000)
  for k in nmin:nmax
    for idxs in combinations(1:nFeatures, k)
      push!(combos, collect(idxs))
    end
  end
  nx = length(combos)
  ny = length(yVariants)
  # pre allocate output matrix safe for concurrent writes via disjoint indices
  resultsMatrix = Matrix{Union{Missing,AllometricModel}}(undef, ny, nx)
  # constants and dimensions
  nQ = size(qMatrix, 2)
  hasQ = nQ > 0
  maxP = nmax + 1 + nQ
  # constant dependent variable metrics
  yReal = cols[1]
  nObs = length(yReal)
  yMean = mean(yReal)
  varYReal = var(yReal)
  sstReal = sum(abs2, yReal .- yMean)
  catSym = hasQ ? sum(catTerms) : nothing
  # partition work into chunks based on available threads
  nThreads = Threads.nthreads()
  chunkSize = max(1, cld(nx, nThreads))
  chunks = Iterators.partition(1:nx, chunkSize)
  # spawn a task for each chunk ensuring isolated memory
  tasks = map(chunks) do chunk
    Threads.@spawn begin
      # allocate local buffers once per task
      xBuf = Matrix{Float64}(undef, nObs, maxP)
      xtxBuf = Matrix{Float64}(undef, maxP, maxP)
      xtyBuf = Vector{Float64}(undef, maxP)
      betaBuf = Vector{Float64}(undef, maxP)
      covBuf = Matrix{Float64}(undef, maxP, maxP)
      yHatBuf = Vector{Float64}(undef, nObs)
      residBuf = Vector{Float64}(undef, nObs)
      # process assigned indices
      for ix in chunk
        idxs = combos[ix]
        k = length(idxs)
        # build design matrix in place
        nCont = k + 1
        p = nCont + nQ
        dof = nObs - p
        if dof <= 0
          for iy in 1:ny
            resultsMatrix[iy, ix] = missing
          end
          continue
        end
        # copy intercept
        copyto!(view(xBuf, :, 1), view(xPool, :, 1))
        # copy continuous features
        for (i, src) in enumerate(idxs)
          copyto!(view(xBuf, :, i + 1), view(xPool, :, src + 1))
        end
        # copy categorical block
        if hasQ
          copyto!(view(xBuf, :, (nCont+1):p), qMatrix)
        end
        xDesign = view(xBuf, :, 1:p)
        # linear algebra prep
        xtx = view(xtxBuf, 1:p, 1:p)
        cov = view(covBuf, 1:p, 1:p)
        xty = view(xtyBuf, 1:p)
        beta = view(betaBuf, 1:p)
        # compute cross product
        mul!(xtx, xDesign', xDesign)
        # cholesky decomposition with error handling
        chol = try
          cholesky!(Symmetric(xtx))
        catch
          for iy in 1:ny
            resultsMatrix[iy, ix] = missing
          end
          continue
        end
        # construct symbolic formula
        rhs = mapfoldl(i -> validXTerms[i], +, idxs, init=β0)
        if hasQ
          rhs += catSym
        end
        rhsSym = MatrixTerm(rhs)
        # iterate over dependent variable variants
        for iy in 1:ny
          (yTerm, yVec) = yVariants[iy]
          # compute rhs vector
          mul!(xty, xDesign', yVec)
          # solve coefficients
          copyto!(beta, xty)
          ldiv!(chol, beta)
          # predictions on transformed scale
          mul!(yHatBuf, xDesign, beta)
          @. residBuf = yVec - yHatBuf
          sseTrans = dot(residBuf, residBuf)
          sigma2 = sseTrans / dof
          # covariance calculation optimization
          fill!(cov, 0.0)
          @inbounds for i in 1:p
            cov[i, i] = 1.0
          end
          ldiv!(chol, cov)
          rmul!(cov, sigma2)
          normality = true
          # bias correction
          if isa(yTerm, FunctionTerm)
            interaction = length(yTerm.args) > 1 ? cols[yTerm.args[2].sym] : nothing
            predict!(yHatBuf, interaction, yTerm, sigma2)
            @. residBuf = yReal - yHatBuf
            sse = dot(residBuf, residBuf)
          else
            sse = sseTrans
          end
          # nonnegative check
          if nonnegative
            negFound = false
            @inbounds for v in yHatBuf
              if v < 0.0
                negFound = true
                break
              end
            end
            if negFound
              resultsMatrix[iy, ix] = missing
              continue
            end
          end
          # manual calculation of metrics to avoid allocation
          absSum = 0.0
          @inbounds for r in residBuf
            absSum += abs(r)
          end
          mae = absSum / nObs
          apeSum = 0.0
          @inbounds for i in 1:nObs
            apeSum += abs(residBuf[i]) / abs(yReal[i])
          end
          mape = (apeSum / nObs) * 100.0
          r2 = 1.0 - (sse / sstReal)
          adjr2 = 1.0 - (1.0 - r2) * (nObs - 1) / dof
          if adjr2 <= 0.0
            resultsMatrix[iy, ix] = missing
            continue
          end
          ev = 1.0 - (var(residBuf) / varYReal)
          mse = sse / dof
          rmse = sqrt(mse)
          cv = (rmse / yMean) * 100.0
          # store result
          resultsMatrix[iy, ix] = AllometricModel(
            FormulaTerm(yTerm, rhsSym),
            cols,
            Vector(beta),
            Matrix(cov),
            sigma2,
            nObs, dof, p,
            sse, sstReal,
            r2, adjr2, ev,
            mae, mape, mse, rmse, cv,
            normality
          )
        end
      end
    end
  end
  # wait for all threads to finish
  wait.(tasks)
  # flatten and return valid models
  models = collect(skipmissing(vec(resultsMatrix)))
  if isempty(models)
    error("failed to fit all models")
  end
  return models
end

function regression(data, yname::S, xnames::S...; contrasts=Dict{Symbol,Any}(), nmin::Int=2, nmax::Int=2, nonnegative::Bool=true)
  # Input Validation
  if !Tables.istable(data)
    throw(ArgumentError("data argument must be a Tables.jl compatible table"))
  end
  if isempty(xnames)
    throw(ArgumentError("no independent variables provided"))
  end
  if nmin < 1
    throw(ArgumentError("nmin must be >= 1"))
  end
  if nmax < nmin
    throw(ArgumentError("nmax must be >= nmin"))
  end
  if nmax > 5
    throw(ArgumentError("nmax ($nmax) exceeds the practical limit for allometric models (max 5). "
                        * "Models with more than 5 terms (e.g., degree > 5) cause severe overfitting, "
                        * "numerical instability (singular matrices), and lack biological meaning."))
  end
  yname = Symbol(yname)
  xnames = Symbol.(xnames)
  yname = Symbol(yname)
  xnames = Symbol.(xnames) |> collect
  fitCombinations(
    prepareModelData(data, yname, xnames, contrasts)...,
    nmin=nmin,
    nmax=nmax,
    nonnegative=nonnegative
  )
end

# julia> @btime reg = regression(df, :ht, :dap);
#   1.062 ms (7870 allocations: 735.15 KiB)
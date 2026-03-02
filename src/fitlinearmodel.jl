function regression(
  data,
  yName::Union{Symbol,AbstractString},
  xNames::Union{Symbol,AbstractString}...;
  contrasts=Dict{Symbol,Any}(),
  nmin::Int=2,
  nmax::Int=2,
  allpositive::Bool=true)
  # input validation
  if !istable(data)
    throw(ArgumentError("data argument must be a tables jl compatible table"))
  end
  if isempty(xNames)
    throw(ArgumentError("no independent variables provided"))
  end
  if nmin < 1
    throw(ArgumentError("nmin must be >= 1"))
  end
  if nmax > 5
    throw(ArgumentError("nmax exceeds practical limits"))
  end

  ySym = Symbol(yName)
  xSyms = collect(Symbol.(xNames))

  # create formula
  f = term(ySym) ~ sum(term.(xSyms))

  # build model frame
  mf = ModelFrame(f, data; contrasts=contrasts)

  # run pipeline
  return fitCombinations(
    prepareModelData(mf)...,
    nmin,
    nmax,
    allpositive
  )
end

function fitCombinations(
  xPool::Matrix{Float64},
  validXTerms::Vector{<:AbstractTerm},
  catTerms::Vector{<:AbstractTerm},
  qMatrix::Matrix{Float64},
  yVariants::Vector{Tuple{AbstractTerm,Vector{Float64}}},
  cols::NamedTuple,
  termIndices::Vector{Vector{Int}},
  nmin::Int,
  nmax::Int,
  allpositive::Bool)

  if length(termIndices) == 1
    group = termIndices[1]
    limitMax = min(nmax, length(group))
    comboIter = powerset(group, nmin, limitMax)
  else
    comboIter = Iterators.product(termIndices...)
  end

  combos = collect(comboIter)
  nx = length(combos)
  ny = length(yVariants)

  fittedmodels = Matrix{Union{Missing,AllometricModel}}(undef, ny, nx)

  nQ = size(qMatrix, 2)
  hasQ = nQ > 0
  maxP = (length(termIndices) == 1 ? nmax : length(termIndices)) + 1 + nQ

  yᵣ = cols[1]
  n = length(yᵣ)
  ȳ = mean(yᵣ)
  varyᵣ = var(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)

  nThreads = Threads.nthreads()
  chunkSize = max(1, cld(nx, nThreads))
  chunks = Iterators.partition(1:nx, chunkSize)

  tasks = map(chunks) do chunk
    Threads.@spawn begin

      XBuf = Matrix{Float64}(undef, n, maxP)
      fill!(view(XBuf, :, 1), 1.0)
      xtxBuf = Matrix{Float64}(undef, maxP, maxP)
      xtyBuf = Vector{Float64}(undef, maxP)
      βBuf = Vector{Float64}(undef, maxP)
      ΣBuf = Matrix{Float64}(undef, maxP, maxP)
      ŷ = Vector{Float64}(undef, n)
      ε = Vector{Float64}(undef, n)

      for ix in chunk
        idxs = combos[ix]
        k = length(idxs)

        nCont = k + 1
        p = nCont + nQ
        ν = n - p

        if ν <= 0
          @inbounds for iy in 1:ny
            fittedmodels[iy, ix] = missing
          end
          continue
        end

        tValue = quantile(TDist(ν), 0.975)

        @inbounds for (i, srcIdx) in enumerate(idxs)
          @views copyto!(XBuf[:, i+1], xPool[:, srcIdx])
        end

        hasQ && @views copyto!(XBuf[:, nCont+1:p], qMatrix)

        X = @view XBuf[:, 1:p]
        xtx = @view xtxBuf[1:p, 1:p]
        xty = @view xtyBuf[1:p]
        β = @view βBuf[1:p]
        Σ = @view ΣBuf[1:p, 1:p]

        mul!(xtx, X', X)

        chol = try
          cholesky!(Symmetric(xtx))
        catch
          @inbounds for iy in 1:ny
            fittedmodels[iy, ix] = missing
          end
          continue
        end

        formula_rhs = if hasQ
          sum(i -> validXTerms[i], idxs, init=β0) + sum(catTerms)
        else
          sum(i -> validXTerms[i], idxs, init=β0)
        end |> MatrixTerm

        for iy in 1:ny
          (yt, y) = yVariants[iy]
          # compute X transpose y for normal equations
          mul!(xty, X', y)
          # initialize beta with Xty
          copyto!(β, xty)
          # solve for regression coefficients β in-place using the Cholesky factor
          ldiv!(chol, β)
          # compute predicted values in-place (ŷ = X * β)
          mul!(ŷ, X, β)
          # compute residuals on transformed scale
          @. ε = y - ŷ
          # sum of squared errors on transformed scale
          sse = ε ⋅ ε
          # residual variance on transformed scale
          σ² = sse / ν
          # compute dispersion matrix (covariance of coefficients)
          fill!(Σ, 0.0)
          @inbounds for i in 1:p
            Σ[i, i] = 1.0
          end
          ldiv!(chol, Σ)
          rmul!(Σ, σ²)
          # correct the predicted values and residuals for models with a function on the left-hand side of the formula
          if isa(yt, FunctionTerm)
            applyBiasCorrection!(ŷ, yt, σ²)
            # recalculate residuals on original scale
            @. ε = yᵣ - ŷ
            # recalculate sum of squared errors on original scale
            sse = ε ⋅ ε
          end
          # check for non negative fitted values if required
          if allpositive
            neg = false
            @inbounds for v in ŷ
              if v < 0.0
                neg = true
                break
              end
            end
            if neg
              fittedmodels[iy, ix] = missing
              continue
            end
          end
          # calculate mean absolute error and mean absolute percentage error loops
          mae = 0.0
          ape = 0.0
          @inbounds @simd for i in 1:n
            r = abs(ε[i])
            mae += r
            ape += r / abs(yᵣ[i])
          end
          # finalize mean absolute error
          mae /= n
          # finalize mean absolute percentage error
          mape = (ape / n) * 100.0
          # compute the coefficient of determination (R²)
          r² = 1.0 - (sse / sst)
          # compute the adjusted R², penalized for the number of predictors
          adjr² = 1.0 - (1.0 - r²) * (n - 1) / ν
          if adjr² <= 0.0
            fittedmodels[iy, ix] = missing
            continue
          end
          # calculate explained variance score
          ev = 1.0 - (var(ε) / varyᵣ)
          # calculate mean squared error
          mse = sse / ν
          # calculate root mean squared error
          rmse = sqrt(mse)
          # calculate coefficient of variation percentage
          cv = (rmse / ȳ) * 100.0
          # calculate relative standard error of estimate (Syx%)
          syx = (rmse / sqrt(n)) / ȳ * 100.0
          # calculate uncertainty percentage at 95% confidence using T-distribution
          uncertainty = tValue * syx
          # construct and store the fitted model object
          fittedmodels[iy, ix] = AllometricModel(
            FormulaTerm(yt, formula_rhs),
            cols,
            Vector(β),
            Matrix(Σ),
            σ²,
            n, ν, p,
            sse, sst,
            r², adjr²,
            ev, mae,
            mape, mse,
            rmse, cv,
            syx, uncertainty
          )
        end
      end
    end
  end

  wait.(tasks)

  models = AllometricModel[]
  sizehint!(models, nx * ny)

  @inbounds for j in 1:nx, i in 1:ny
    m = fittedmodels[i, j]
    m !== missing && push!(models, m)
  end

  isempty(models) && error("failed to fit all Allometric Linear Regression Models")

  return models
end


function fit(::Type{AllometricModel}, formula::FormulaTerm, data; contrasts=Dict{Symbol,Any}(), nonnegative::Bool=true)
  # Construct Model Frame (Handle missing data, categorical contrasts, etc.)
  mf = ModelFrame(formula, data; model=AllometricModel, contrasts=contrasts)
  # Extract Components from the processed ModelFrame
  formula = mf.f
  # Dependent Variable Term
  yt = formula.lhs
  # Validate supported transformations on dependent variable
  if isa(yt, FunctionTerm)
    fname = nameof(yt.f)
    if fname ∉ (:log, :log1p, :inv, :sqrt, :cbrt, :petterson, :naslund, :prodan)
      throw(ArgumentError("""
        Transformation ':$fname' on the dependent variable is not supported by AllometricModel.
        We only support transformations with defined bias correction methods.
        Allowed: (:log, :log1p, :inv, :sqrt, :cbrt, :petterson, :naslund, :prodan)
      """))
    end
  end
  # Data Columns
  cols = mf.data
  # the real dependent variable
  yᵣ = cols[1]
  # total sum of squares
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)
  # Build Model Matrix (X) and Transformed Y Columns (y)
  X = modelmatrix(mf)
  y = modelcols(yt, cols)
  # Determine the number of observations (n) and the number of predictors (p)
  (n, p) = size(X)
  # Calculate the degrees of freedom for residuals
  ν = n - p
  # Allocate space for regression coefficients β
  β = Vector{Float64}(undef, p)
  # Allocate space for predicted values (ŷ) 
  ŷ = Vector{Float64}(undef, n)
  # Allocate space for residuals (ε)
  ε = Vector{Float64}(undef, n)

  try
    chol = cholesky!(Symmetric(X'X))
    invchol = inv(chol)
    # β = (X'X)⁻¹ * (X'y)
    mul!(β, X', y)
    # Solve for regression coefficients β in-place using the Cholesky factor
    ldiv!(chol, β)
    # Compute predicted values in-place (ŷ = X * β)
    mul!(ŷ, X, β)
    # compute residuals
    @. ε = y - ŷ
    # sum of squared errors
    sse = ε ⋅ ε
    # residual variance
    σ² = sse / ν
    # compute dispersion matrix
    Σ = copy(invchol)
    rmul!(Σ, σ²)
    # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
    if isa(yt, FunctionTerm)
      # Apply the function-specific prediction logic
      applyBiasCorrection!(ŷ, yt, σ²)
      @. ε = yᵣ - ŷ
      sse = ε ⋅ ε
    end

    # check for nonnegative fitted values if required
    if nonnegative && any(v -> v < 0, ŷ)
      return throw(ErrorException("fitted values contain negative predictions"))
    end

    # Compute the coefficient of determination (R²), a measure of model fit
    r² = 1 - sse / sst
    # Compute the adjusted R², penalized for the number of predictors
    adjr² = 1 - (1 - r²) * (n - 1) / ν
    # Explained Variance (EV)
    ev = 1.0 - (var(ε) / var(yᵣ))
    # Calculate the Mean Absolute Error (MAE) as the average absolute residual value
    mae = mean(abs, ε)
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean(abs.(ε) ./ abs.(yᵣ)) * 100
    # Calculate the variance of the residuals (Mean Squared Error - Unbiased)
    mse = sse / ν
    # Standard Error of the Estimate (RMSE - Absolute)
    rmse = √(mse)
    # Standard Error of the Estimate % (Syx% - Relative)
    cv = (rmse / ȳ) * 100
    # calculate relative standard error of estimate (Syx%)
    syx = (rmse / sqrt(n)) / ȳ * 100.0
    # calculate uncertainty percentage at 95% confidence using T-distribution
    tValue = quantile(TDist(ν), 0.975)
    uncertainty = tValue * syx
    # store fitted model
    return AllometricModel(
      formula, cols, β, Σ, σ², n, ν, p, sse, sst, r², adjr², ev, mae, mape, mse, rmse, cv, syx, uncertainty)
  catch err
    throw(ErrorException("model fit failed: $(err)"))
  end
end

function prepareModelData(mf::ModelFrame)
  yTerm = mf.f.lhs
  yTerm isa CategoricalTerm &&
    throw(ArgumentError("dependent variable must be continuous"))

  cols = mf.data
  nObs = length(cols[1])

  # Extract terms
  contTerms = filter(t -> t isa ContinuousTerm, mf.f.rhs.terms)
  isempty(contTerms) && throw(ArgumentError("no continuous independent variables provided"))

  catTerms = filter(t -> t isa CategoricalTerm, mf.f.rhs.terms)

  # Buffers
  validXTerms = AbstractTerm[]
  termIndices = Vector{Vector{Int}}()

  xCols = Vector{Float64}[]

  for varTerm in contTerms
    candidates = xTerms(varTerm)
    # Store start index for this group
    startIdx = length(validXTerms) + 1

    for t in candidates
      try
        colData = modelcols(t, cols)
        if allfinite(colData)
          push!(xCols, colData)
          push!(validXTerms, t)
        end
      catch
      end
    end

    # Calculate indices based on what was added
    endIdx = length(validXTerms)
    if endIdx >= startIdx
      push!(termIndices, collect(startIdx:endIdx))
    end
  end

  if isempty(xCols)
    throw(ArgumentError("no valid continuous transformations found"))
  end

  # Optimized matrix construction
  nCols = length(xCols)
  xPool = Matrix{Float64}(undef, nObs, nCols)
  @inbounds for j in 1:nCols
    copyto!(view(xPool, :, j), xCols[j])
  end

  # Build categorical matrix
  qMatrix = if isempty(catTerms)
    Matrix{Float64}(undef, nObs, 0)
  else
    modelmatrix(sum(catTerms), cols)
  end

  # Prepare dependent variable variants
  yVariants = Vector{Tuple{AbstractTerm,Vector{Float64}}}()

  for t in yTerms(yTerm)
    try
      yVec = modelcols(t, cols)
      if allfinite(yVec)
        push!(yVariants, (t, yVec))
      end
    catch
    end
  end

  isempty(yVariants) && throw(ArgumentError("no valid dependent variable transformations found"))

  return (xPool, validXTerms, collect(catTerms), qMatrix, yVariants, cols, termIndices)
end

xTerms(t::AbstractTerm) = AbstractTerm[
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

yTerms(t::AbstractTerm) = AbstractTerm[
  t,
  FunctionTerm(log, [t], :(log($(t)))),
  FunctionTerm(log1p, [t], :(log1p($(t)))),
  FunctionTerm(sqrt, [t], :(√$(t))),
  FunctionTerm(cbrt, [t], :(∛$(t)))
]

# Fast finite check
function allfinite(v)
  @inbounds for i in eachindex(v)
    isfinite(v[i]) || return false
  end
  return true
end

"""
    applyBiasCorrection!(ŷ::Vector{<:Real}, ft::FunctionTerm, σ²::Real)

In-place version. Overwrites the input vector `ŷ` with the bias-corrected predictions on the original scale.
Internal helper for `predict`.
"""
function applyBiasCorrection!(ŷ::Vector{<:Real}, ft::FunctionTerm, σ²::Real)
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
    end
  end
end

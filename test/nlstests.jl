@testset "fitnls" begin
  # Curtis on the heights with the constant variance: the least squares of LsqFit.jl
  ols = fitnls(curtis, heights, :h, :d, [3.5, -10.0]; variance=:constant)
  x, y = ustrip.(heights.d), ustrip.(heights.h)
  reference = curve_fit((x, θ) -> curtis.(x, Ref(θ)), x, y, [3.5, -10.0])
  @test coef(ols) ≈ reference.param rtol = 1e-6
  @test ols.k == 0
  @test ols.σ ≈ sqrt(mean(abs2, reference.resid))
  @test nobs(ols) == nrow(heights)
  @test_throws ArgumentError fitnls(curtis, heights, :h, :d, [3.5, -10.0]; variance=:exponential)

  # with the variance function, the likelihood can only grow, and the least squares can only be
  # smaller with the constant variance
  model = fitnls(curtis, heights, :h, :d, [3.5, -10.0])
  @test model.loglik >= ols.loglik
  sse(θ) = sum(abs2, y .- curtis.(x, Ref(θ)))
  @test sse(coef(ols)) <= sse(coef(model))

  # Schumacher–Hall of two predictors: the standard deviation grows with the volume, k close to 1
  # (a multiplicative error), and the exponents agree with the GLM of the Gamma family
  sh = fitnls(schumacherhall, volume, :v, (:d, :h), [1e-4, 2.0, 1.0])
  @test 0.5 < sh.k < 1.5
  shglm = fitglm(@formula(v ~ log(d) + log(h)), volume)
  @test coef(sh)[2:3] ≈ coef(shglm)[2:3] rtol = 0.02

  # the biomass of a power of the diameter
  b = fitnls(power, biomass, :b, :d, [0.1, 2.4])
  @test unit(eltype(predict(b, biomass))) == u"kg"

  # the predictions in the unit of the response, and the residuals
  ŷ = predict(model, heights)
  @test unit(eltype(ŷ)) == u"m"
  @test ŷ ≈ curtis.(x, Ref(coef(model))) .* u"m"
  @test residuals(model) ≈ heights.h .- ŷ

  # the intervals: the one of a new tree is wider than the one of the mean
  conf = predict(model, heights[1:5, :]; interval=:confidence)
  pred = predict(model, heights[1:5, :]; interval=:prediction)
  @test conf.prediction ≈ ŷ[1:5]
  @test all(pred.lower .< conf.lower .< conf.prediction .< conf.upper .< pred.upper)
  @test_throws ArgumentError predict(model, heights; interval=:credible)

  # about 95% of the trees inside the prediction interval
  for (m, data, column) in ((model, heights, :h), (sh, volume, :v))
    p = predict(m, data; interval=:prediction)
    @test 0.9 < mean(p.lower .<= data[!, column] .<= p.upper) <= 1
  end

  # shown with the name of the equation and its parameters
  @test repr(sh) == "v = schumacherhall(d, h; θ = $(round.(coef(sh); sigdigits=4)))"
  @test startswith(repr(fitnls((d, θ) -> curtis(d, θ), heights, :h, :d, [3.5, -10.0])), "h = f(d; θ = ")
  @test occursin("Nonlinear regression of v (m^3) on d, h (n = 91)", repr(MIME("text/plain"), sh))

  # the sandwich covariance agrees in size with the one of the model
  @test size(robustvcov(model)) == (2, 2)
  @test all(0.3 .< sqrt.(diag(robustvcov(sh))) ./ stderror(sh) .< 3)
end

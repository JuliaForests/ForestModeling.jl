@testset "fitnls" begin
  # the height curve: the asymptote 25 (the plots move it), b = 0.07, c = 1.1
  model = fitnls(cr, data, :h, :d, [20.0, 0.05, 1.0])
  @test model isa NLSFit
  @test coef(model) ≈ [25.0, 0.07, 1.1] rtol = 0.08
  @test nobs(model) == nrow(data)
  @test length(stderror(model)) == 3

  # the volume: Schumacher–Hall with a multiplicative error, so the standard deviation grows with
  # the mean, k ≈ 1, and σ ≈ 0.08
  volume = fitnls(schumacherhall, data, :v, (:d, :h), [1e-4, 2.0, 1.0])
  @test coef(volume)[2:3] ≈ [1.9, 1.0] atol = 0.06
  @test volume.k ≈ 1.0 atol = 0.1
  @test volume.σ ≈ 0.08 atol = 0.01

  # the constant variance is the ordinary least squares
  ols = fitnls(cr, data, :h, :d, [20.0, 0.05, 1.0]; variance=:constant)
  @test ols.k == 0
  x, y = ustrip.(data.d), ustrip.(data.h)
  sse(θ) = sum(abs2, y .- cr.(x, Ref(θ)))
  @test sse(coef(ols)) <= sse(coef(model))
  @test ols.σ ≈ sqrt(sse(coef(ols)) / length(y))
  @test_throws ArgumentError fitnls(cr, data, :h, :d, [20.0, 0.05, 1.0]; variance=:exponential)

  # the predictions in the unit of the response, and the residuals
  ŷ = predict(model, data)
  @test unit(eltype(ŷ)) == u"m"
  @test ŷ ≈ [cr(ustrip(d), coef(model)) for d in data.d] .* u"m"
  @test residuals(model) ≈ data.h .- ŷ

  # the intervals: the one of a new tree is wider than the one of the mean
  conf = predict(model, data[1:5, :]; interval=:confidence)
  pred = predict(model, data[1:5, :]; interval=:prediction)
  @test conf.prediction ≈ ŷ[1:5]
  @test all(pred.lower .< conf.lower .< conf.prediction .< conf.upper .< pred.upper)
  @test_throws ArgumentError predict(model, data; interval=:credible)

  # about 95% of the trees inside the prediction interval
  all95 = predict(model, data; interval=:prediction)
  @test 0.9 < mean(all95.lower .<= data.h .<= all95.upper) < 0.99

  # the sandwich covariance agrees in size with the one of the model
  @test size(robustvcov(model)) == (3, 3)
  @test all(0.3 .< sqrt.(diag(robustvcov(volume))) ./ stderror(volume) .< 3)
end

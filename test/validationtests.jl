@testset "metrics" begin
  m = metrics([10.0, 12.0, 15.0], [11.0, 12.0, 13.5])
  @test m.rmse ≈ sqrt((1 + 0 + 2.25) / 3)
  @test m.bias ≈ (-1 + 0 + 1.5) / 3
  @test m.mape ≈ 100 * (1 / 10 + 0 + 1.5 / 15) / 3

  # in the unit of y, whatever the unit of the predictions
  q = metrics([10.0, 12.0]u"m", [1100.0, 1200.0]u"cm")
  @test q.rmse ≈ sqrt(0.5)u"m"
  @test q.bias ≈ -0.5u"m"
  @test q.mape ≈ 5.0
  @test_throws DimensionMismatch metrics([1.0, 2.0], [1.0])
end

@testset "blockcv" begin
  fit(t) = fitglm(@formula(v ~ log(d) + log(h)), t)
  cv = blockcv(fit, data, :v, :plot; rng=MersenneTwister(1))
  @test keys(cv) == (:rmse, :bias, :mape)
  @test unit(cv.rmse) == u"m^3"
  # the error out of the plots of the fit is at least that of the fit
  @test cv.rmse >= metrics(data.v, predict(fit(data), data)).rmse
  @test blockcv(fit, data, :v, :plot; rng=MersenneTwister(1)) == cv
  @test_throws ArgumentError blockcv(fit, data[data.plot .∈ Ref(["1", "2"]), :], :v, :plot)

  # a mixed model predicts the plots left out by the population curve: a larger error than the
  # one of the plots of the fit, with their random effects
  mixed(t) = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), t)
  cvmixed = blockcv(mixed, data, :h, :plot; rng=MersenneTwister(1))
  @test cvmixed.rmse > metrics(data.h, predict(mixed(data), data)).rmse
end

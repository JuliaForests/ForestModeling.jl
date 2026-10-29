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
  # with four plots and k = 4 each plot is predicted by the model of the other three
  curtisfit(t) = fitnls(curtis, t, :h, :d, [3.5, -10.0])
  cv = blockcv(curtisfit, heights, :h, :plot; k=4)
  @test keys(cv) == (:rmse, :bias, :mape)
  @test unit(cv.rmse) == u"m"
  # leaving out one plot after the other the folds do not depend on the order
  @test blockcv(curtisfit, heights, :h, :plot; k=4, rng=MersenneTwister(1)) == cv
  others = [heights[heights.plot .!= p, :] for p in ["A", "B", "C", "D"]]
  outof = vcat([predict(curtisfit(o), heights[heights.plot .== p, :]) for (o, p) in zip(others, ["A", "B", "C", "D"])]...)
  @test cv.rmse ≈ metrics(vcat([heights.h[heights.plot .== p] for p in ["A", "B", "C", "D"]]...), outof).rmse
  @test_throws ArgumentError blockcv(curtisfit, heights, :h, :plot; k=5)

  # a mixed model predicts the plots left out by the population curve: a larger error than the
  # one of the plots of the fit, with their random effects
  mixed(t) = fitmixed(@formula(h ~ 1 + log(d) + (1 | plot)), t)
  cvmixed = blockcv(mixed, heights, :h, :plot; k=4)
  @test cvmixed.rmse > metrics(heights.h, predict(mixed(heights), heights)).rmse
end

@testset "graybilltest" begin
  sh = fitglm(@formula(v ~ log(d) + log(h)), volume)
  y, ŷ = volume.v, predict(sh, volume)
  g = graybilltest(y, ŷ)
  @test keys(g) == (:β0, :β1, :F, :pvalue)
  @test unit(g.β0) == u"m^3"
  # the F of the hypothesis β = (0, 1): the increase of the residual sum of squares from the
  # regression to the identity line, over the 2 parameters and the residual variance
  Y, Ŷ = ustrip.(y), ustrip.(ŷ)
  ols = lm([ones(length(Y)) Y], Ŷ)
  s² = deviance(ols) / dof_residual(ols)
  @test g.F ≈ (sum(abs2, Ŷ .- Y) - deviance(ols)) / 2 / s²
  @test coef(ols) ≈ [ustrip(g.β0), g.β1]
  @test g.pvalue > 0.05
  # predictions 10% too high are rejected
  @test graybilltest(y, 1.1 .* ŷ).pvalue < 0.001
  # the predictions in another unit
  @test graybilltest(y, uconvert.(u"dm^3", ŷ)).F ≈ g.F
end

@testset "identitytest" begin
  curtisfit(t) = fitnls(curtis, t, :h, :d, [3.5, -10.0])
  test = identitytest(curtisfit, heights, :plot)
  @test keys(test) == (:lr, :df, :pvalue)
  complete = fitbygroup(curtisfit, heights, :plot).model
  @test test.lr ≈ 2 * (sum(m -> m.loglik, complete) - curtisfit(heights).loglik)
  @test test.df == 4 * 4 - 4          # θ (2), σ and k of each curve
  @test 0 <= test.pvalue <= 1
  # two copies of the same trees share one line: the likelihood ratio is zero
  linear(t) = fitglm(@formula(h ~ log(d)), t; family=Normal(), link=IdentityLink())
  copies = vcat(transform(heights, :plot => (p -> fill("one", length(p))) => :copy),
                transform(heights, :plot => (p -> fill("two", length(p))) => :copy))
  same = identitytest(linear, copies, :copy)
  @test same.df == 3
  @test same.lr ≈ 0 atol = 1e-8
  @test same.pvalue ≈ 1
end

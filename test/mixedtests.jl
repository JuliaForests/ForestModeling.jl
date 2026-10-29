@testset "fitmixed" begin
  formula = @formula(h ~ 1 + log(d) + (1 | plot))
  model = fitmixed(formula, data)
  @test model isa UnitfulModel

  # the same fit as MixedModels.jl on the values in the units of the columns
  plain = fit(MixedModel, formula, DataFrame(plot=data.plot, d=ustrip.(data.d), h=ustrip.(data.h)); progress=false)
  @test coef(model) ≈ coef(plain)
  @test residuals(model) ≈ (response(plain) .- fitted(plain)) .* u"m"

  # the plots of the fit with their random effects, a new plot with the population curve
  ŷ = predict(model, data)
  @test ustrip.(u"m", ŷ) ≈ fitted(plain)
  new = DataFrame(plot=["new"], d=[20.0u"cm"])
  @test ustrip(u"m", only(predict(model, new))) ≈ coef(plain)[1] + coef(plain)[2] * log(20)
  @test_throws ArgumentError predict(model, data; interval=:confidence)

  @testset "calibrate" begin
    # all the trees of a plot of the fit measured: the random effect of the fit itself
    plot1 = data[data.plot .== "1", :]
    @test calibrate(model, plot1, plot1) ≈ predict(model, plot1)

    # two trees measured in a new plot: between the population curve and those trees
    newplot = transform(plot1, :plot => (p -> fill("new", length(p))) => :plot)
    population = predict(model, newplot)
    calibrated = calibrate(model, newplot[1:2, :], newplot)
    shift = calibrated .- population
    @test all(shift .≈ shift[1])          # a random intercept shifts the whole curve
    r = mean(newplot.h[1:2] .- population[1:2])
    @test 0 < shift[1] / r < 1          # pulled toward the population curve
    # the shift has the closed form σ²_u / (σ²_u + σ²/n) of the mean residual
    σu², σ² = only(VarCorr(plain).σρ.plot.σ)^2, plain.σ^2
    @test shift[1] ≈ σu² / (σu² + σ² / 2) * r

    # a plot without measured trees keeps the population curve
    @test calibrate(model, newplot[1:0, :], newplot) ≈ population

    # with random slopes too, the closed form for a vector of random effects
    slopes = fitmixed(@formula(h ~ 1 + log(d) + (1 + log(d) | plot)), data)
    @test calibrate(slopes, plot1, plot1) ≈ predict(slopes, plot1)
  end
end

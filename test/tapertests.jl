# ==============================================================================
# TAPER FIXTURES
# ==============================================================================
# 6 synthetic trees × 4 stem sections each (near-base, lower, mid, upper),
# generated from a *known* Kozak1969-shaped relative-diameter curve plus a
# small deterministic (no RNG) perturbation — enough to keep every model's
# fit away from the degenerate "residuals are exactly zero" edge case (which
# trips up `isNormality`'s Shapiro-Wilk test on a constant sample), while
# still being reproducible across platforms/Julia versions.

const TAPER_DBH = repeat([20.0, 25.0, 30.0, 35.0, 40.0, 45.0], inner=4)
const TAPER_HT = repeat([18.0, 20.0, 22.0, 24.0, 26.0, 28.0], inner=4)
const TAPER_HR = repeat([0.02, 0.3, 0.6, 0.9], outer=6)
const TAPER_HI = TAPER_HR .* TAPER_HT
_taperDrTrue(hr) = 1.05 - 0.9hr - 0.15hr^2
const TAPER_DI = TAPER_DBH .* (_taperDrTrue.(TAPER_HR) .+ [0.01sin(3.7i) for i in 1:24])

const TAPER_MODELS = (Kozak1969(), Schoepfer1966(), Matte1949(), Demaerschalk1972(), Clutter1980(),
  MaxBurkhart1976(), Johnson1911(), Kozak1988(), Kozak2004(), Bi2000())

@testset "taper/models.jl + taper/fitting.jl" begin
  @testset "ncoef / islinear catalog" begin
    @test ncoef(Kozak1969()) == 3 && islinear(Kozak1969())
    @test ncoef(Schoepfer1966()) == 6 && islinear(Schoepfer1966())
    @test ncoef(Matte1949()) == 3 && islinear(Matte1949())
    @test ncoef(Demaerschalk1972()) == 2 && !islinear(Demaerschalk1972())
    @test ncoef(Clutter1980()) == 2 && !islinear(Clutter1980())
    @test ncoef(MaxBurkhart1976()) == 4 && !islinear(MaxBurkhart1976())
    @test ncoef(Johnson1911()) == 3 && !islinear(Johnson1911())
    @test ncoef(Kozak1988()) == 4 && !islinear(Kozak1988())
    @test ncoef(Kozak2004()) == 9 && !islinear(Kozak2004())
    @test ncoef(Bi2000()) == 7 && !islinear(Bi2000())

    @test MaxBurkhart1976().α1 == 0.11 && MaxBurkhart1976().α2 == 0.73
    @test MaxBurkhart1976(α1=0.2, α2=0.8).α1 == 0.2
    @test Kozak1988().p == 0.25
    @test Kozak2004(p=0.3).p == 0.3
  end

  @testset "fit: every model converges with a sane fit" begin
    for model in TAPER_MODELS
      ft = fit(model, TAPER_DBH, TAPER_HT, TAPER_HI, TAPER_DI)
      @test ft isa TaperFit
      @test ft isa ForestModeling.RegressionModel
      @test length(coef(ft)) == ncoef(model)
      @test nobs(ft) == 24
      @test dof_residual(ft) == 24 - ncoef(model)
      @test r2(ft) > 0.95
      @test dispersion(ft) < 0.05
      @test length(fitted(ft)) == 24
      @test residuals(ft) ≈ (TAPER_DI ./ TAPER_DBH) .- fitted(ft) atol = 1e-10
    end
  end

  @testset "fit: rejects mismatched lengths / too few observations" begin
    @test_throws DimensionMismatch fit(Kozak1969(), TAPER_DBH, TAPER_HT, TAPER_HI, TAPER_DI[1:end-1])
    @test_throws ArgumentError fit(Kozak2004(), TAPER_DBH[1:5], TAPER_HT[1:5], TAPER_HI[1:5], TAPER_DI[1:5])
  end

  @testset "predict: absolute diameter, scalar and vector" begin
    ft = fit(Kozak1969(), TAPER_DBH, TAPER_HT, TAPER_HI, TAPER_DI)
    d1 = predict(ft, 30.0, 22.0, 5.0)
    @test d1 isa Float64
    @test 0 < d1 < 35.0

    dvec = predict(ft, 30.0, 22.0, [1.0, 5.0, 10.0])
    @test dvec isa Vector{Float64}
    @test length(dvec) == 3
    @test dvec[1] ≈ predict(ft, 30.0, 22.0, 1.0)
  end

  @testset "criteriaTable / criteriaSelection accept a Vector{TaperFit} unchanged" begin
    fits = [fit(m, TAPER_DBH, TAPER_HT, TAPER_HI, TAPER_DI) for m in TAPER_MODELS]
    table = criteriaTable(fits, :adjr2, :rmse)
    @test nrow(table) == min(10, length(fits))
    @test issorted(table.rank)

    best = criteriaSelection(fits, :adjr2, :rmse)
    @test best isa TaperFit
    @test best === fits[argmin(ForestModeling.calculateScore(fits, [:adjr2, :rmse]))]
  end
end

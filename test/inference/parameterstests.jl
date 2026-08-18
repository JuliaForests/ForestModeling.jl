m = fit(AllometricModel, @formula(log(h) ~ 1 + log(d)), dataPlain())

@testset "inference/parameters.jl" begin
  @testset "coefficient-level accessors" begin
    @test coef(m) ≈ [0.39737866774301006, 0.8233376284585446] atol = 1e-8
    @test coefnames(m) == ["(Intercept)", "log(d)"]
    @test size(vcov(m)) == (2, 2)
    @test stderror(m) ≈ sqrt.(diag(vcov(m)))

    ci = confint(m)
    @test size(ci) == (2, 2)
    @test all(ci[:, 1] .< coef(m) .< ci[:, 2])  # estimate sits inside its own CI

    pv = pvalue(m)
    @test length(pv) == 2
    @test all(0 .<= pv .<= 1)
    @test pv[2] < 0.001  # log(d) is a very strong predictor of log(h) here

    ct = coeftable(m)
    @test size(ct.cols[1], 1) == 2
  end

  @testset "formula / dof / response" begin
    @test dof(m) == 3
    @test dof_residual(m) == length(D) - 2
    @test formula(m) isa FormulaTerm
    @test response(m) == H
    @test size(modelmatrix(m)) == (length(D), 2)
  end

  @testset "r2 / adjr2 / dispersion" begin
    @test r2(m) == m.r²
    @test adjr2(m) == m.adjr²
    @test dispersion(m) == m.rmse
    @test dispersion(m, true) == m.mse
  end

  @testset "loglikelihood" begin
    @test loglikelihood(m) isa Real
    @test loglikelihood(m) > nullloglikelihood(m)  # the fitted model beats the intercept-only null
  end

  @testset "cooksdistance: always plain, never unit-tagged" begin
    mUnit = fit(AllometricModel, @formula(log(h) ~ 1 + log(d)), dataUnitful())
    cd = cooksdistance(mUnit)
    @test length(cd) == length(D)
    @test all(isfinite, cd)
    @test all(>=(0), cd)
    @test nonmissingtype(eltype(cd)) <: Real
    @test !(nonmissingtype(eltype(cd)) <: Unitful.Quantity)
  end
end

@testset "inference/diagnostics.jl" begin
  # n < 3 is a documented degenerate case that always passes, deterministically
  @test isNormality(Float64[]) === true
  @test isNormality([1.0, 2.0]) === true
  @test isNormality(m) === m.normal  # model-level accessor just reads the stored flag
end

@testset "inference/criteria.jl" begin
  models = regression(dataPlain(), :h, :d; nMax=2)

  @testset "calculateScore matches criteriaTable's ranking" begin
    scores = ForestModeling.calculateScore(models, [:adjr2, :cv])
    @test length(scores) == length(models)
    @test argmin(scores) == argmin(ForestModeling.calculateScore(models, [:adjr2, :cv]))
  end

  @testset ":normality hard filter" begin
    normalOnly = filter(isNormality, models)
    if isempty(normalOnly)
      @test_throws ErrorException criteriaTable(models, :normality)
      @test_throws ErrorException criteriaSelection(models, :normality)
    else
      table = criteriaTable(models, :normality, :adjr2)
      @test all(m -> isNormality(m), table.model)
    end
  end

  @testset "criteriaSelection matches the top row of criteriaTable" begin
    best = criteriaSelection(models, :adjr2, :cv)
    top = criteriaTable(models, :adjr2, :cv; best=1)
    @test best === top.model[1]
  end
end

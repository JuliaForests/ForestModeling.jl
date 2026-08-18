@testset "fitting/mixed.jl" begin
  @testset "fitMixed: basic, plain numbers" begin
    m = fitMixed(@formula(log(h) ~ 1 + log(d)), groupedDataPlain(), :species)
    @test m isa MixedAllometricModel
    @test length(coef(m)) == 2
    @test nobs(m) == length(GROUP_D)
    @test m.adjr² > 0.9

    p = predict(m)
    @test nonmissingtype(eltype(p)) <: Real
    @test fitted(m) == p
    @test length(residuals(m)) == length(GROUP_D)
  end

  @testset "fitMixed: single Symbol vs Vector{Symbol} group spec agree" begin
    m1 = fitMixed(@formula(log(h) ~ 1 + log(d)), groupedDataPlain(), :species)
    m2 = fitMixed(@formula(log(h) ~ 1 + log(d)), groupedDataPlain(), [:species])
    @test coef(m1) ≈ coef(m2)
  end

  @testset "fitMixed: Unitful in, Unitful out" begin
    m = fitMixed(@formula(log(h) ~ 1 + log(d)), groupedDataUnitful(), :species)
    p = predict(m)
    @test nonmissingtype(eltype(p)) <: Unitful.Quantity
    @test unit(p[1]) == u"m"
    @test nonmissingtype(eltype(residuals(m))) <: Unitful.Quantity
  end

  @testset "predict: unseen group falls back to population-level, not an error" begin
    m = fitMixed(@formula(log(h) ~ 1 + log(d)), groupedDataPlain(), :species)
    newRow = DataFrame(d=[20.0], h=[0.0], species=["Willow"])
    p = predict(m, newRow)
    @test all(isfinite, skipmissing(p))
  end

  @testset "regressionMixed: combinatorial search reuses criteriaTable/criteriaSelection" begin
    models = regressionMixed(groupedDataPlain(), :h, :d, :species)
    @test !isempty(models)
    @test all(m -> m isa MixedAllometricModel, models)

    best = criteriaSelection(models, :adjr2, :cv)
    @test best isa MixedAllometricModel
    @test best.adjr² > 0.9

    table = criteriaTable(models, :adjr2, :cv)
    @test nrow(table) == min(10, length(models))

    @test_throws ArgumentError regressionMixed(groupedDataPlain(), :h, :d, :species; nMax=6)

    # same fit-time-vs-predict-time consistency guard as fitting/ols.jl — every candidate's
    # stored sse must match what predict() reproduces afterward
    for m in models
      ε = response(m) .- predict(m)
      @test sum(abs2, ε) ≈ m.sse atol = 1e-6
    end
  end
end

@testset "formula/transforms.jl" begin
  @testset "named forestry linearizations" begin
    @test petterson(4.0) ≈ 0.5
    @test naslund(4.0, 10.0) ≈ 5.0
    @test prodan(4.0, 10.0) ≈ 25.0
  end

  @testset "expandXTerms / transformYTerms shapes" begin
    mf = ModelFrame(@formula(h ~ 1 + d), dataPlain())
    xTerm = only(filter(t -> t isa ContinuousTerm, mf.f.rhs.terms))
    yTerm = mf.f.lhs

    xCandidates = ForestModeling.expandXTerms(xTerm)
    @test length(xCandidates) == 6
    @test all(t -> t isa AbstractTerm, xCandidates)

    yCandidates = ForestModeling.transformYTerms(yTerm, xCandidates)
    # identity, log, inv, sqrt, petterson, plus naslund+prodan for every x candidate
    @test length(yCandidates) == 5 + 2 * length(xCandidates)
  end
end

@testset "formula/classicequations.jl" begin
  @testset "classicEquations catalog" begin
    @test issubset((:curtis, :henriksen, :stoffels, :trorey, :prodan1, :prodan2, :petterson, :naslund, :linear, :cubic),
      keys(classicEquations))
  end

  @testset "fitClassic" begin
    m = fitClassic(:curtis, :h, :d, dataPlain())
    @test m isa AllometricModel
    @test m.adjr² > 0.9

    @test_throws ArgumentError fitClassic(:not_a_real_equation, :h, :d, dataPlain())

    # every catalog entry must actually fit and predict without erroring — the
    # response transform must be named (see the module's own note) or
    # predictBiasCorrected! rejects it even though `fit` itself succeeds
    for name in keys(classicEquations)
      mc = fitClassic(name, :h, :d, dataPlain())
      @test mc isa AllometricModel
      @test all(isfinite, predict(mc))
    end

    # :prodan2 predicts height *above breast height* (h - 1.3), not total
    # height — the offset baked into its own response transform
    mProdan2 = fitClassic(:prodan2, :h, :d, dataPlain())
    @test all(isapprox.(predict(mProdan2) .+ 1.3, H, atol=1.5))
  end
end

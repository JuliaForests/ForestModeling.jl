@testset "fitting/grouped.jl" begin
  @testset "regressionGrouped: basic shape" begin
    gm = regressionGrouped(groupedDataPlain(), :h, :d, :species)
    @test gm isa GroupedModel
    @test gm.general isa AllometricModel
    @test gm.qualy isa AllometricModel
    @test length(gm.grouped) == 3
    @test Set(keys(gm.grouped)) == Set(GROUP_SPECIES)
    @test gm.groupNames == [:species]
  end

  @testset "regressionGrouped: single Symbol vs Vector{Symbol} agree" begin
    gm1 = regressionGrouped(groupedDataPlain(), :h, :d, :species)
    gm2 = regressionGrouped(groupedDataPlain(), :h, :d, [:species])
    @test gm1.groupNames == gm2.groupNames
    @test coef(gm1.general) ≈ coef(gm2.general)
  end

  @testset "criteriaTable(::GroupedModel): pooled comparison" begin
    gm = regressionGrouped(groupedDataPlain(), :h, :d, :species)
    table = criteriaTable(gm, :adjr2, :cv, :ev)
    @test nrow(table) == 3
    @test Set(table.type) == Set(["General", "Qualy", "Grouped"])

    # stratifying by species is, by construction of the fixture, a real win over pooling
    generalRow = only(filter(:type => ==("General"), table))
    groupedRow = only(filter(:type => ==("Grouped"), table))
    @test groupedRow.adjr2 > generalRow.adjr2

    # per-group breakdown reuses the existing Vector{<:RegressionModel} method unchanged
    perGroup = criteriaTable(collect(values(gm.grouped)), :adjr2)
    @test nrow(perGroup) == 3

    @test_throws ArgumentError criteriaTable(gm, :not_a_real_criterion)
  end

  @testset "criteriaTable(::GroupedModel): default criteria" begin
    gm = regressionGrouped(groupedDataPlain(), :h, :d, :species)
    table = criteriaTable(gm)
    @test Set(propertynames(table)) ⊇ Set([:type, :model, :adjr2, :cv, :ev])
  end

  @testset "regressionGrouped: Unitful in, Unitful out" begin
    gm = regressionGrouped(groupedDataUnitful(), :h, :d, :species)
    p = predict(gm.general)
    @test nonmissingtype(eltype(p)) <: Unitful.Quantity
    for m in values(gm.grouped)
      @test nonmissingtype(eltype(predict(m))) <: Unitful.Quantity
    end
  end

  @testset "GroupedModel: xName field" begin
    gm = regressionGrouped(groupedDataPlain(), :h, :d, :species)
    @test gm.xName == :d
  end

  @testset "predictBounded: hierarchical range fallback" begin
    # Cedar deliberately given a much narrower d-range (10-17) than the global
    # range (10-37, from Oak/Pine) — the only way to actually exercise every
    # branch of the fallback (global-out-of-range, group-out-of-range,
    # unknown-group, missing, in-range).
    oakD = collect(10.0:3.0:37.0)
    pineD = collect(10.0:3.0:37.0)
    cedarD = [10.0, 12.0, 14.0, 16.0, 11.0, 13.0, 15.0, 17.0, 10.5, 12.5]
    d = vcat(oakD, pineD, cedarD)
    species = vcat(fill("Oak", 10), fill("Pine", 10), fill("Cedar", 10))
    h = [0.3 + 0.7 * log(d[i]) for i in eachindex(d)]
    data = DataFrame(d=d, h=h, species=species)
    gm = regressionGrouped(data, :h, :d, :species)

    @test extrema(gm.grouped["Cedar"].data[:d]) == (10.0, 17.0)
    @test extrema(gm.general.data[:d]) == (10.0, 37.0)

    @testset "in range: uses the group's own model" begin
      row = DataFrame(d=[12.0], species=["Cedar"])
      @test predictBounded(gm, row)[1] ≈ predict(gm.grouped["Cedar"], row)[1] atol = 1e-10
    end

    @testset "in global range, out of group range: falls back to qualy" begin
      row = DataFrame(d=[30.0], species=["Cedar"])
      @test predictBounded(gm, row)[1] ≈ predict(gm.qualy, row)[1] atol = 1e-10
    end

    @testset "out of global range: falls back to general" begin
      row = DataFrame(d=[1000.0], species=["Cedar"])
      @test predictBounded(gm, row)[1] ≈ predict(gm.general, row)[1] atol = 1e-10
    end

    @testset "unknown group (no model at all): falls back to general" begin
      row = DataFrame(d=[20.0], species=["Birch"])
      @test predictBounded(gm, row)[1] ≈ predict(gm.general, row)[1] atol = 1e-10
    end

    @testset "missing predictor: missing, no fallback attempted" begin
      row = DataFrame(d=[missing], species=["Cedar"])
      @test ismissing(predictBounded(gm, row)[1])
    end

    @testset "multi-row: each row resolves independently" begin
      rows = DataFrame(d=[12.0, 30.0, 1000.0], species=["Cedar", "Cedar", "Cedar"])
      preds = predictBounded(gm, rows)
      @test length(preds) == 3
      @test preds[1] ≈ predict(gm.grouped["Cedar"], rows[1:1, :])[1] atol = 1e-10
      @test preds[2] ≈ predict(gm.qualy, rows[2:2, :])[1] atol = 1e-10
      @test preds[3] ≈ predict(gm.general, rows[3:3, :])[1] atol = 1e-10
    end
  end
end

@testset "units.jl" begin
  @testset "stripcolumnunits" begin
    data = DataFrame(d=D .* u"cm", h=H .* u"m", note=fill("x", length(D)))
    cols, units = ForestModeling.stripcolumnunits(data)

    @test cols.d == D
    @test cols.h == H
    @test cols.note == fill("x", length(D))
    @test units.d == u"cm"
    @test units.h == u"m"
    @test units.note === nothing

    # plain numbers round-trip untouched, with a `nothing` unit recorded
    plainCols, plainUnits = ForestModeling.stripcolumnunits(dataPlain())
    @test plainCols.d == D
    @test plainUnits.d === nothing
  end

  @testset "matchunits" begin
    _, units = ForestModeling.stripcolumnunits(dataUnitful())

    # a Quantity column gets uconverted to the stored unit, then stripped
    mm = DataFrame(d=(D .* 10) .* u"mm")  # 10x in mm == same magnitude in cm
    matched = ForestModeling.matchunits(mm, units)
    @test matched.d ≈ D

    # a bare-number column is assumed already expressed in the fit unit
    bare = DataFrame(d=D)
    matchedBare = ForestModeling.matchunits(bare, units)
    @test matchedBare.d == D

    # a column absent from `units` (or recorded with no unit) passes through
    extra = DataFrame(d=D .* u"cm", note=["a" for _ in D])
    matchedExtra = ForestModeling.matchunits(extra, units)
    @test matchedExtra.note == extra.note
  end

  @testset "_restoreunit" begin
    @test ForestModeling._restoreunit([1.0, 2.0], nothing) == [1.0, 2.0]
    restored = ForestModeling._restoreunit([1.0, 2.0], u"m")
    @test restored == [1.0, 2.0]u"m"
  end
end

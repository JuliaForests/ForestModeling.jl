@testset "assortment" begin
  @testset "taper equations" begin
    h, D, H = 10.0, 25.0, 28.0
    z = h / H
    @test kozak1969(h, D, H, [1.06, -1.82, 0.749]) ≈ D * sqrt(1.06 - 1.82z + 0.749z^2)
    @test kozak1969(28.0, D, H, [1.0, -2.0, 0.5]) == 0          # negative near the top: zero
    @test schoepfer1966(h, D, H, [1.12, -3.07, 10.6, -19.7, 16.0, -4.94]) ≈
          D * (1.12 - 3.07z + 10.6z^2 - 19.7z^3 + 16.0z^4 - 4.94z^5)
    @test demaerschalk1972(h, D, H, [0.885, 0.903, 0.804, -0.665]) ≈ 0.885 * D^0.903 * (H - h)^0.804 * H^-0.665
    @test maxburkhart1976(h, D, H, [-3.7, 1.83, -1.81, 46.8]; a1=0.75, a2=0.1) ≈
          D * sqrt(-3.7(z - 1) + 1.83(z^2 - 1) - 1.81(0.75 - z)^2)
    # at the height of reference the diameter is D
    @test ormerod1973(1.3, D, H, [0.804]; href=1.3) ≈ D
    @test bi2000(1.3, D, H, [1.15, 0.0788, 0.0249, -0.619, 0.00107, 0.0181, -0.0604]; href=1.3) ≈ D
    θ = [0.971, 1.03, -0.0248, 0.503, -0.744, 0.645, 3.44, 0.0159, -0.149]
    @test kozak2004(1.3, D, H, θ; href=1.3) ≈ 0.971 * D^1.03 * H^-0.0248      # X = 1
    @test kozak1988(0.0, D, H, [1.0, 1.0, 1.0, 0, 0, 0, 0, 0]; p=0.25) ≈ D / (1 - sqrt(0.25))^0
  end

  # the stem profiles of 50 trees (995 diameters d at the heights h, with D and H of each tree) are
  # local data, not in the repository: without them the fits and the assortments are not tested
  path = joinpath(@__DIR__, "..", "data", "cubage.csv")
  if isfile(path)
    cubage = parseunits(CSV.read(path, DataFrame))
    k04(h, D, H, θ) = kozak2004(h, D, H, θ; href=1.3)
    taper = fitnls(k04, cubage, :d, (:h, :D, :H), [1.0, 0.97, 0.03, 0.49, -0.87, 0.50, 3.88, 0.03, -0.19])

    @testset "fitting" begin
      starts = [
        ((h, D, H, θ) -> kozak1969(h, D, H, θ), [1.2, -2.2, 1.1], :constant),
        ((h, D, H, θ) -> schoepfer1966(h, D, H, θ), [1.2, -3.0, 10.0, -20.0, 18.0, -6.0], :power),
        ((h, D, H, θ) -> ormerod1973(h, D, H, θ; href=1.3), [0.8], :power),
        ((h, D, H, θ) -> demaerschalk1972(h, D, H, θ), [1.0, 1.0, 0.8, -0.8], :power),
        ((h, D, H, θ) -> maxburkhart1976(h, D, H, θ; a1=0.75, a2=0.1), [-3.0, 1.5, -1.5, 50.0], :power),
        ((h, D, H, θ) -> kozak1988(h, D, H, θ; p=0.25), [1.0, 1.0, 1.0, 0.5, -0.1, 0.5, -0.5, 0.1], :power),
        ((h, D, H, θ) -> bi2000(h, D, H, θ; href=1.3), [1.8, -0.2, -0.04, -0.9, -0.0006, 0.07, -0.14], :power),
      ]
      for (f, θ0, variance) in starts
        model = fitnls(f, cubage, :d, (:h, :D, :H), θ0; variance)
        @test metrics(cubage.d, predict(model, cubage)).rmse < 3u"cm"
      end
      # the variable-exponent models describe the stems best
      @test metrics(cubage.d, predict(taper, cubage)).rmse < 2u"cm"
    end

    D, H = 25.0u"cm", 28.0u"m"

    @testset "taperdiameter and taperheight" begin
      h = [0.1, 1.3, 10.0, 20.0, 27.0]u"m"
      d = taperdiameter(taper, D, H, h)
      @test d ≈ predict(taper, DataFrame(h=h, D=fill(D, 5), H=fill(H, 5)))
      @test issorted(d; rev=true)
      @test taperdiameter(taper, 250.0u"mm", 2800.0u"cm", h) ≈ d          # the units of the fit
      # the height of a diameter is the inverse of the diameter at a height
      @test taperdiameter(taper, D, H, [taperheight(taper, D, H, 10.0u"cm")]) ≈ [10.0u"cm"]
      @test taperheight(taper, D, H, d[3]) ≈ 10.0u"m"
      @test_throws DomainError taperheight(taper, D, H, 60.0u"cm")
      # a model that is not a taper
      @test_throws ArgumentError taperheight(fitnls(curtis, heights, :h, :d, [3.5, -10.0]), D, H, 10.0u"cm")
    end

    @testset "taperedvolume" begin
      v = taperedvolume(taper, D, H)
      @test unit(v) == u"m^3"
      # the volumes of the sections add up to the one of the stem
      @test taperedvolume(taper, D, H, 0.0u"m", 12.0u"m") + taperedvolume(taper, D, H, 12.0u"m", H) ≈ v
      # the stem of a cylinder of the diameter at the base would be larger
      @test v < quartπ * taperdiameter(taper, D, H, [0.0u"m"])[1]^2 * H
      # close to the Smalian volumes of the measured trees
      for tree in 1:10
        t = cubage[cubage.tree .== tree, :]
        smalian = sum(quartπ .* (t.d[1:end-1] .^ 2 .+ t.d[2:end] .^ 2) ./ 2 .* diff(t.h))
        @test taperedvolume(taper, t.D[1], t.H[1], t.h[1], t.h[end]) ≈ smalian rtol = 0.15
      end
      @test_throws DomainError taperedvolume(taper, D, H, 12.0u"m", 2.0u"m")
      @test_throws DomainError taperedvolume(taper, D, H, 0.0u"m", 30.0u"m")
    end

    @testset "logassortment" begin
      products = DataFrame(name=["sawlog", "pulpwood"], sed=[18.0, 8.0]u"cm", minlength=[2.5, 2.0]u"m",
                           maxlength=[4.0, 6.0]u"m", kerf=[0.005, 0.005]u"m")
      D = 30.0u"cm"
      logs = logassortment(taper, D, H, products; stump=0.1u"m")
      @test names(logs) == ["name", "n", "v"]
      @test logs.name == products.name
      @test all(logs.n .> 0)
      # the logs take part of the stem, never more than all of it
      @test sum(logs.v) < taperedvolume(taper, D, H)
      # the sawlogs end where the stem narrows to 18 cm: the first sawlogs have the longest length,
      # the last one ends at most at that height
      top = taperheight(taper, D, H, 18.0u"cm")
      sawlogs = logs.n[1]
      @test 0.1u"m" + (sawlogs - 1) * (4.0u"m" + 0.005u"m") + 2.5u"m" <= top
      # a product thicker than the base of the stem takes no log, and the stem goes to the next one
      thick = DataFrame(name=["pole", "pulpwood"], sed=[80.0, 8.0]u"cm", minlength=[2.0, 2.0]u"m",
                        maxlength=[6.0, 6.0]u"m", kerf=[0.0, 0.0]u"m")
      pulp = logassortment(taper, D, H, thick)
      @test pulp.n[1] == 0 && pulp.v[1] == 0u"m^3"
      @test pulp.n[2] > 0
      # with logs of the longest length only: the sawlogs fit below the top of the product, the
      # shorter last log is not cut, and no log is longer than before
      fixed = logassortment(taper, D, H, products; stump=0.1u"m", fixedlength=true)
      @test 0.1u"m" + fixed.n[1] * 4.0u"m" + (fixed.n[1] - 1) * 0.005u"m" <= top
      @test 0.1u"m" + (fixed.n[1] + 1) * 4.0u"m" + fixed.n[1] * 0.005u"m" > top
      @test fixed.n[1] <= logs.n[1]
      @test sum(fixed.v) <= sum(logs.v)
      @test_throws ArgumentError logassortment(taper, D, H, select(products, Not(:kerf)))
      @test_throws DomainError logassortment(taper, D, H, products; stump=30.0u"m")
    end
  else
    @info "data/cubage.csv not found: the tests of the taper fits and of the assortments are skipped."
  end
end

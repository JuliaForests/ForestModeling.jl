@testset "fitbygroup" begin
  fit(t) = fitnls(curtis, t, :h, :d, [3.5, -10.0])
  models = fitbygroup(fit, heights, :plot)
  @test nrow(models) == 4
  @test sort(models.plot) == ["A", "B", "C", "D"]
  for row in eachrow(models)
    @test coef(row.model) ≈ coef(fit(heights[heights.plot .== row.plot, :]))
  end
  # each cell shows the equation of its group
  @test all(startswith.(repr.(models.model), "h = curtis(d; θ = "))

  # several grouping columns, and any fitting function
  bysize = transform(heights, :d => (d -> ifelse.(d .< 25u"cm", "small", "large")) => :size)
  counts = fitbygroup(nrow, bysize, [:plot, :size])
  @test sum(counts.model) == nrow(heights)
  @test names(counts) == ["plot", "size", "model"]
end

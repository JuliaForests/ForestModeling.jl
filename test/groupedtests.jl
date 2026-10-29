@testset "fitbygroup" begin
  fit(t) = fitglm(@formula(v ~ log(d) + log(h)), t)
  models = fitbygroup(fit, data, :clone)
  @test nrow(models) == 2
  @test sort(models.clone) == ["A", "B"]
  for row in eachrow(models)
    @test coef(row.model) ≈ coef(fit(data[data.clone .== row.clone, :]))
  end

  # several grouping columns, and any fitting function
  bycloneplot = fitbygroup(t -> nrow(t), data, [:clone, :plot])
  @test sum(bycloneplot.model) == nrow(data)
  @test names(bycloneplot) == ["clone", "plot", "model"]
end

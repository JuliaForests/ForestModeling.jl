@testset "fitglm" begin
  model = fitglm(@formula(v ~ log(d) + log(h)), data)
  @test model isa UnitfulModel

  # the same fit as GLM.jl on the values in the units of the columns
  plain = glm(@formula(v ~ log(d) + log(h)), DataFrame(d=ustrip.(data.d), h=ustrip.(data.h), v=ustrip.(data.v)),
              Gamma(), LogLink())
  @test coef(model) ≈ coef(plain)
  @test vcov(model) ≈ vcov(plain)
  @test nobs(model) == nrow(data)

  # Schumacher–Hall recovered: β₁ = 1.9 and β₂ = 1.0
  @test coef(model)[2] ≈ 1.9 atol = 0.05
  @test coef(model)[3] ≈ 1.0 atol = 0.08

  # the predictions and the residuals in the unit of the response, on the original scale
  ŷ = predict(model, data)
  @test unit(eltype(ŷ)) == u"m^3"
  @test ustrip.(u"m^3", ŷ) ≈ predict(plain)
  @test residuals(model) ≈ data.v .- ŷ

  # the new data are read in the units of the fit: diameters in mm give the same prediction
  inmm = transform(data, :d => (d -> uconvert.(u"mm", d)) => :d)
  @test predict(model, inmm) ≈ ŷ

  # the confidence interval of the mean contains the prediction
  p = predict(model, data[1:5, :]; interval=:confidence)
  @test names(p) == ["prediction", "lower", "upper"]
  @test all(p.lower .< p.prediction .< p.upper)

  # a categorical column as a fixed effect: an intercept and a slope per clone
  byclone = fitglm(@formula(v ~ log(d) * clone + log(h)), data)
  @test length(coef(byclone)) == 5

  # plain numbers, the ordinary linear regression of Spurr
  numbers = DataFrame(d2h=ustrip.(data.d) .^ 2 .* ustrip.(data.h), v=ustrip.(data.v))
  spurrmodel = fitglm(@formula(v ~ d2h), numbers; family=Normal(), link=IdentityLink())
  @test coef(spurrmodel) ≈ coef(lm(@formula(v ~ d2h), numbers))
  @test eltype(predict(spurrmodel, numbers)) <: Real
  @test predict(spurrmodel, numbers[1:3, :]; interval=:prediction) isa DataFrame

  # the response is a column: the link replaces its transformation
  @test_throws ArgumentError fitglm(@formula(log(v) ~ log(d) + log(h)), data)

  # the sandwich covariance of CovarianceMatrices.jl
  @test robustvcov(model) ≈ CovarianceMatrices.vcov(CovarianceMatrices.HC3(), model.model)
end

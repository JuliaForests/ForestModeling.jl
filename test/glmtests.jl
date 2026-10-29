@testset "fitglm" begin
  model = fitglm(@formula(v ~ log(d) + log(h)), volume)
  @test model isa UnitfulModel

  # the same fit as GLM.jl on the values in the units of the columns
  reference = glm(@formula(v ~ log(d) + log(h)), plain(volume), Gamma(), LogLink())
  @test coef(model) ≈ coef(reference)
  @test vcov(model) ≈ vcov(reference)
  @test stderror(model) ≈ stderror(reference)
  @test nobs(model) == nrow(volume)

  # the predictions and the residuals in the unit of the response, on the original scale
  ŷ = predict(model, volume)
  @test unit(eltype(ŷ)) == u"m^3"
  @test ustrip.(u"m^3", ŷ) ≈ predict(reference)
  @test residuals(model) ≈ volume.v .- ŷ

  # the new data are read in the units of the fit: diameters in mm give the same prediction
  inmm = transform(volume, :d => (d -> uconvert.(u"mm", d)) => :d)
  @test predict(model, inmm) ≈ ŷ

  # the confidence interval of the mean contains the prediction
  p = predict(model, volume[1:5, :]; interval=:confidence)
  @test names(p) == ["prediction", "lower", "upper"]
  @test all(p.lower .< p.prediction .< p.upper)

  # Curtis on the heights, the inverse of the diameter with the log link
  curtisglm = fitglm(@formula(h ~ inv(d)), heights)
  @test coef(curtisglm) ≈ coef(glm(@formula(h ~ inv(d)), plain(heights), Gamma(), LogLink()))

  # a categorical column as a fixed effect: an intercept and a slope per plot
  byplot = fitglm(@formula(h ~ inv(d) * plot), heights)
  @test length(coef(byplot)) == 8

  # the ordinary linear regression of Spurr, with plain numbers
  numbers = DataFrame(d2h=ustrip.(volume.d) .^ 2 .* ustrip.(volume.h), v=ustrip.(volume.v))
  spurrmodel = fitglm(@formula(v ~ d2h), numbers; family=Normal(), link=IdentityLink())
  @test coef(spurrmodel) ≈ coef(lm(@formula(v ~ d2h), numbers))
  @test eltype(predict(spurrmodel, numbers)) <: Real
  @test predict(spurrmodel, numbers[1:3, :]; interval=:prediction) isa DataFrame

  # the response is a column: the link replaces its transformation
  @test_throws ArgumentError fitglm(@formula(log(v) ~ log(d) + log(h)), volume)

  # shown as its equation on the original scale, one line in a table, with the coefficients in full
  @test repr(model) == "v = exp(" * join(string.(round.(coef(model); sigdigits=4)) .* ["", " * log(d)", " * log(h)"], " + ") * ")"
  @test repr(curtisglm) == "h = exp(" * string(round(coef(curtisglm)[1]; sigdigits=4)) * " - " *
                           string(round(-coef(curtisglm)[2]; sigdigits=4)) * " * inv(d))"
  @test occursin("Response: v (m^3), n = $(nrow(volume))", repr(MIME("text/plain"), model))

  # the sandwich covariance of CovarianceMatrices.jl
  @test robustvcov(model) ≈ CovarianceMatrices.vcov(CovarianceMatrices.HC3(), model.model)
end

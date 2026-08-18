@testset "fitting/ols.jl" begin
  @testset "fit: basic + known coefficients" begin
    m = fit(AllometricModel, @formula(log(h) ~ 1 + log(d)), dataPlain())
    @test m isa AllometricModel
    @test length(coef(m)) == 2
    @test coef(m) ≈ [0.39737866774301006, 0.8233376284585446] atol = 1e-8
    @test m.adjr² > 0.98
    @test nobs(m) == length(D)
    @test dof_residual(m) == length(D) - 2

    @test_throws ArgumentError fit(AllometricModel, @formula(cbrt(h) ~ 1 + d), dataPlain())
  end

  @testset "regression: every candidate's stored stats match a fresh predict() on it" begin
    # guards against the σ² (transformed-scale residual variance) computed for the
    # in-loop bias correction ever diverging from what gets stored on the model — if it
    # did, predict() would silently apply the wrong-scale correction to new predictions
    # even though the model's own reported r²/rmse/etc. (computed from the *correct*
    # in-loop ŷ) looked perfectly fine.
    models = regression(dataPlain(), :h, :d; nMax=2)
    for m in models
      ε = response(m) .- predict(m)
      @test sum(abs2, ε) ≈ m.sse atol = 1e-6
    end
  end

  @testset "predict: plain numbers unchanged, bias-corrected" begin
    m = fit(AllometricModel, @formula(log(h) ~ 1 + log(d)), dataPlain())
    p = predict(m)
    @test nonmissingtype(eltype(p)) <: Real
    @test !(nonmissingtype(eltype(p)) <: Unitful.Quantity)
    @test all(isapprox.(collect(skipmissing(p)), H, atol=1.0))  # bias-corrected fit should track observed heights, elementwise
    @test fitted(m) == p
  end

  @testset "predict: Unitful in, Unitful out, matches plain numerically" begin
    mPlain = fit(AllometricModel, @formula(log(h) ~ 1 + log(d)), dataPlain())
    mUnit = fit(AllometricModel, @formula(log(h) ~ 1 + log(d)), dataUnitful())

    pPlain = predict(mPlain)
    pUnit = predict(mUnit)
    @test nonmissingtype(eltype(pUnit)) <: Unitful.Quantity
    @test unit(pUnit[1]) == u"m"
    @test all(isapprox.(ustrip.(collect(skipmissing(pUnit))), collect(skipmissing(pPlain)), atol=1e-8))

    # predicting from data in a different but compatible unit converts first
    newDataCm = DataFrame(d=[300.0]u"cm")
    newDataMm = DataFrame(d=[3000.0]u"mm")
    @test isapprox(ustrip(predict(mUnit, newDataCm)[1]), ustrip(predict(mUnit, newDataMm)[1]), atol=1e-8)

    # response()/residuals() stay unit-consistent
    @test unit(response(mUnit)[1]) == u"m"
    @test nonmissingtype(eltype(residuals(mUnit))) <: Unitful.Quantity
  end

  @testset "regression + criteriaSelection + criteriaTable" begin
    models = regression(dataPlain(), :h, :d; nMax=2)
    @test length(models) > 50
    @test all(m -> m isa AllometricModel, models)

    best = criteriaSelection(models, :adjr2, :cv)
    @test best.adjr² > 0.99

    table = criteriaTable(models, :adjr2, :cv, :ev; best=5)
    @test nrow(table) == 5
    @test issorted(table.rank)

    @test_throws ArgumentError criteriaTable(models, :not_a_real_criterion)
    @test_throws ArgumentError regression(dataPlain(), :h, :d; nMax=6)
  end

  @testset "regression: Unitful models rank identically and predict with units" begin
    modelsU = regression(dataUnitful(), :h, :d; nMax=2)
    bestU = criteriaSelection(modelsU, :adjr2, :cv)
    pU = predict(bestU)
    @test nonmissingtype(eltype(pU)) <: Unitful.Quantity
    @test unit(pU[1]) == u"m"
  end

  @testset "fitRobust" begin
    f = @formula(log(h) ~ 1 + log(d))
    mOLS = fit(AllometricModel, f, dataPlain())
    mOLSviaRobust = fitRobust(f, dataPlain(), OLS)
    @test coef(mOLSviaRobust) ≈ coef(mOLS)

    for method in (SSE, MAE, HUBER, MSLE, MAPE)
      m = fitRobust(f, dataPlain(), method)
      @test m isa AllometricModel
      @test all(isfinite, coef(m))
      @test m.rmse > 0
      # the loss is minimized on the *transformed* scale (matching `fit`), so predictions
      # must land back near the observed heights, not still be on the log(h) scale (nor
      # any multiple-orders-of-magnitude-off artifact of back-transforming a value that
      # was never on the transformed scale to begin with)
      @test all(isapprox.(collect(skipmissing(predict(m))), H, atol=3.0))
    end

    mUnit = fitRobust(f, dataUnitful(), HUBER)
    pUnit = predict(mUnit)
    @test nonmissingtype(eltype(pUnit)) <: Unitful.Quantity
    @test all(isapprox.(ustrip.(collect(skipmissing(pUnit))), H, atol=3.0))
  end
end

@testset "equations" begin
  @test chapmanrichards(20.0, [25.0, 0.08, 1.2]; href=1.3) ≈ 1.3 + 25 * (1 - exp(-0.08 * 20))^1.2
  @test weibull(20.0, [25.0, 0.02, 1.3]; href=1.3) ≈ 1.3 + 25 * (1 - exp(-0.02 * 20^1.3))
  @test prodan(20.0, [2.0, 0.8, 0.03]; href=1.3) ≈ 1.3 + 400 / (2 + 16 + 12)
  @test curtis(20.0, [3.3, -8.0]) ≈ exp(3.3 - 0.4)
  @test schumacherhall(20.0, 18.0, [5e-5, 1.9, 1.0]) ≈ 5e-5 * 20^1.9 * 18
  @test spurr(20.0, 18.0, [0.01, 3.5e-5]) ≈ 0.01 + 3.5e-5 * 400 * 18
  @test power(20.0, [0.1, 2.4]) ≈ 0.1 * 20^2.4

  # the height of reference is the one given, the curve starts there
  @test chapmanrichards(0.0, [25.0, 0.08, 1.2]; href=1.37) == 1.37
  @test chapmanrichards(1e6, [25.0, 0.08, 1.2]; href=1.3) ≈ 26.3
end

@testset "volume equations" begin
  @test honer(20.0, 18.0, [400.0, 20000.0]) ≈ 400 / (400 + 20000 / 18)
  @test takata(20.0, 18.0, [20000.0, 400.0]) ≈ 400 * 18 / (20000 + 400 * 20)
end

# Run with LUXNLPMODELS_TEST_GPU=true; requires CUDA and LuxCUDA (loaded in runtests.jl).
@testset "GPU agrees with CPU" begin
  CUDA.allowscalar(false)
  dev = gpu_device()
  X, Y = toy_data(Float32, Xoshiro(9))
  mk(device) = LuxNLPModel(toy_mlp(), ArrayBatchSource((X, Y); batchsize = 16, seed = 3);
                           T = Float32, device = device, hvp_backend = :finite_difference)
  ncpu, ngpu = mk(cpu_device()), mk(dev)
  xc = copy(ncpu.meta.x0)
  xg = dev(xc)
  v = randn(Xoshiro(10), Float32, length(xc))

  @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5
  @test Array(grad(ngpu, xg)) ≈ grad(ncpu, xc) rtol = 1e-4
  @test Array(hprod(ngpu, xg, dev(v))) ≈ hprod(ncpu, xc, v) rtol = 1e-2
  next_batch!(ncpu)
  next_batch!(ngpu)
  @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5
  @test mean_objective(ngpu, xg) ≈ mean_objective(ncpu, xc) rtol = 1e-5
  @test accuracy(ngpu, xg, (X, Y)) ≈ accuracy(ncpu, xc, (X, Y)) atol = 1 / 64   # one near-tie allowed

  f0 = obj(ngpu, xg)
  stats = fomo(ngpu; max_iter = 20, max_time = 60.0)                    # FOMO supports GPU arrays
  @test stats.objective < f0
end

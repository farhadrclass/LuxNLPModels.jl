# Run with LUXNLPMODELS_TEST_GPU=true; requires CUDA and LuxCUDA (loaded in runtests.jl).
@testset "GPU agrees with CPU" begin
  CUDA.allowscalar(false)
  dev = gpu_device()
  data = toy_data(Float32, Xoshiro(13))
  mk(device) = LuxNLPModel(toy_mlp(), data, data; T = Float32, batchsize = 16, seed = 3,
                           device = device)
  ncpu, ngpu = mk(cpu_device()), mk(dev)
  xc = copy(ncpu.meta.x0)
  xg = dev(xc)
  v = randn(Xoshiro(14), Float32, length(xc))

  @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5
  @test Array(grad(ngpu, xg)) ≈ grad(ncpu, xc) rtol = 1e-4
  @test Array(hprod(ngpu, xg, dev(v))) ≈ hprod(ncpu, xc, v) rtol = 1e-2
  for _ in 1:4                                       # crosses an epoch boundary
    minibatch_next_train!(ncpu)
    minibatch_next_train!(ngpu)
  end
  @test epoch(ngpu) == 1
  @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5    # batch after the boundary is on the device
  @test accuracy(ngpu, xg) ≈ accuracy(ncpu, xc) atol = 1 / 64

  f0 = obj(ngpu, xg)
  stats = fomo(ngpu; max_iter = 20, max_time = 60.0)
  @test stats.objective < f0
end

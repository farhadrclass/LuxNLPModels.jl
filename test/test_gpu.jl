# Run with LUXNLPMODELS_TEST_GPU=true; requires CUDA and LuxCUDA (loaded in runtests.jl).
@testset "GPU" begin
  CUDA.allowscalar(false)
  dev = gpu_device()
  data = toy_data(Float32, Xoshiro(13))

  @testset "agrees with CPU" begin
    mk(device) = LuxNLPModel(toy_mlp(), data; T = Float32, batchsize = 16, seed = 3, device)
    ncpu, ngpu = mk(cpu_device()), mk(dev)
    xc = copy(ncpu.meta.x0)
    xg = dev(xc)
    v = randn(Xoshiro(14), Float32, length(xc))

    @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5
    @test Array(grad(ngpu, xg)) ≈ grad(ncpu, xc) rtol = 1e-4
    @test Array(hprod(ngpu, xg, dev(v))) ≈ hprod(ncpu, xc, v) rtol = 1e-2
    for _ in 1:4                                     # crosses an epoch boundary
      minibatch_next_train!(ncpu)
      minibatch_next_train!(ngpu)
    end
    @test epoch(ngpu) == 1
    @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5  # batch after the boundary is on the device
    set_batchsize!(ncpu, 32)
    set_batchsize!(ngpu, 32)
    @test obj(ngpu, xg) ≈ obj(ncpu, xc) rtol = 1e-5

    f0 = obj(ngpu, xg)
    stats = fomo(ngpu; max_iter = 20, max_time = 60.0)
    @test stats.objective < f0
  end

  layers = (BatchNorm(5), Dropout(0.5))
  @testset "f and ∇f describe one function with $(nameof(typeof(layer)))" for layer in layers
    m = Chain(Dense(3 => 5, tanh), layer, Dense(5 => 2))
    nlp = LuxNLPModel(m, data; T = Float32, batchsize = 16, device = dev)
    x = nlp.meta.x0
    for _ in 1:2
      @test obj(nlp, x) ≈ obj(nlp, x) rtol = 1e-6    # same batch statistics / Dropout mask
      @test all(isfinite, Array(grad(nlp, x)))
      minibatch_next_train!(nlp)
    end
  end
end

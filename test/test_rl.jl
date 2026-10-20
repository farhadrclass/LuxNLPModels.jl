@testset "RL: PPO objective on a replaceable rollout" begin
  nobs, nact, n = 4, 2, 32
  rollout(rng) = (obs = randn(rng, nobs, n), act = Float64.(I(nact)[:, rand(rng, 1:nact, n)]),
                  logp_old = fill(log(0.5), n), adv = randn(rng, n), ret = randn(rng, n))
  rng = Xoshiro(7)
  policy = Chain(Dense(nobs => 8, tanh), BranchLayer(Dense(8 => nact), Dense(8 => 1)))
  nlp = LuxNLPModel(policy, ArrayBatchSource(rollout(rng); batchsize = 8);
                    objective = ppo_objective(), T = Float64)
  x = nlp.meta.x0
  v = randn(rng, length(x))
  @test isfinite(obj(nlp, x))
  @test dot(grad(nlp, x), v) ≈ fd_dir(z -> obj(nlp, z), x, v) rtol = 1e-5
  for _ in 1:5
    next_batch!(nlp)
  end
  @test epoch(nlp) >= 1
  replace_data!(nlp, rollout(rng))                 # new rollout: passes restart
  @test epoch(nlp) == 0
  @test isfinite(obj(nlp, x))
end

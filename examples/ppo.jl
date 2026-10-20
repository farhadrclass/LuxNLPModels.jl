# PPO where each policy update is one solve on a fixed rollout. The environment loop is user
# code; `collect_rollout` is a placeholder to implement (e.g. with CommonRLInterface.jl).
#
# Caveats for comparisons with standard PPO:
# - TRAdam resets its moments at every update; PPO implementations usually keep Adam's state;
# - PPO's clipping is already a trust region in policy space; TRAdam adds one in parameter
#   space, so an ablation is needed to attribute any difference;
# - report many seeds with robust aggregates (e.g. interquartile mean, bootstrap intervals).
using Random, LinearAlgebra, Lux, LuxNLPModels, NLPModels
include(joinpath(@__DIR__, "rl.jl"))
include(joinpath(@__DIR__, "solvers", "tradam.jl"))

nobs_env, nact, nsteps, batchsize = 4, 2, 2048, 64           # e.g. CartPole
policy = Chain(Dense(nobs_env => 64, tanh), BranchLayer(Dense(64 => nact), Dense(64 => 1)))

"""
Run the policy for `nsteps`; return `(obs, act, logp_old, adv, ret)`, Float32 arrays with
observations along columns, one-hot `act`, normalized `adv`; keep these array types.
"""
collect_rollout(policy, ps, st, env, nsteps) = error("implement collect_rollout for your environment")

env = nothing

# Build the model on a placeholder rollout of the right types and shapes, then replace it.
placeholder = (obs = zeros(Float32, nobs_env, nsteps), act = zeros(Float32, nact, nsteps),
               logp_old = zeros(Float32, nsteps), adv = zeros(Float32, nsteps),
               ret = zeros(Float32, nsteps))
nlp = LuxNLPModel(policy, placeholder; loss_f = ppo_objective(), batchsize)
x = copy(nlp.meta.x0)

for update in 1:100
  rollout = collect_rollout(policy, params_view(nlp, x), Lux.testmode(nlp.st), env, nsteps)
  set_data_train!(nlp, rollout)
  # max_iter counts rejected iterations too: at most 10 passes over the rollout.
  stats = TRAdam(nlp; x = x, stochastic = true, max_iter = 10 * (nsteps ÷ batchsize),
                 max_time = Inf, atol = 0.0f0, rtol = 0.0f0, on_accept! = minibatch_next_train!)
  x .= stats.solution
end

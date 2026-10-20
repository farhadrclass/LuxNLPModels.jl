# Reinforcement learning. The environment loop is user code. A rollout is a NamedTuple of arrays
# held by an `ArrayBatchSource`; after each rollout, `replace_data!(nlp, rollout)` installs the new
# data. Each policy update is then a fixed finite-sum NLP (e.g. the PPO surrogate) that any JSO
# solver can minimize. Optimization theory covers this inner problem only.

"""
    ppo_objective(; clip = 0.2f0, vf_coef = 0.5f0, ent_coef = 0.01f0)

Clipped PPO surrogate for discrete actions (Schulman et al., 2017). Expects batches
`(obs, act, logp_old, adv, ret)` with one-hot `act`, and a model returning `(logits, value)`,
e.g. `Chain(trunk, BranchLayer(policy_head, value_head))` (CHECK: `BranchLayer` without a fusion
returns a tuple). Normalize advantages before `replace_data!`.
The objective is only piecewise smooth (`min`, `clamp`).
"""
function ppo_objective(; clip = 0.2f0, vf_coef = 0.5f0, ent_coef = 0.01f0)
  return function (model, ps, st, batch)
    (logits, value), st_new = model(batch.obs, ps, st)
    logp_all = logsoftmax(logits; dims = 1)
    logp = vec(sum(logp_all .* batch.act; dims = 1))
    ratio = exp.(logp .- batch.logp_old)
    surr = min.(ratio .* batch.adv, clamp.(ratio, 1 - clip, 1 + clip) .* batch.adv)
    l_pi = -mean(surr)
    l_v = mean(abs2, vec(value) .- batch.ret)
    ent = -mean(sum(exp.(logp_all) .* logp_all; dims = 1))
    return l_pi + vf_coef * l_v - ent_coef * ent, st_new, (; l_pi, l_v, ent)
  end
end

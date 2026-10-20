# Evaluation for reporting: not counted in the counters or pass counts, never switches the batch,
# runs layers in test mode (no dropout, running BatchNorm statistics). Data are moved to the
# device in chunks; lower `chunk` for large convolutional networks.

"""
    recalibrate_state!(nlp, x; nbatches = 50, seed = 0)

Recompute running normalization statistics (BatchNorm) at `x` with `nbatches` training-mode
forward passes on random batches of the *nominal* batch size, then commit them. Apply before
every evaluation, for every optimizer alike: methods that never commit a state (e.g. a plain
Adam loop) would otherwise be evaluated with stale statistics.

Only normalization statistics change: RNGs of stochastic layers are restored, so calling this
inside a solver callback does not change what the solver sees (training-mode outputs do not
depend on running statistics).
"""
function recalibrate_state!(nlp::LuxNLPModel, x::AbstractVector; nbatches::Int = 50, seed::Int = 0)
  data = alldata(nlp.source)
  n = _nobs(data)
  bs = nominal_batchsize(nlp.source)
  ps = params_view(nlp, x)
  rng = Xoshiro(seed)
  st = nlp.st
  for _ in 1:nbatches
    idx = rand(rng, 1:n, bs)
    _, st, _ = nlp.objective(nlp.model, ps, st, nlp.device(_obs(data, idx)))
  end
  st = _restore_rngs(st, nlp.st)
  nlp.st = st
  nlp.st_last = st
  return nlp
end

"""
    mean_objective(nlp, x, data = training data; chunk = 1_000)

Mean objective over `data` in test mode, in chunks. Assumes the objective is a mean over
observations; in test mode BatchNorm acts per observation, so chunking does not change the value.
This is not the training-mode objective that solvers minimize when the model has BatchNorm.
"""
function mean_objective(nlp::LuxNLPModel, x::AbstractVector, data = alldata(nlp.source);
                        chunk::Int = 1_000)
  n = _nobs(data)
  ps = params_view(nlp, x)
  st = Lux.testmode(nlp.st)
  total = 0.0
  for lo in 1:chunk:n
    idx = lo:min(lo + chunk - 1, n)
    fx, _, _ = nlp.objective(nlp.model, ps, st, nlp.device(_obs(data, idx)))
    total += Float64(fx) * length(idx)
  end
  return total / n
end

"""
    accuracy(nlp, x, (X, Y); chunk = 1_000)

Classification accuracy at `x`, `Y` one-hot. The forward pass runs on the device; the argmax is
taken on the host (logits are small), which avoids GPU-specific reductions.
"""
function accuracy(nlp::LuxNLPModel, x::AbstractVector, (X, Y); chunk::Int = 1_000)
  n = _nobs(X)
  ps = params_view(nlp, x)
  st = Lux.testmode(nlp.st)
  host = cpu_device()
  correct = 0
  for lo in 1:chunk:n
    idx = lo:min(lo + chunk - 1, n)
    ŷ, _ = nlp.model(nlp.device(_obs(X, idx)), ps, st)
    ŷh, Yh = host(ŷ), host(_obs(Y, idx))
    for j in axes(ŷh, 2)
      correct += argmax(view(ŷh, :, j)) == argmax(view(Yh, :, j))
    end
  end
  return correct / n
end

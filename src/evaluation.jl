"""
    accuracy(nlp, x; data = nlp.data_test)

Fraction of correct predictions at `x` on `data = (X, Y)` with one-hot `Y`, in test mode.
Not counted in the counters and never changes the minibatch or `nlp.st`, so it can be called
from a solver callback.
"""
function accuracy(nlp::LuxNLPModel, x::AbstractVector; data = nlp.data_test)
  data === nothing && throw(ArgumentError("no test data in the model; pass data = (X, Y)"))
  ps = params_view(nlp, x)
  st = _testmode_state(nlp, ps)
  cpu = cpu_device()
  correct = 0
  for (X, Y) in DataLoader(data; batchsize = 1_000)    # forward pass on the device, in chunks
    ŷ = cpu(first(nlp.model(nlp.device(X), ps, st)))
    correct += count(argmax(ŷ; dims = 1) .== argmax(cpu(Y); dims = 1))
  end
  return correct / numobs(data)
end

# `obj` and `grad!` never update BatchNorm's running statistics, so before switching to test mode
# they are recomputed at `ps` with training-mode passes on random training batches. Without
# such statistics no pass is made.
function _testmode_state(nlp::LuxNLPModel, ps; nbatches::Int = 50)
  st = nlp.st
  if _has_running_stats(st)
    data = nlp.data_train
    n = numobs(data)
    batchsize = min(numobs(nlp.batch), 1024)
    rng = Xoshiro(0)
    for _ in 1:nbatches
      batch = nlp.device(getobs(data, rand(rng, 1:n, batchsize)))
      _, st, _ = nlp.loss_f(nlp.model, ps, st, batch)
    end
  end
  return Lux.testmode(st)
end

_has_running_stats(st::NamedTuple) =
  (haskey(st, :running_mean) && st.running_mean !== nothing) || any(_has_running_stats, values(st))
_has_running_stats(st::Tuple) = any(_has_running_stats, st)
_has_running_stats(_) = false

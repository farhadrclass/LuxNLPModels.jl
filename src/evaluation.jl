# Evaluation in test mode. Not counted in the counters; never changes the minibatch or `nlp.st`,
# so it can be called from a solver callback without changing what the solver sees.

"""
    testmode_state(nlp, x; nbatches = 50, batchsize = min(train batch size, 1024), seed = 0)

Layer state for evaluating the network at `x` in test mode. If the model keeps running
normalization statistics (BatchNorm), they are first recomputed at `x` with `nbatches`
training-mode passes on random training batches, so evaluation does not depend on which solver
produced `x` or on whether it ever updated those statistics. Without such statistics no pass is
made. `nlp.st` is not modified.

Use it for any test-mode metric: `ŷ, _ = nlp.model(X, params_view(nlp, x), testmode_state(nlp, x))`.
"""
function testmode_state(nlp::LuxNLPModel, x::AbstractVector; nbatches::Int = 50,
                        batchsize::Int = min(nlp.loader_opts.batchsize, 1024), seed::Int = 0)
  st = nlp.st
  if _has_running_stats(st)
    data = nlp.data_train
    n = numobs(data)
    ps = params_view(nlp, x)
    rng = Xoshiro(seed)
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

"""
    accuracy(nlp, x; data = nlp.data_test, st = testmode_state(nlp, x), chunk = 1_000)

Fraction of correct predictions at `x` on `data = (X, Y)` with one-hot `Y`, in test mode.
The forward pass runs on the device in chunks; the argmax is taken on the host. Pass `st` to
reuse a state computed by `testmode_state`.
"""
function accuracy(nlp::LuxNLPModel, x::AbstractVector; data = nlp.data_test,
                  st = nothing, chunk::Int = 1_000)
  data === nothing && throw(ArgumentError("no test data in the model; pass data = (X, Y)"))
  st === nothing && (st = testmode_state(nlp, x))
  ps = params_view(nlp, x)
  host = cpu_device()
  correct = 0
  for (X, Y) in DataLoader(data; batchsize = chunk, shuffle = false, partial = true)
    ŷ = host(first(nlp.model(nlp.device(X), ps, st)))
    Yh = host(Y)
    for j in axes(ŷ, 2)
      correct += argmax(view(ŷ, :, j)) == argmax(view(Yh, :, j))
    end
  end
  return correct / numobs(data)
end

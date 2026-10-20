# The stochastic contract, for any solver, first or second order:
#   1. f(x_k), f(x_k + s_k), g(x_k) and any Hessian-vector product at x_k use the same batch;
#   2. the batch changes only after an accepted step: the solver calls `commit_state!` and then
#      `next_batch!` (typically through an `on_accept!` hook), and recomputes f and g;
#   3. callbacks and evaluation code never switch the batch.
# Stochastic layers (Dropout) draw from the committed state, which is reseeded at every batch
# switch: all evaluations on one batch use the same masks, a new batch gets new masks.

"Move to the next batch of the source (in-place gather on the host, one host-to-device copy)."
function next_batch!(nlp::LuxNLPModel)
  advance!(nlp.source)
  return _batch_switched!(nlp)
end

"""
    set_batch!(nlp, idx)

Use the observations `idx`. If the size differs from the current batch, the device buffer is
reallocated and AD re-prepared. The next `next_batch!` returns to the nominal batch size.
"""
function set_batch!(nlp::LuxNLPModel, idx)
  set_indices!(nlp.source, idx)
  return _batch_switched!(nlp)
end

"""
    use_full_batch!(nlp)

Make the whole training set the current batch, for deterministic solvers. One forward and one
backward pass then run over all observations at once: use small problems or subsets. With
BatchNorm, this objective uses full-data batch statistics; it is not the average of minibatch
objectives.
"""
use_full_batch!(nlp::LuxNLPModel) = set_batch!(nlp, 1:_nobs(alldata(nlp.source)))

"""
    replace_data!(nlp, data)

Replace the source data (same types, same shape per observation), restart its passes and load
the first batch. Use after collecting a new RL rollout.
"""
function replace_data!(nlp::LuxNLPModel{T}, data) where {T}
  _check_eltype(T, data)
  replace_data!(nlp.source, data)
  return _batch_switched!(nlp)
end

"Adopt the layer state returned by the last `obj` call; call after accepting a trial point."
commit_state!(nlp::LuxNLPModel) = (nlp.st = nlp.st_last; nlp)

current_batch(nlp::LuxNLPModel) = nlp.batch
epoch(nlp::LuxNLPModel) = epoch(nlp.source)

function _batch_switched!(nlp::LuxNLPModel)
  host = getbatch(nlp.source)
  if _nobs(host) == _nobs(nlp.batch)
    _copyto_batch!(nlp.batch, host)                # host -> device, no allocation
  else
    nlp.batch = nlp.device(host)
    _prepare_grad!(nlp)
    nlp.prep_hvp = nothing
  end
  nlp.st = _reseed_rngs(nlp.st, nlp.rng)
  nlp.st_last = nlp.st
  return nlp
end

_copyto_batch!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
_copyto_batch!(dst::Union{Tuple, NamedTuple}, src::Union{Tuple, NamedTuple}) =
  map(_copyto_batch!, dst, src)

# Every RNG found in the layer state gets its own fresh seed (so two Dropout layers of the same
# size do not share masks); the RNG type is kept, so the state type does not change.
# CHECK on GPU: MLDataDevices keeps Xoshiro states as Xoshiro.
_reseed_rngs(st::Union{NamedTuple, Tuple}, rng::AbstractRNG) = map(s -> _reseed_rngs(s, rng), st)
function _reseed_rngs(r::AbstractRNG, rng::AbstractRNG)
  r2 = deepcopy(r)
  Random.seed!(r2, rand(rng, UInt64))
  return r2
end
_reseed_rngs(x, ::AbstractRNG) = x

# Take everything from `new` except the RNGs, which are taken from `old`.
_restore_rngs(new::Union{NamedTuple, Tuple}, old::Union{NamedTuple, Tuple}) = map(_restore_rngs, new, old)
_restore_rngs(::AbstractRNG, old::AbstractRNG) = old
_restore_rngs(new, old) = new

"""
    minibatch_next_train!(nlp) -> Bool

Load the next training minibatch and reseed random layers. Returns `false` when the pass was
complete: a new pass starts (reshuffled), its first minibatch is loaded and `epoch(nlp)`
increases. Every evaluation between two calls uses the same minibatch and the same masks, so a
solver comparing f at two points between two calls compares values of one function.
"""
function minibatch_next_train!(nlp::LuxNLPModel)
  next = iterate(nlp.loader, nlp.loader_state)
  complete = next === nothing
  if complete
    nlp.epoch += 1
    next = iterate(nlp.loader)
  end
  host, nlp.loader_state = next
  _load!(nlp, host)
  return !complete
end

"""
    reset_minibatch_train!(nlp)

Restart the current pass (reshuffled) at its first minibatch; `epoch(nlp)` is unchanged.
"""
function reset_minibatch_train!(nlp::LuxNLPModel)
  host, nlp.loader_state = iterate(nlp.loader)
  _load!(nlp, host)
  return nlp
end

"""
    set_data_train!(nlp, data)

Replace the training data, e.g. by a new reinforcement-learning rollout. `data` must have the
same type and shape per observation as the current data, and at least `batchsize`
observations. Restarts at epoch 0 and loads the first minibatch.
"""
function set_data_train!(nlp::LuxNLPModel{T}, data) where {T}
  typeof(data) == typeof(nlp.data_train) ||
    throw(ArgumentError("new data must have type $(typeof(nlp.data_train))"))
  _obsshape(data) == _obsshape(nlp.data_train) ||
    throw(DimensionMismatch("new data must have the same shape per observation"))
  numobs(data) >= nlp.loader_opts.batchsize ||
    throw(ArgumentError("new data has fewer than batchsize = $(nlp.loader_opts.batchsize) observations"))
  _check_eltype(T, data)
  nlp.data_train = data
  nlp.loader = DataLoader(data; nlp.loader_opts..., partial = false, buffer = true)
  nlp.epoch = 0
  return reset_minibatch_train!(nlp)
end

"Number of completed passes over the training data."
epoch(nlp::LuxNLPModel) = nlp.epoch

function _load!(nlp::LuxNLPModel, host)
  _copyto_batch!(nlp.batch, host)                 # host -> device, into a preallocated buffer
  nlp.st = _reseed_rngs(nlp.st, nlp.rng)
  return nlp
end

_copyto_batch!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
_copyto_batch!(dst::Union{Tuple, NamedTuple}, src::Union{Tuple, NamedTuple}) = map(_copyto_batch!, dst, src)

_obsshape(A::AbstractArray) = Base.front(size(A))
_obsshape(t::Union{Tuple, NamedTuple}) = map(_obsshape, Tuple(t))

# Each RNG in the layer state gets its own fresh seed (two Dropout layers of the same size do
# not share masks). The RNG type is kept, so the state type does not change.
_reseed_rngs(st::Union{NamedTuple, Tuple}, rng::AbstractRNG) = map(s -> _reseed_rngs(s, rng), st)
function _reseed_rngs(r::AbstractRNG, rng::AbstractRNG)
  r2 = deepcopy(r)
  Random.seed!(r2, rand(rng, UInt64))
  return r2
end
_reseed_rngs(x, ::AbstractRNG) = x

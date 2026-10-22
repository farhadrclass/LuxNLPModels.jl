"""
    minibatch_next_train!(nlp) -> Bool

Load the next training minibatch and reseed random layers (new Dropout masks). Returns `false`
when the pass was complete: a new, reshuffled pass then starts at its first minibatch and
`epoch(nlp)` increases. With a first-order solver, call it in the callback:
`R2(nlp; callback = (nlp, solver, stats) -> minibatch_next_train!(nlp))`. With `trunk`, call it
between warm-started solves (see the README).
"""
function minibatch_next_train!(nlp::LuxNLPModel)
  next = iterate(nlp.loader, nlp.loader_state)
  complete = next === nothing
  if complete
    nlp.epoch += 1
    next = iterate(nlp.loader)
  end
  _load!(nlp, next)
  return !complete
end

"""
    set_batchsize!(nlp, batchsize)

Use minibatches of `batchsize` observations from now on, e.g. to grow the batch as a
trust-region method converges. The current pass restarts (reshuffled if `shuffle`) with
minibatches of the new size; `epoch(nlp)` is unchanged.
"""
function set_batchsize!(nlp::LuxNLPModel, batchsize::Int)
  nlp.loader = _loader(nlp.data_train, batchsize, nlp.shuffle, nlp.rng)
  _load!(nlp, iterate(nlp.loader))
  # Zygote and finite differences ignore sizes; prepare again for backends that do not.
  nlp.prep, nlp.hvp_prep = _prepare(nlp.grad_backend, nlp.hvp_backend, nlp.meta.x0, _contexts(nlp))
  return nlp
end

"Number of completed passes over the training data."
epoch(nlp::LuxNLPModel) = nlp.epoch

# Make the host minibatch of `next = (batch, loader state)` current, with new Dropout masks.
function _load!(nlp::LuxNLPModel, (host, state))
  nlp.batch = nlp.device(host)
  nlp.loader_state = state
  nlp.st = _reseed_rngs(nlp.st, nlp.rng)
  return nlp
end

# Each RNG in the layer state gets its own fresh seed (two Dropout layers of the same size do
# not share masks). The RNG type is kept, so the state type does not change.
_reseed_rngs(st::Union{NamedTuple, Tuple}, rng::AbstractRNG) = map(s -> _reseed_rngs(s, rng), st)
function _reseed_rngs(r::AbstractRNG, rng::AbstractRNG)
  r2 = deepcopy(r)
  Random.seed!(r2, rand(rng, UInt64))
  return r2
end
_reseed_rngs(x, ::AbstractRNG) = x

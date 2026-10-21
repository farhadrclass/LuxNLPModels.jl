"""
    minibatch_next_train!(nlp) -> Bool

Load the next training minibatch and reseed random layers (new Dropout masks). Returns `false`
when the pass was complete: a new, reshuffled pass then starts at its first minibatch and
`epoch(nlp)` increases. To train an existing solver on minibatches, call it in the callback:
`R2(nlp; callback = (nlp, solver, stats) -> minibatch_next_train!(nlp))`.
"""
function minibatch_next_train!(nlp::LuxNLPModel)
  next = iterate(nlp.loader, nlp.loader_state)
  complete = next === nothing
  if complete
    nlp.epoch += 1
    next = iterate(nlp.loader)
  end
  host, nlp.loader_state = next
  nlp.batch = nlp.device(host)
  nlp.st = _reseed_rngs(nlp.st, nlp.rng)
  return !complete
end

"Number of completed passes over the training data."
epoch(nlp::LuxNLPModel) = nlp.epoch

# Each RNG in the layer state gets its own fresh seed (two Dropout layers of the same size do
# not share masks). The RNG type is kept, so the state type does not change.
_reseed_rngs(st::Union{NamedTuple, Tuple}, rng::AbstractRNG) = map(s -> _reseed_rngs(s, rng), st)
function _reseed_rngs(r::AbstractRNG, rng::AbstractRNG)
  r2 = deepcopy(r)
  Random.seed!(r2, rand(rng, UInt64))
  return r2
end
_reseed_rngs(x, ::AbstractRNG) = x

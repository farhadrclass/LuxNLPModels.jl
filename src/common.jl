# Functionality shared by LuxNLPModel and LuxNLSModel.

const LuxModels{T, S} = Union{LuxNLPModel{T, S}, LuxNLSModel{T, S}}

# ---------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------

"""
    unflatten(nlp, x)

`ComponentArray` view of the flat vector `x` with the structure of the Lux parameters
(no copy), e.g. `unflatten(nlp, x).layer_1.weight`.
"""
unflatten(nlp::LuxModels, x::AbstractVector) = ComponentArray(x, nlp.axes)

"""
    predict(nlp, x, input; st = LuxCore.testmode(nlp.st))

Evaluate the network with parameters `x` on `input` (inference mode by default).
"""
predict(nlp::LuxModels, x::AbstractVector, input; st = LuxCore.testmode(nlp.st)) =
  first(LuxCore.apply(nlp.model, input, unflatten(nlp, x), st))

# ---------------------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------------------

"""
    current_batch(nlp)

The batch on which the objective (or residual) and its derivatives are currently evaluated.
"""
current_batch(nlp::LuxModels) = nlp.batch

"""
    set_batch!(nlp, batch)

Evaluate on `batch` from now on (same type as the batches drawn from the data, already on
the right device). For a [`LuxNLSModel`](@ref) the batch must produce `nequ` residuals.
"""
function set_batch!(nlp::LuxModels, batch)
  shp = _shape(batch)
  if shp != nlp.prep_shape
    nlp isa LuxNLSModel && _check_nequ(nlp, batch)
    _clear_preps!(nlp)          # AD preparations may depend on sizes
    nlp.prep_shape = shp
  end
  nlp.batch = batch
  nlp.batch_nobs = _batch_nobs(nlp.sampler, batch)
  return nlp
end

function _check_nequ(nls::LuxNLSModel, batch)
  m = length(_eval_residual_on(nls, nls.backend, nls.meta.x0, batch))
  m == nls.nls_meta.nequ || throw(DimensionMismatch(
    "this batch produces $m residuals but the model has nequ = $(nls.nls_meta.nequ); " *
    "build a separate LuxNLSModel for data of another size"))
  return nothing
end

"""
    next_batch!(nlp)

Advance to the next minibatch (reshuffling at the end of every epoch). Returns `nlp`.
No-op in full-batch mode.
"""
function next_batch!(nlp::LuxModels)
  nlp.sampler === nothing && return nlp
  return set_batch!(nlp, _draw!(nlp.sampler, nlp.data))
end

"""
    reset_batches!(nlp)

Restart minibatch sampling at epoch 1 with a fresh permutation.
"""
function reset_batches!(nlp::LuxModels)
  nlp.sampler === nothing && return nlp
  _reset_sampler!(nlp.sampler)
  return next_batch!(nlp)
end

"""
    use_full_batch!(nlp)

Switch to full-batch mode (deterministic objective on all the data). Minibatch sampling is
discarded. For a [`LuxNLSModel`](@ref) the full data must produce `nequ` residuals.
"""
function use_full_batch!(nlp::LuxModels)
  set_batch!(nlp, nlp.data)
  nlp.sampler = nothing
  nlp.batch_nobs = nlp.nobs
  return nlp
end

"""
    epoch(nlp)

Epoch of the current minibatch (1-based); `1` in full-batch mode.
"""
epoch(nlp::LuxModels) = nlp.sampler === nothing ? 1 : _epoch(nlp.sampler)

"""
    minibatch_next_train!(nlp) -> Bool

FluxNLPModels-compatible batch switching: advance to the next minibatch and return `true`,
or `false` when the epoch is exhausted (the new batch is then the first of the next epoch,
with a new permutation). Always `false` in full-batch mode.
"""
function minibatch_next_train!(nlp::LuxModels)
  nlp.sampler === nothing && return false
  e = epoch(nlp)
  next_batch!(nlp)
  return epoch(nlp) == e
end

"""
    reset_minibatch_train!(nlp)

FluxNLPModels-compatible alias of [`reset_batches!`](@ref).
"""
reset_minibatch_train!(nlp::LuxModels) = reset_batches!(nlp)

# ---------------------------------------------------------------------------------------
# Work accounting
# ---------------------------------------------------------------------------------------

"""
    work(nlp)

Named tuple with forward/backward passes, Jacobian- and Hessian-vector products,
processed observations and the corresponding number of data passes (`epochs = samples / nobs`).
"""
function work(nlp::LuxModels)
  w = nlp.work
  return (forward = w.forward, backward = w.backward, jvp = w.jvp, hvp = w.hvp,
          samples = w.samples, epochs = w.samples / nlp.nobs)
end

function reset_work!(nlp::LuxModels)
  nlp.work = WorkCounters()
  return nlp
end

function NLPModels.reset!(nlp::LuxModels)
  reset!(nlp.counters)
  reset_work!(nlp)
  return nlp
end

@inline function _account!(nlp::LuxModels; forward = 0, backward = 0, jvp = 0, hvp = 0)
  w = nlp.work
  w.forward += forward
  w.backward += backward
  w.jvp += jvp
  w.hvp += hvp
  w.samples += (forward + jvp + hvp) * nlp.batch_nobs
  return nothing
end

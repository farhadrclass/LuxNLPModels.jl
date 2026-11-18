"""
    WorkCounters

Implementation-independent work measures, complementary to the NLPModels `Counters`
(which count *calls*, e.g. `objgrad!` increments both `neval_obj` and `neval_grad` although
it costs a single forward + backward pass).

- `forward`:  forward passes through the network (from `obj`, `grad!`, `objgrad!`);
- `backward`: reverse passes (from `grad!`, `objgrad!`);
- `hvp`:      Hessian-vector products (from `hprod!`);
- `samples`:  observations processed by forward passes and Hessian-vector products.
              `samples / nobs` is the number of
              data passes ("epochs equivalent"), the natural budget unit for comparing
              minibatch methods.
"""
mutable struct WorkCounters
  forward::Int
  backward::Int
  hvp::Int
  samples::Int
end
WorkCounters() = WorkCounters(0, 0, 0, 0)

"""
    LuxNLPModel{T, S} <: AbstractNLPModel{T, S}

Unconstrained NLPModel `min_x f(x) = loss(model, ps(x), st, batch)` where `ps(x)` is the
`ComponentArray` view of the flat vector `x` and `batch` is the current batch (the full data
set in full-batch mode, a minibatch otherwise).

See [`LuxNLPModel(model, loss, data; kwargs...)`](@ref) for construction.
"""
mutable struct LuxNLPModel{T, S <: AbstractVector{T}, M, L, AX, ST, D, B, AD, HAD} <:
               AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  work::WorkCounters
  model::M
  loss::L
  axes::AX
  st::ST
  data::D
  batch::B
  nobs::Int          # observations in `data` (1 if not countable: one pass = one evaluation)
  batch_nobs::Int    # observations in `batch` (1 if not countable)
  sampler::Union{Nothing, MinibatchSampler}
  backend::AD
  hvp_backend::HAD
  grad_prep::Any
  hvp_prep::Any
  prep_shape::Any
end

"""
    LuxNLPModel(model, loss, data; kwargs...)

Build an NLPModel whose objective is the training loss of the Lux `model`.

# Arguments
- `model`: any `LuxCore.AbstractLuxLayer`.
- `loss`: objective in Lux's training convention, `loss(model, ps, st, batch)`, returning
  either a scalar or a tuple `(value, st, stats)` (only `value` is used). All Lux losses
  (`MSELoss()`, `CrossEntropyLoss(; logits = Val(true))`, ...) satisfy this when `batch = (x, y)`.
  Custom losses (e.g. physics-informed residuals) can take any `batch` object.
- `data`: the full training data, e.g. `(x, y)`, passed to `loss` as `batch` in full-batch mode.

# Keyword arguments
- `rng = Random.default_rng()`: used for `LuxCore.setup` (if `ps`/`st` are not given) and shuffling.
- `ps = nothing`, `st = nothing`: initial parameters (NamedTuple or ComponentArray) and states.
- `device = identity`: applied to parameters, states and data, e.g. `gpu_device()` from Lux.
- `batchsize = nothing`: `nothing` for full-batch (deterministic objective); an integer to
  enable minibatch mode, where the current batch is advanced with [`next_batch!`](@ref).
- `shuffle = true`, `partial = false`: minibatch sampling options.
- `backend = AutoZygote()`: AD backend (ADTypes) for gradients.
- `hvp_backend = SecondOrder(AutoForwardDiff(), backend)`: backend for `hprod`.
- `name = "LuxNLPModel"`.

# Notes
- The Lux state `st` is *not* updated by evaluations, so the objective is a deterministic
  function of `x` for a fixed batch (a fixed dropout mask, batch statistics for BatchNorm in
  training mode). Pass `st = LuxCore.testmode(st)` to evaluate in inference mode.
- The element type of the problem is the element type of the parameters (Float32 by default
  in Lux). Pass `ps = f64(ps)` and Float64 data for double precision.
- Work accounting ([`work`](@ref)) counts observations with [`numobs`](@ref). Data without a
  single observation count (e.g. PINN interior and boundary points of different sizes) are
  treated as one observation, so one data pass is one evaluation on the full data.
"""
function LuxNLPModel(model, loss, data;
                     rng::AbstractRNG = Random.default_rng(),
                     ps = nothing,
                     st = nothing,
                     device = identity,
                     batchsize::Union{Nothing, Integer} = nothing,
                     shuffle::Bool = true,
                     partial::Bool = false,
                     backend::AbstractADType = AutoZygote(),
                     hvp_backend = DI.SecondOrder(AutoForwardDiff(), backend),
                     name::AbstractString = "LuxNLPModel")
  if ps === nothing || st === nothing
    ps0, st0 = LuxCore.setup(rng, model)
    ps = ps === nothing ? ps0 : ps
    st = st === nothing ? st0 : st
  end
  ps_ca = device(ps isa ComponentArray ? ps : ComponentArray(ps))
  st = device(st)
  data = device(data)

  x0 = copy(getdata(ps_ca))
  ax = getaxes(ps_ca)
  T = eltype(x0)
  S = typeof(x0)
  n = length(x0)
  # No bound analysis: it calls `findall` on the bound vectors, which not every GPU array
  # type implements; `unconstrained(nlp)` then falls back to a GPU-friendly reduction.
  meta = NLPModelMeta{T, S}(n; x0 = x0,
                            lvar = fill!(similar(x0), T(-Inf)),
                            uvar = fill!(similar(x0), T(Inf)),
                            variable_bounds_analysis = false,
                            name = name)

  # Data without a single observation count (e.g. PINN interior and boundary points of
  # different sizes) count as one observation: one data pass = one full evaluation.
  nobs = _count(data)
  countable = nobs > 0
  countable || (nobs = 1)
  if batchsize === nothing
    sampler = nothing
    batch = data
  else
    countable || throw(ArgumentError("minibatch mode needs a data container with countable observations; see `LuxNLPModels.numobs`"))
    sampler = MinibatchSampler(nobs, batchsize; shuffle, partial, rng)
    batch = getobs(data, next_indices!(sampler))
  end

  return LuxNLPModel(meta, Counters(), WorkCounters(), model, loss, ax, st, data, batch,
                     nobs, max(_count(batch), 1), sampler, backend, hvp_backend,
                     nothing, nothing, _shape(batch))
end

_count(x) = try
  numobs(x)
catch
  0
end

function Base.show(io::IO, nlp::LuxNLPModel)
  mode = nlp.sampler === nothing ? "full batch" :
         "minibatch (batchsize = $(nlp.sampler.batchsize), epoch $(nlp.sampler.epoch))"
  print(io, "LuxNLPModel: ", nlp.meta.nvar, " variables of type ", eltype(nlp.meta.x0),
        " in ", typeof(nlp.meta.x0).name.name, ", ", mode)
end

# ---------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------

"""
    unflatten(nlp, x)

`ComponentArray` view of the flat vector `x` with the structure of the Lux parameters
(no copy), e.g. `unflatten(nlp, x).layer_1.weight`.
"""
unflatten(nlp::LuxNLPModel, x::AbstractVector) = ComponentArray(x, nlp.axes)

"""
    predict(nlp, x, input; st = LuxCore.testmode(nlp.st))

Evaluate the network with parameters `x` on `input` (inference mode by default).
"""
predict(nlp::LuxNLPModel, x::AbstractVector, input; st = LuxCore.testmode(nlp.st)) =
  first(LuxCore.apply(nlp.model, input, unflatten(nlp, x), st))

# ---------------------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------------------

"""
    current_batch(nlp)

The batch on which `obj`, `grad!`, `hprod!` are currently evaluated.
"""
current_batch(nlp::LuxNLPModel) = nlp.batch

"""
    set_batch!(nlp, batch)

Evaluate the objective on `batch` from now on (must have the same type as the batches
produced from the data, already on the right device).
"""
function set_batch!(nlp::LuxNLPModel, batch)
  nlp.batch = batch
  nlp.batch_nobs = max(_count(batch), 1)
  shp = _shape(batch)
  if shp != nlp.prep_shape  # AD preparations may depend on sizes
    nlp.grad_prep = nothing
    nlp.hvp_prep = nothing
    nlp.prep_shape = shp
  end
  return nlp
end

"""
    next_batch!(nlp)

Advance to the next minibatch (reshuffling at the end of every epoch). Returns `nlp`.
No-op in full-batch mode.
"""
function next_batch!(nlp::LuxNLPModel)
  nlp.sampler === nothing && return nlp
  return set_batch!(nlp, getobs(nlp.data, next_indices!(nlp.sampler)))
end

"""
    reset_batches!(nlp)

Restart minibatch sampling at epoch 1 with a fresh permutation.
"""
function reset_batches!(nlp::LuxNLPModel)
  nlp.sampler === nothing && return nlp
  _reset_sampler!(nlp.sampler)
  return set_batch!(nlp, getobs(nlp.data, next_indices!(nlp.sampler)))
end

"""
    use_full_batch!(nlp)

Switch to full-batch mode (deterministic objective on all the data). Minibatch sampling
is discarded.
"""
function use_full_batch!(nlp::LuxNLPModel)
  nlp.sampler = nothing
  return set_batch!(nlp, nlp.data)
end

"""
    epoch(nlp)

Epoch of the current minibatch (1-based); `1` in full-batch mode.
"""
epoch(nlp::LuxNLPModel) = nlp.sampler === nothing ? 1 : nlp.sampler.epoch

# ---------------------------------------------------------------------------------------
# Work accounting
# ---------------------------------------------------------------------------------------

"""
    work(nlp)

Named tuple with the forward/backward/hvp passes, processed observations and the
corresponding number of data passes (`epochs = samples / nobs`).
"""
function work(nlp::LuxNLPModel)
  w = nlp.work
  return (forward = w.forward, backward = w.backward, hvp = w.hvp, samples = w.samples,
          epochs = nlp.nobs > 0 ? w.samples / nlp.nobs : NaN)
end

function reset_work!(nlp::LuxNLPModel)
  nlp.work = WorkCounters()
  return nlp
end

function NLPModels.reset!(nlp::LuxNLPModel)
  reset!(nlp.counters)
  reset_work!(nlp)
  return nlp
end

@inline function _account!(nlp::LuxNLPModel; forward = 0, backward = 0, hvp = 0)
  w = nlp.work
  w.forward += forward
  w.backward += backward
  w.hvp += hvp
  w.samples += (forward + hvp) * nlp.batch_nobs
  return nothing
end

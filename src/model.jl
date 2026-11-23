"""
    WorkCounters

Implementation-independent work measures, complementary to the NLPModels counters
(which count *calls*, e.g. `objgrad!` increments both `neval_obj` and `neval_grad` although
it costs a single forward + backward pass).

- `forward`:  forward passes through the network (objective, residual, gradient, Jᵀv);
- `backward`: reverse passes (gradient, Jᵀv);
- `jvp`:      forward-mode Jacobian-vector products of the residual (`jprod_residual!`);
- `hvp`:      Hessian-vector products (`hprod!`);
- `samples`:  observations processed by forward passes, Jacobian- and Hessian-vector
              products. `samples / nobs` is the number of data passes ("epochs
              equivalent"), the natural budget unit for comparing minibatch methods.
"""
mutable struct WorkCounters
  forward::Int
  backward::Int
  jvp::Int
  hvp::Int
  samples::Int
end
WorkCounters() = WorkCounters(0, 0, 0, 0, 0)

# Parameters, state, data, minibatch sampler and first batch, shared by both model types.
function _lux_setup(model, data; rng, ps, st, device, batchsize, shuffle, partial)
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

  if batchsize === nothing
    sampler = nothing
    batch = data
    # Data without a single observation count (e.g. PINN interior and boundary points of
    # different sizes) count as one observation: one data pass = one full evaluation.
    nobs = max(_count(data), 1)
  elseif batchsize isa NamedTuple
    data isa NamedTuple || throw(ArgumentError("a NamedTuple `batchsize` needs NamedTuple data"))
    sampler = FieldSampler(data, batchsize; shuffle, partial, rng)
    batch = _draw!(sampler, data)
    nobs = _sampler_nobs(sampler, data)
  else
    _count(data) > 0 || throw(ArgumentError(
      "minibatch mode needs data with a single observation count (see `LuxNLPModels.numobs`); " *
      "for fields of different sizes pass a NamedTuple `batchsize`, e.g. (interior = 64, boundary = nothing)"))
    sampler = MinibatchSampler(_count(data), batchsize; shuffle, partial, rng)
    batch = _draw!(sampler, data)
    nobs = _sampler_nobs(sampler, data)
  end
  batch_nobs = sampler === nothing ? nobs : _batch_nobs(sampler, batch)
  return (; x0, ax, st, data, sampler, batch, nobs, batch_nobs)
end

function _unconstrained_meta(x0::S, name) where {S}
  T = eltype(x0)
  # No bound analysis: it calls `findall` on the bound vectors, which not every GPU array
  # type implements; `unconstrained(nlp)` then falls back to a GPU-friendly reduction.
  return NLPModelMeta{T, S}(length(x0); x0 = x0,
                            lvar = fill!(similar(x0), T(-Inf)),
                            uvar = fill!(similar(x0), T(Inf)),
                            variable_bounds_analysis = false,
                            name = name)
end

"""
    LuxNLPModel{T, S} <: AbstractNLPModel{T, S}

Unconstrained NLPModel `min_x f(x) = loss(model, ps(x), st, batch)` where `ps(x)` is the
`ComponentArray` view of the flat vector `x` and `batch` is the current batch (the full data
set in full-batch mode, a minibatch otherwise).

See [`LuxNLPModel(model, loss, data; kwargs...)`](@ref) for construction and
[`LuxNLSModel`](@ref) for the least-squares variant.
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
  nobs::Int          # observations per epoch (1 if not countable: one pass = one evaluation)
  batch_nobs::Int    # observations in `batch`
  sampler::Union{Nothing, AbstractBatchSampler}
  backend::AD
  hvp_backend::HAD
  grad_prep::Any
  hvp_prep::Any
  prep_shape::Any
  backend_cache::Any # used by non-DI backends (ReactantBackend)
  max_hess_nvar::Int # size guard for the explicit (dense) Hessian, see `hess_coord!`
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
- `batchsize = nothing`: `nothing` for full batch (deterministic objective); an integer for
  minibatches of the observations; a NamedTuple such as `(interior = 64, boundary = nothing)`
  to minibatch fields of different sizes separately (see `FieldSampler`). Batches are
  advanced with [`next_batch!`](@ref) or [`minibatch_next_train!`](@ref).
- `shuffle = true`, `partial = false`: minibatch sampling options.
- `backend = AutoZygote()`: AD backend (ADTypes) for gradients, or `ReactantBackend()`
  (requires `using Reactant, Enzyme`) to compile the objective and gradient with Reactant.
- `hvp_backend = SecondOrder(AutoForwardDiff(), backend)`: backend for `hprod`.
- `max_hess_nvar = 5_000`: largest number of parameters for which the explicit Hessian
  (`hess`, `hess_coord!`) may be formed. Not recommended for neural networks; see `hess_coord!`.
- `name = "LuxNLPModel"`.

# Notes
- The Lux state `st` is *not* updated by evaluations, so the objective is a deterministic
  function of `x` for a fixed batch (a fixed dropout mask, batch statistics for BatchNorm in
  training mode). Pass `st = LuxCore.testmode(st)` to evaluate in inference mode.
- The element type of the problem is the element type of the parameters (Float32 by default
  in Lux). Pass `ps = f64(ps)` and Float64 data for double precision.
- Work accounting ([`work`](@ref)) counts observations with [`numobs`](@ref). Data without a
  single observation count (e.g. PINN interior and boundary points of different sizes) are
  treated as one observation in full-batch mode, so one data pass is one full evaluation.
"""
function LuxNLPModel(model, loss, data;
                     rng::AbstractRNG = Random.default_rng(),
                     ps = nothing,
                     st = nothing,
                     device = identity,
                     batchsize = nothing,
                     shuffle::Bool = true,
                     partial::Bool = false,
                     backend = AutoZygote(),
                     hvp_backend = backend isa ReactantBackend ? nothing : DI.SecondOrder(AutoForwardDiff(), backend),
                     max_hess_nvar::Integer = 5_000,
                     name::AbstractString = "LuxNLPModel")
  _check_backend_loaded(backend)
  c = _lux_setup(model, data; rng, ps, st, device, batchsize, shuffle, partial)
  meta = _unconstrained_meta(c.x0, name)
  return LuxNLPModel(meta, Counters(), WorkCounters(), model, loss, c.ax, c.st, c.data,
                     c.batch, c.nobs, c.batch_nobs, c.sampler, backend, hvp_backend,
                     nothing, nothing, _shape(c.batch), nothing, Int(max_hess_nvar))
end

function Base.show(io::IO, nlp::LuxNLPModel)
  print(io, "LuxNLPModel: ", nlp.meta.nvar, " variables of type ", eltype(nlp.meta.x0),
        " in ", typeof(nlp.meta.x0).name.name, ", ", _mode_string(nlp))
end

_mode_string(nlp) = nlp.sampler === nothing ? "full batch" :
                    "minibatch ($(nlp.batch_nobs) observations per batch, epoch $(epoch(nlp)))"

_clear_preps!(nlp::LuxNLPModel) =
  (nlp.grad_prep = nothing; nlp.hvp_prep = nothing; nlp.backend_cache = nothing; nlp)

"""
    ReactantBackend()

Backend marker: compile the objective / residual and their derivatives with
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and differentiate them with Enzyme
(forward mode for `J(x)v`, reverse mode for `J(x)ᵀw` and gradients). Requires
`using Reactant, Enzyme` (package extension). Reactant supports nested AD of any order, so
residuals may differentiate the network with respect to its inputs with `Enzyme.gradient`
(as in Lux's PINN example). Compiled functions are cached per batch shape; the first call
for a new shape compiles (seconds to minutes). The flat parameter vector stays on the host
and is copied to Reactant's device at each call. `hprod!` is not available.
"""
struct ReactantBackend end

function _check_backend_loaded(backend)
  if backend isa ReactantBackend &&
     Base.get_extension(@__MODULE__, :LuxNLPModelsReactantExt) === nothing
    throw(ArgumentError("ReactantBackend requires `using Reactant, Enzyme` before building the model"))
  end
  return nothing
end

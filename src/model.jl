abstract type AbstractLuxNLPModel{T, S} <: AbstractNLPModel{T, S} end

"""
    LuxNLPModel(model, data_train, data_test = nothing; kwargs...)

NLP whose variables are the parameters of the Lux layer `model` and whose objective is
`loss_f` on the current minibatch of `data_train`.

`data_train`, `data_test`: an array, or a Tuple / NamedTuple of arrays, with observations
along the last dimension (any MLUtils data container whose batches are such arrays works).
Floating-point arrays must already have element type `T`.

Keywords:
- `loss_f = CrossEntropyLoss(; logits = Val(true))`: any function with Lux's training
  signature `loss_f(model, ps, st, batch) -> (loss, st, stats)`; Lux's losses qualify.
- `batchsize = 128`, `shuffle = true`: minibatches have a fixed size; the last incomplete
  batch of each pass is dropped. Use `batchsize = numobs(data_train)` for full-batch solvers.
- `T = Float32`: `Float32` or `Float64`; parameters and layer state are converted to `T`.
- `device = cpu_device()`: or `gpu_device()`.
- `seed = 0`: parameter initialization, batch order and random layers.
- `grad_backend = AutoZygote()`: DifferentiationInterface backend for gradients.
- `hvp_backend`: second-order DifferentiationInterface backend, or `:finite_difference`
  (two gradients per product). Default: forward-over-reverse on CPU, finite differences on GPU.

Every evaluation uses the current minibatch and the current layer state; both change only in
`minibatch_next_train!`, `reset_minibatch_train!` and `set_data_train!`. Layers with
randomness (Dropout) therefore use the same masks for all evaluations on one minibatch, and
`obj`, `grad!` and `hprod!` describe one function (training mode) between two batch switches.
"""
mutable struct LuxNLPModel{T, S, M, AX, ST, BD, F, D, A} <: AbstractLuxNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  model::M
  ax::AX            # ComponentArray axes: view the flat vector as structured parameters
  st::ST            # layer state used by every evaluation (training mode)
  batch::BD         # current minibatch on the device; never aliases the loader's buffer
  loss_f::F
  device::D
  ad::A             # (; grad_backend, prep_grad, hvp_backend, prep_hvp)
  rng::Xoshiro      # reseeds random layers at each batch switch
  w1::S             # workspaces for finite-difference products
  w2::S
  w3::S
  # data and iteration: used once per batch switch, not on the evaluation path
  data_train::Any
  data_test::Any
  loader::Any
  loader_state::Any
  loader_opts::NamedTuple
  epoch::Int
end

function LuxNLPModel(
  model,
  data_train,
  data_test = nothing;
  loss_f = CrossEntropyLoss(; logits = Val(true)),
  batchsize::Int = 128,
  shuffle::Bool = true,
  T::Type{<:Union{Float32, Float64}} = Float32,
  device = cpu_device(),
  seed::Int = 0,
  grad_backend = AutoZygote(),
  hvp_backend = device isa CPUDevice ? SecondOrder(AutoForwardDiff(), AutoZygote()) : :finite_difference,
  name::String = "LuxNLPModel",
)
  _check_eltype(T, data_train)
  data_test === nothing || _check_eltype(T, data_test)
  n = numobs(data_train)
  1 <= batchsize <= n || throw(ArgumentError("batchsize must be in 1:$n"))

  rng = Xoshiro(seed)
  ps, st = Lux.setup(rng, model)
  ps_ca = ComponentArray(_to_precision(T, ps))
  ax = getaxes(ps_ca)
  x0 = device(collect(getdata(ps_ca)))
  st = device(_to_precision(T, st))      # BatchNorm statistics are Float32 by default in Lux

  loader_opts = (; batchsize, shuffle, rng = Xoshiro(rand(rng, UInt64)))
  loader = DataLoader(data_train; loader_opts..., partial = false, buffer = true)
  host, loader_state = iterate(loader)
  batch = device(_copy_arrays(host))

  meta = NLPModelMeta(length(x0); x0 = x0, name = name, nnzh = 0, hess_available = false)
  ctx = _contexts(model, ax, st, batch, loss_f)
  prep_grad = prepare_gradient(_loss, grad_backend, x0, ctx...)
  prep_hvp = hvp_backend === :finite_difference ? nothing :
             prepare_hvp(_loss, hvp_backend, x0, (zero(x0),), ctx...)
  ad = (; grad_backend, prep_grad, hvp_backend, prep_hvp)

  return LuxNLPModel(meta, Counters(), model, ax, st, batch, loss_f, device, ad, rng,
                     similar(x0), similar(x0), similar(x0),
                     data_train, data_test, loader, loader_state, loader_opts, 0)
end

_to_precision(::Type{Float32}, x) = f32(x)
_to_precision(::Type{Float64}, x) = f64(x)

# Converting once, before building the model, avoids a conversion at every batch copy.
_check_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} =
  eltype(A) === T ||
  throw(ArgumentError("data has eltype $(eltype(A)) but T = $T; convert the data first"))
_check_eltype(::Type{T}, t::Union{Tuple, NamedTuple}) where {T} = foreach(a -> _check_eltype(T, a), t)
_check_eltype(::Type, _) = nothing

_copy_arrays(A::AbstractArray) = copy(A)
_copy_arrays(t::Union{Tuple, NamedTuple}) = map(_copy_arrays, t)

"Structured parameters at `x`: a ComponentArray view of the flat vector, no copy."
params_view(nlp::LuxNLPModel, x::AbstractVector) = ComponentArray(x, nlp.ax...)

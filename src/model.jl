"""
    LuxNLPModel(model, data_train; kwargs...)

NLP whose variables are the parameters of the Lux layer `model` and whose objective is
`loss_f` on the current minibatch of `data_train`.

`data_train`: an array, or a Tuple / NamedTuple of arrays such as `(X, Y)`, with observations
along the last dimension. Floating-point arrays are converted to `T`.

Keywords:
- `loss_f = CrossEntropyLoss(; logits = Val(true))`: any function with Lux's training
  signature `loss_f(model, ps, st, batch) -> (loss, st, stats)`; Lux's losses qualify.
- `batchsize = 128`, `shuffle = true`: the last incomplete minibatch of each pass is dropped.
  Use `batchsize = numobs(data_train)` for a deterministic, full-batch problem; change it later
  with `set_batchsize!`.
- `T = Float32`: `Float32` or `Float64`.
- `device = cpu_device()`: or `gpu_device()`.
- `seed = 0`: parameter initialization, minibatch order and Dropout masks.
- `grad_backend = AutoZygote()`: DifferentiationInterface backend for gradients.
- `hvp_backend = nothing`: `hprod!` is then a finite difference of two gradients. For exact
  products pass a DifferentiationInterface second-order backend, e.g.
  `SecondOrder(AutoForwardDiff(), AutoZygote())` after `using ForwardDiff`.

Every evaluation uses the current minibatch, which changes only in `minibatch_next_train!` and
`set_batchsize!`. Between two switches, `obj`, `grad!` and `hprod!` therefore describe one
function: same minibatch, same Dropout masks, training-mode normalization.
"""
mutable struct LuxNLPModel{T, S, M, AX, ST, B, F, D, G, P, H, Q} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  model::M
  ax::AX            # ComponentArray axes: view the flat vector as structured parameters
  st::ST            # layer state used by every evaluation (training mode)
  batch::B          # current minibatch, on the device
  loss_f::F
  device::D
  grad_backend::G
  prep::P           # gradient preparation, redone only when the batch size changes
  hvp_backend::H    # nothing: finite-difference Hessian-vector products
  hvp_prep::Q
  rng::Xoshiro      # reseeds random layers at each minibatch switch
  w::S              # finite-difference workspaces, empty for an exact hvp_backend
  gm::S
  # data and iteration: used once per minibatch switch, not on the evaluation path
  data_train::Any
  loader::Any
  loader_state::Any
  shuffle::Bool
  epoch::Int
end

function LuxNLPModel(
  model,
  data_train;
  loss_f = CrossEntropyLoss(; logits = Val(true)),
  batchsize::Int = 128,
  shuffle::Bool = true,
  T::Type{<:Union{Float32, Float64}} = Float32,
  device = cpu_device(),
  seed::Int = 0,
  grad_backend = AutoZygote(),
  hvp_backend = nothing,
)
  data_train = _convert(T, data_train)

  rng = Xoshiro(seed)
  ps, st = Lux.setup(rng, model)
  to_T = T === Float32 ? f32 : f64
  ps_ca = ComponentArray(to_T(ps))
  ax = getaxes(ps_ca)
  x0 = device(collect(getdata(ps_ca)))
  st = device(to_T(st))                  # BatchNorm statistics are Float32 by default in Lux

  loader = _loader(data_train, batchsize, shuffle, rng)
  host, loader_state = iterate(loader)
  batch = device(host)

  meta = NLPModelMeta(length(x0); x0 = x0, name = "LuxNLPModel", nnzh = 0,
                      hess_available = false)
  prep, hvp_prep = _prepare(grad_backend, hvp_backend, x0, _contexts(model, ax, st, batch, loss_f))
  nw = hvp_backend === nothing ? length(x0) : 0

  return LuxNLPModel(meta, Counters(), model, ax, st, batch, loss_f, device, grad_backend, prep,
                     hvp_backend, hvp_prep, rng, similar(x0, nw), similar(x0, nw), data_train,
                     loader, loader_state, shuffle, 0)
end

# Floating-point data are converted to T once, here, rather than at every minibatch.
_convert(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} = eltype(A) === T ? A : T.(A)
_convert(::Type{T}, t::Union{Tuple, NamedTuple}) where {T} = map(a -> _convert(T, a), t)
_convert(::Type, data) = data

function _loader(data, batchsize, shuffle, rng)
  n = numobs(data)
  1 <= batchsize <= n || throw(ArgumentError("batchsize must be in 1:$n"))
  return DataLoader(data; batchsize, shuffle, partial = false, rng = Xoshiro(rand(rng, UInt64)))
end

"Structured parameters at `x`: a ComponentArray view of the flat vector, no copy."
params_view(nlp::LuxNLPModel, x::AbstractVector) = ComponentArray(x, nlp.ax...)

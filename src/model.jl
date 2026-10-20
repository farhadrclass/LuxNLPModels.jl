"""
    PassCounts

Forward and backward passes through the network made by the NLPModels API (evaluation code is
not counted). Unlike `Counters`, these measure cost: `objgrad!` is one forward and one backward
pass, a finite-difference `hprod!` two of each. Forward-over-reverse products are counted in
`hvp_ad`, because their cost is not a whole number of passes.
"""
mutable struct PassCounts
  forward::Int
  backward::Int
  hvp_ad::Int
end
PassCounts() = PassCounts(0, 0, 0)

"""
    LuxNLPModel(model, source; objective, seed, T, device, grad_backend, hvp_backend, name)

- `model`: a Lux layer.
- `source`: an `AbstractBatchSource`, e.g. `ArrayBatchSource((X, Y))`.
- `objective`: Lux-signature objective (default `classification_objective`).
- `seed`: initializes the parameters and the reseeding of stochastic layers.
- `T`: `Float32` or `Float64`; parameters *and* layer state are converted to `T`, and the data
  must already have element type `T`.
- `device`: `cpu_device()` or `gpu_device()`.
- `grad_backend`: DifferentiationInterface backend for gradients (default `AutoZygote()`).
- `hvp_backend`: second-order backend, or `:finite_difference` (two gradients per product).
  On GPU use `:finite_difference` until forward-over-reverse has been tested on your hardware.
  Change it later with `set_hvp_backend!`.

The Hessian is never formed: `nnzh = 0`, `hess_available = false`, `hprod_available = true`.
"""
mutable struct LuxNLPModel{T, S, M, AX, ST, B, BD, O, D, G} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  passes::PassCounts
  model::M
  ax::AX              # ComponentArray axes (tuple), to view the flat vector as structured params
  st::ST              # committed layer state
  st_last::ST         # state returned by the last `obj` call
  source::B
  batch::BD           # device buffer holding the current batch
  objective::O
  device::D
  grad_backend::G
  hvp_backend::Any    # backend or :finite_difference; untyped so it can be switched
  prep_grad::Any      # prepared gradient (re-prepared when the batch size changes)
  prep_hvp::Any       # prepared HVP, created on first use
  rng::Xoshiro        # reseeds stochastic layers at every batch switch
  w1::S               # workspaces for finite-difference HVPs
  w2::S
  w3::S
end

function LuxNLPModel(
  model,
  source::AbstractBatchSource;
  objective = classification_objective,
  seed::Int = 0,
  T::Type{<:Union{Float32, Float64}} = Float32,
  device = cpu_device(),
  grad_backend = AutoZygote(),
  hvp_backend = SecondOrder(AutoForwardDiff(), AutoZygote()),
  name::String = "LuxNLPModel",
)
  _check_eltype(T, getbatch(source))
  rng = Xoshiro(seed)
  ps, st = Lux.setup(rng, model)
  ps_ca = ComponentArray(_to_precision(T, ps))
  ax = getaxes(ps_ca)
  x0 = device(collect(getdata(ps_ca)))
  st = device(_to_precision(T, st))     # BatchNorm statistics are Float32 by default in Lux
  batch = device(getbatch(source))      # on CPU this may alias the host buffer, which is fine
  meta = NLPModelMeta(length(x0); x0 = x0, name = name, nnzh = 0, hess_available = false)
  nlp = LuxNLPModel(meta, Counters(), PassCounts(), model, ax, st, st, source, batch, objective,
                    device, grad_backend, hvp_backend, nothing, nothing, rng,
                    similar(x0), similar(x0), similar(x0))
  _prepare_grad!(nlp)
  return nlp
end

_to_precision(::Type{Float32}, x) = f32(x)
_to_precision(::Type{Float64}, x) = f64(x)

# Floating-point data must already have the variable type T (convert once, before building the
# source). Otherwise every batch copy would convert, and on GPU a host-to-device copy between
# different element types is not guaranteed to work.
_check_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} =
  eltype(A) === T ||
  throw(ArgumentError("data has eltype $(eltype(A)) but T = $T; convert the data first"))
_check_eltype(::Type{T}, A::AbstractArray) where {T} = nothing          # integer / Bool data
_check_eltype(::Type{T}, t::Union{Tuple, NamedTuple}) where {T} =
  foreach(a -> _check_eltype(T, a), t)

"Flat vector -> structured parameters (a view, no copy)."
params_view(nlp::LuxNLPModel, x::AbstractVector) = ComponentArray(x, nlp.ax...)

"Forward/backward pass counts since the last `reset!`."
pass_counts(nlp::LuxNLPModel) = nlp.passes

function _count!(nlp::LuxNLPModel, nforward::Int, nbackward::Int)
  nlp.passes.forward += nforward
  nlp.passes.backward += nbackward
  return nlp
end

"Reset the NLPModels counters and the pass counts."
function NLPModels.reset!(nlp::LuxNLPModel)
  reset!(nlp.counters)
  p = nlp.passes
  p.forward = 0
  p.backward = 0
  p.hvp_ad = 0
  return nlp
end

# NLPModels API on the current minibatch, through DifferentiationInterface preparations made
# in the constructor. Costs: obj = 1 forward pass; grad! and objgrad! = 1 forward + 1 backward;
# hprod! = 2 gradients (finite differences) or 1 forward-over-reverse pass (hvp_backend).

# TODO: measure whether ComponentArrays makes gradients allocate in proportion to depth. Its
# Zygote rule for `getproperty` allocates a full-size zero array per field access, and `Chain`
# reads `ps.layer_i` for every layer, so one gradient may cost about 2 × depth × n floats.
# Check with depth = 2, 8, 32 (on GPU use `CUDA.@time` instead of `@allocated`):
#   model = Chain([Dense(256 => 256, relu) for _ in 1:depth]..., Dense(256 => 10))
#   X = randn(Float32, 256, 1024); Y = Float32.((1:10) .== rand(1:10, 1, 1024))
#   nlp = LuxNLPModel(model, (X, Y); batchsize = 128); x = nlp.meta.x0; g = similar(x)
#   grad!(nlp, x, g); @allocated(grad!(nlp, x, g)) / sizeof(x)
# Ratio constant in depth: keep ComponentArrays. Ratio growing with depth: build parameters
# with `x0, re = Optimisers.destructure(ps)` and use `re(x)` below (one flat gradient), or
# use `grad_backend = AutoEnzyme()`, which does not go through that rule.
_loss(x, model, ax, st, batch, loss_f) = first(loss_f(model, ComponentArray(x, ax...), st, batch))

_contexts(model, ax, st, batch, loss_f) =
  (Constant(model), Constant(ax), Constant(st), Constant(batch), Constant(loss_f))
_contexts(nlp::LuxNLPModel) = _contexts(nlp.model, nlp.ax, nlp.st, nlp.batch, nlp.loss_f)

function _prepare(grad_backend, hvp_backend, x, contexts)
  prep = prepare_gradient(_loss, grad_backend, x, contexts...)
  hvp_prep = hvp_backend === nothing ? nothing :
             prepare_hvp(_loss, hvp_backend, x, (zero(x),), contexts...)
  return prep, hvp_prep
end

_gradient!(nlp::LuxNLPModel, g, x) =
  gradient!(_loss, g, nlp.prep, nlp.grad_backend, x, _contexts(nlp)...)

# The gradient is prepared for vectors of type S; say so instead of failing inside DI.
_check_type(::LuxNLPModel{<:Any, S}, x) where {S} =
  x isa S || throw(ArgumentError("expected a vector of type $S, got $(typeof(x)); " *
                                 "the precision and device are fixed at construction"))

function NLPModels.obj(nlp::LuxNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  _check_type(nlp, x)
  increment!(nlp, :neval_obj)
  return _loss(x, nlp.model, nlp.ax, nlp.st, nlp.batch, nlp.loss_f)
end

function NLPModels.grad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  _check_type(nlp, x)
  increment!(nlp, :neval_grad)
  return _gradient!(nlp, g, x)
end

"One forward and one backward pass (the NLPModels fallback would make two forward passes)."
function NLPModels.objgrad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  _check_type(nlp, x)
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  fx, _ = value_and_gradient!(_loss, g, nlp.prep, nlp.grad_backend, x, _contexts(nlp)...)
  return fx, g
end

function NLPModels.hprod!(nlp::LuxNLPModel{T}, x::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight = one(T)) where {T}
  @lencheck nlp.meta.nvar x v Hv
  _check_type(nlp, x)
  _check_type(nlp, v)
  increment!(nlp, :neval_hprod)
  σ = T(obj_weight)                     # a Float64 weight would run GPU kernels in Float64
  iszero(σ) && return fill!(Hv, zero(T))
  _hvp!(nlp, nlp.hvp_backend, x, v, Hv)
  isone(σ) || (Hv .*= σ)
  return Hv
end

# Lagrangian form required by the NLPModels API; without constraints it is the objective HVP.
function NLPModels.hprod!(nlp::LuxNLPModel{T}, x::AbstractVector, y::AbstractVector,
                          v::AbstractVector, Hv::AbstractVector; obj_weight = one(T)) where {T}
  @lencheck nlp.meta.ncon y
  return hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

# Central difference of the gradient along v, O(h²) truncation error; h ∝ cbrt(eps)(1 + ‖x‖)/‖v‖
# balances truncation and rounding.
function _hvp!(nlp::LuxNLPModel{T}, ::Nothing, x, v, Hv) where {T}
  nv = norm(v)
  iszero(nv) && return fill!(Hv, zero(T))
  h = cbrt(eps(T)) * (one(T) + norm(x)) / nv
  w, gm = nlp.w, nlp.gm
  @. w = x + h * v
  _gradient!(nlp, Hv, w)
  @. w = x - h * v
  _gradient!(nlp, gm, w)
  @. Hv = (Hv - gm) / (2h)
  return Hv
end

# Exact product from a DifferentiationInterface second-order backend.
_hvp!(nlp::LuxNLPModel, backend, x, v, Hv) =
  hvp!(_loss, (Hv,), nlp.hvp_prep, backend, x, (v,), _contexts(nlp)...)

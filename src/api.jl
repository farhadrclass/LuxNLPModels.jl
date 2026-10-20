# NLPModels API on the current minibatch, through DifferentiationInterface operators prepared
# once in the constructor. Costs: obj = 1 forward pass; grad! and objgrad! = 1 forward +
# 1 backward; hprod! = 1 forward-over-reverse product, or 2 gradients with :finite_difference.

_loss(x, model, ax, st, batch, loss_f) = first(loss_f(model, ComponentArray(x, ax...), st, batch))

_contexts(model, ax, st, batch, loss_f) =
  (Constant(model), Constant(ax), Constant(st), Constant(batch), Constant(loss_f))
_contexts(nlp::LuxNLPModel) = _contexts(nlp.model, nlp.ax, nlp.st, nlp.batch, nlp.loss_f)

# The operators are prepared for vectors of type S; say so instead of failing inside DI.
_check_type(nlp::LuxNLPModel{T, S}, x) where {T, S} =
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
  gradient!(_loss, g, nlp.ad.prep_grad, nlp.ad.grad_backend, x, _contexts(nlp)...)
  return g
end

"One forward and one backward pass (the NLPModels fallback would make two forward passes)."
function NLPModels.objgrad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  _check_type(nlp, x)
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  fx, _ = value_and_gradient!(_loss, g, nlp.ad.prep_grad, nlp.ad.grad_backend, x, _contexts(nlp)...)
  return fx, g
end

function NLPModels.hprod!(nlp::LuxNLPModel{T}, x::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight = one(T)) where {T}
  @lencheck nlp.meta.nvar x v Hv
  _check_type(nlp, x)
  _check_type(nlp, v)
  increment!(nlp, :neval_hprod)
  if nlp.ad.hvp_backend === :finite_difference
    _fd_hprod!(nlp, x, v, Hv)
  else
    hvp!(_loss, (Hv,), nlp.ad.prep_hvp, nlp.ad.hvp_backend, x, (v,), _contexts(nlp)...)
  end
  Hv .*= obj_weight
  return Hv
end

# Central difference of the gradient along v, O(h²) truncation error; h ∝ cbrt(eps)(1 + ‖x‖)/‖v‖
# balances truncation and rounding.
function _fd_hprod!(nlp::LuxNLPModel{T}, x, v, Hv) where {T}
  nv = norm(v)
  iszero(nv) && return fill!(Hv, zero(T))
  h = cbrt(eps(T)) * (one(T) + norm(x)) / nv
  w, gp, gm = nlp.w1, nlp.w2, nlp.w3
  @. w = x + h * v
  gradient!(_loss, gp, nlp.ad.prep_grad, nlp.ad.grad_backend, w, _contexts(nlp)...)
  @. w = x - h * v
  gradient!(_loss, gm, nlp.ad.prep_grad, nlp.ad.grad_backend, w, _contexts(nlp)...)
  @. Hv = (gp - gm) / (2h)
  return Hv
end

# Lagrangian form required by the NLPModels API; without constraints it is the objective HVP.
function NLPModels.hprod!(nlp::LuxNLPModel{T}, x::AbstractVector, y::AbstractVector,
                          v::AbstractVector, Hv::AbstractVector; obj_weight = one(T)) where {T}
  @lencheck nlp.meta.ncon y
  return hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

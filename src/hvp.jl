# Second-order API: Hessian-vector products of the objective on the current batch.
#
# Extension points for second-order solvers (not implemented):
# - Gauss-Newton / Fisher products (positive semidefinite, often preferred for networks);
# - products on a sub-batch of the current batch (subsampled Newton): the contract still holds
#   as long as the sub-batch is fixed between two accepted steps.

"""
    set_hvp_backend!(nlp, backend)

`backend`: a DifferentiationInterface second-order backend, e.g.
`SecondOrder(AutoForwardDiff(), AutoZygote())`, or `:finite_difference`.
"""
function set_hvp_backend!(nlp::LuxNLPModel, backend)
  nlp.hvp_backend = backend
  nlp.prep_hvp = nothing
  return nlp
end

function _prepare_hvp!(nlp::LuxNLPModel)
  t = fill!(nlp.w1, zero(eltype(nlp.w1)))          # tangent used only for its type and size
  nlp.prep_hvp = prepare_hvp(_loss, nlp.hvp_backend, nlp.meta.x0, (t,), _contexts(nlp)...)
  return nlp
end

function NLPModels.hprod!(
  nlp::LuxNLPModel{T},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)
  if nlp.hvp_backend === :finite_difference
    _fd_hprod!(nlp, x, v, Hv)
  else
    nlp.prep_hvp === nothing && _prepare_hvp!(nlp)
    nlp.passes.hvp_ad += 1
    hvp!(_loss, (Hv,), nlp.prep_hvp, nlp.hvp_backend, x, (v,), _contexts(nlp)...)
  end
  Hv .*= obj_weight
  return Hv
end

# Central difference of the gradient along v: two gradients, O(h²) truncation error.
# h ∝ cbrt(eps) (1 + ‖x‖) / ‖v‖ balances truncation and rounding error for a central difference.
function _fd_hprod!(nlp::LuxNLPModel{T}, x, v, Hv) where {T}
  nv = norm(v)
  if iszero(nv)
    fill!(Hv, zero(T))
    return Hv
  end
  h = cbrt(eps(T)) * (one(T) + norm(x)) / nv
  w, gp, gm = nlp.w1, nlp.w2, nlp.w3
  @. w = x + h * v
  gradient!(_loss, gp, nlp.prep_grad, nlp.grad_backend, w, _contexts(nlp)...)
  @. w = x - h * v
  gradient!(_loss, gm, nlp.prep_grad, nlp.grad_backend, w, _contexts(nlp)...)
  @. Hv = (gp - gm) / (2h)
  _count!(nlp, 2, 2)
  return Hv
end

# Lagrangian form required by the NLPModels API; with no constraints it is the objective HVP.
function NLPModels.hprod!(
  nlp::LuxNLPModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  @lencheck nlp.meta.ncon y
  return hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

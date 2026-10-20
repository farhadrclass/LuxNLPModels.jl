# First-order NLPModels API on the current batch. Derivatives use DifferentiationInterface with
# operators prepared once and reused while the batch size is unchanged (DI 0.7 signatures
# `gradient!(f, g, prep, backend, x, contexts...)`, `value_and_gradient!`, `Constant`).
#
# Cost: obj = 1 forward; grad! and objgrad! = 1 forward + 1 backward (see `pass_counts`).
# AD allocates; these methods are not allocation-free, the solver side can be.

_loss(x, model, ax, st, batch, objective) =
  first(objective(model, ComponentArray(x, ax...), st, batch))

_contexts(nlp::LuxNLPModel) = (Constant(nlp.model), Constant(nlp.ax), Constant(nlp.st),
                               Constant(nlp.batch), Constant(nlp.objective))

function _prepare_grad!(nlp::LuxNLPModel)
  nlp.prep_grad = prepare_gradient(_loss, nlp.grad_backend, nlp.meta.x0, _contexts(nlp)...)
  return nlp
end

"Forward pass. Records the returned layer state in `st_last` (not committed)."
function NLPModels.obj(nlp::LuxNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  _count!(nlp, 1, 0)
  fx, st_new, _ = nlp.objective(nlp.model, params_view(nlp, x), nlp.st, nlp.batch)
  nlp.st_last = st_new
  return fx
end

function NLPModels.grad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  _count!(nlp, 1, 1)
  gradient!(_loss, g, nlp.prep_grad, nlp.grad_backend, x, _contexts(nlp)...)
  return g
end

"One forward and one backward pass (the NLPModels fallback would do two forwards)."
function NLPModels.objgrad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  _count!(nlp, 1, 1)
  fx, _ = value_and_gradient!(_loss, g, nlp.prep_grad, nlp.grad_backend, x, _contexts(nlp)...)
  return fx, g
end

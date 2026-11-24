"""
    f = local_loss(nlp, ps, batch)

Evaluate the loss of `nlp` on `batch` with the network parameters `ps`, a `ComponentArray`.
The default method computes `nlp.loss_f(ŷ, y)` for a batch `(x, y)`.
Add a method for another batch type to define another objective.
"""
function local_loss(nlp::LuxNLPModel, ps, (x, y)::Tuple)
  ŷ, _ = nlp.model(x, ps, nlp.st)
  return nlp.loss_f(ŷ, y)
end

loss_w(nlp, w, batch) = local_loss(nlp, ComponentArray(w, nlp.ps_axes), batch)
gradient_loss_w(nlp, w, batch) = Zygote.gradient(w -> loss_w(nlp, w, batch), w)[1]

"""
    f = obj(nlp, w)

Evaluate `f(w)`, the loss on the current training minibatch.
"""
function NLPModels.obj(nlp::AbstractLuxNLPModel, w::AbstractVector)
  @lencheck nlp.meta.nvar w
  increment!(nlp, :neval_obj)
  return loss_w(nlp, w, nlp.current_training_minibatch)
end

"""
    g = grad!(nlp, w, g)

Evaluate `∇f(w)` in place, with Zygote.
"""
function NLPModels.grad!(nlp::AbstractLuxNLPModel, w::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_grad)
  g .= gradient_loss_w(nlp, w, nlp.current_training_minibatch)
  return g
end

"""
    f, g = objgrad!(nlp, w, g)

Evaluate `f(w)` and `∇f(w)` in a single forward and reverse pass.
"""
function NLPModels.objgrad!(nlp::AbstractLuxNLPModel, w::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  batch = nlp.current_training_minibatch
  f, ∇f = Zygote.withgradient(w -> loss_w(nlp, w, batch), w)
  g .= ∇f[1]
  return f, g
end

"""
    Hv = hprod!(nlp, w, v, Hv; obj_weight = 1)

Evaluate `∇²f(w) v` in place, with ForwardDiff over Zygote.
"""
function NLPModels.hprod!(
  nlp::AbstractLuxNLPModel,
  w::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(w)),
)
  @lencheck nlp.meta.nvar w v Hv
  increment!(nlp, :neval_hprod)
  batch = nlp.current_training_minibatch
  Hv .= ForwardDiff.derivative(t -> gradient_loss_w(nlp, w .+ t .* v, batch), zero(eltype(w)))
  Hv .*= obj_weight
  return Hv
end

"""
    rows, cols = hess_structure!(nlp, rows, cols)

Structure of the lower triangle of the Hessian, which is dense for a neural network.
"""
function NLPModels.hess_structure!(
  nlp::AbstractLuxNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
  n = nlp.meta.nvar
  k = 0
  for j = 1:n, i = j:n
    k += 1
    rows[k] = i
    cols[k] = j
  end
  return rows, cols
end

"""
    vals = hess_coord!(nlp, w, y, vals; obj_weight = 1)

Lower triangle of the dense Hessian, by forward differentiation of the gradient.
Not recommended beyond small networks: it costs `n(n+1)/2` storage and about `n`
Hessian-vector products.
"""
function NLPModels.hess_coord!(
  nlp::AbstractLuxNLPModel,
  w::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(w)),
)
  @lencheck nlp.meta.nvar w
  @lencheck 0 y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  batch = nlp.current_training_minibatch
  H = ForwardDiff.jacobian(w -> gradient_loss_w(nlp, w, batch), w)
  n = nlp.meta.nvar
  k = 1
  for j = 1:n
    copyto!(vals, k, H, (j - 1) * n + j, n - j + 1)
    k += n - j + 1
  end
  vals .*= obj_weight
  return vals
end

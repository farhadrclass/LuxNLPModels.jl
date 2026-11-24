"""
    F = local_residual(nlp, ps, batch)

Evaluate the residual vector of `nlp` on `batch` with the network parameters `ps`,
a `ComponentArray`.
The default method computes `vec(nlp.residual_f(ŷ, y))` for a batch `(x, y)`.
Add a method for another batch type to define another residual.
"""
function local_residual(nlp::LuxNLSModel, ps, (x, y)::Tuple)
  ŷ, _ = nlp.model(x, ps, nlp.st)
  return vec(nlp.residual_f(ŷ, y))
end

residual_w(nlp, w, batch) = local_residual(nlp, ComponentArray(w, nlp.ps_axes), batch)

"""
    Fx = residual!(nlp, w, Fx)

Evaluate the residual on the current training minibatch.
"""
function NLPModels.residual!(nlp::AbstractLuxNLSModel, w::AbstractVector, Fx::AbstractVector)
  @lencheck nlp.meta.nvar w
  @lencheck nlp.nls_meta.nequ Fx
  increment!(nlp, :neval_residual)
  Fx .= residual_w(nlp, w, nlp.current_training_minibatch)
  return Fx
end

"""
    Jv = jprod_residual!(nlp, w, v, Jv)

Evaluate `J(w) v`, with ForwardDiff.
"""
function NLPModels.jprod_residual!(
  nlp::AbstractLuxNLSModel,
  w::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nvar w v
  @lencheck nlp.nls_meta.nequ Jv
  increment!(nlp, :neval_jprod_residual)
  batch = nlp.current_training_minibatch
  Jv .= ForwardDiff.derivative(t -> residual_w(nlp, w .+ t .* v, batch), zero(eltype(w)))
  return Jv
end

"""
    Jtv = jtprod_residual!(nlp, w, v, Jtv)

Evaluate `J(w)ᵀ v`, with Zygote.
"""
function NLPModels.jtprod_residual!(
  nlp::AbstractLuxNLSModel,
  w::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar w Jtv
  @lencheck nlp.nls_meta.nequ v
  increment!(nlp, :neval_jtprod_residual)
  batch = nlp.current_training_minibatch
  _, back = Zygote.pullback(w -> residual_w(nlp, w, batch), w)
  Jtv .= back(v)[1]
  return Jtv
end

"""
    rows, cols = jac_structure_residual!(nlp, rows, cols)

Structure of the residual Jacobian, which is dense for a neural network (column-major order).
"""
function NLPModels.jac_structure_residual!(
  nlp::AbstractLuxNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.nls_meta.nnzj rows cols
  m, n = nlp.nls_meta.nequ, nlp.meta.nvar
  k = 0
  for j = 1:n, i = 1:m
    k += 1
    rows[k] = i
    cols[k] = j
  end
  return rows, cols
end

"""
    vals = jac_coord_residual!(nlp, w, vals)

Entries of the dense residual Jacobian, with ForwardDiff.
Not recommended beyond small networks: prefer subsolvers that only use `J v` and `Jᵀ v`.
"""
function NLPModels.jac_coord_residual!(
  nlp::AbstractLuxNLSModel,
  w::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nlp.meta.nvar w
  @lencheck nlp.nls_meta.nnzj vals
  increment!(nlp, :neval_jac_residual)
  batch = nlp.current_training_minibatch
  vals .= vec(ForwardDiff.jacobian(w -> residual_w(nlp, w, batch), w))
  return vals
end

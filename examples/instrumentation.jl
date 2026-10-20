# Research instrumentation, kept out of the package.
#
# PassCounter wraps a LuxNLPModel and counts forward and backward passes through the network.
# NLPModels' Counters count oracle calls, which is not cost: `objgrad!` increments both
# `neval_obj` and `neval_grad` but costs one forward and one backward pass, and a
# finite-difference `hprod!` counts once but costs two gradients. Solvers get the wrapper; the
# counters object is shared with the wrapped model, so `neval_*` still work.
using NLPModels, Lux, LuxNLPModels
using MLUtils: DataLoader

mutable struct PassCounter{T, S, M <: LuxNLPModel{T, S}} <: AbstractNLPModel{T, S}
  nlp::M
  meta::NLPModelMeta{T, S}
  counters::Counters
  forward::Int
  backward::Int
  hvp_ad::Int          # forward-over-reverse products: cost not a whole number of passes
end
PassCounter(nlp::LuxNLPModel) = PassCounter(nlp, nlp.meta, nlp.counters, 0, 0, 0)

function NLPModels.obj(m::PassCounter, x::AbstractVector)
  m.forward += 1
  return obj(m.nlp, x)
end

function NLPModels.grad!(m::PassCounter, x::AbstractVector, g::AbstractVector)
  m.forward += 1
  m.backward += 1
  return grad!(m.nlp, x, g)
end

function NLPModels.objgrad!(m::PassCounter, x::AbstractVector, g::AbstractVector)
  m.forward += 1
  m.backward += 1
  return objgrad!(m.nlp, x, g)
end

function NLPModels.hprod!(m::PassCounter{T}, x::AbstractVector, v::AbstractVector, Hv::AbstractVector;
                          obj_weight = one(T)) where {T}
  if m.nlp.ad.hvp_backend === :finite_difference
    m.forward += 2
    m.backward += 2
  else
    m.hvp_ad += 1
  end
  return hprod!(m.nlp, x, v, Hv; obj_weight = obj_weight)
end

function NLPModels.hprod!(m::PassCounter{T}, x::AbstractVector, y::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight = one(T)) where {T}
  @lencheck m.meta.ncon y
  return hprod!(m, x, v, Hv; obj_weight = obj_weight)
end

function NLPModels.reset!(m::PassCounter)
  NLPModels.reset!(m.counters)
  m.forward = m.backward = m.hvp_ad = 0
  return m
end

# minibatch control passes through, so callbacks and on_accept! hooks work on the wrapper
LuxNLPModels.minibatch_next_train!(m::PassCounter) = minibatch_next_train!(m.nlp)
LuxNLPModels.reset_minibatch_train!(m::PassCounter) = reset_minibatch_train!(m.nlp)
LuxNLPModels.epoch(m::PassCounter) = epoch(m.nlp)

"""
    mean_loss(nlp, x, data; st = testmode_state(nlp, x), chunk = 1_000)

Test-mode loss averaged over `data`, in chunks. Assumes `nlp.loss_f` averages over
observations (as Lux's losses do with the default `agg = mean`).
"""
function mean_loss(nlp::LuxNLPModel, x::AbstractVector, data; st = testmode_state(nlp, x),
                   chunk::Int = 1_000)
  ps = params_view(nlp, x)
  total, n = 0.0, 0
  for batch in DataLoader(data; batchsize = chunk, shuffle = false, partial = true)
    k = size(last(batch), ndims(last(batch)))
    fx, _, _ = nlp.loss_f(nlp.model, ps, st, nlp.device(batch))
    total += Float64(fx) * k
    n += k
  end
  return total / n
end

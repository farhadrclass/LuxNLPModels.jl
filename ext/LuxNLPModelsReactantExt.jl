module LuxNLPModelsReactantExt

# ReactantBackend: compile the residual / objective and their derivatives with Reactant.jl,
# differentiating with Enzyme (forward mode for J(x)v, reverse mode for J(x)ᵀw and gradients).
# Reactant supports nested AD of any order, so residuals may take input derivatives with
# Enzyme (e.g. second derivatives for PINNs) and still be differentiated in the parameters.

using LuxNLPModels: LuxNLPModels, LuxNLPModel, LuxNLSModel, ReactantBackend,
                    _residual, _nls_objective, _objective
import LuxNLPModels: _eval_residual_on, _eval_jvp!, _eval_vjp!, _eval_value_and_grad!, _eval_obj_on
using Reactant: Reactant
using Enzyme: Enzyme
using LinearAlgebra: dot

# Compiled functions and device copies of the state and the current batch. A cache is valid
# for one batch shape: `set_batch!` drops it when the shape changes. Batches of the same
# shape (minibatches) reuse the compiled functions; only the device copy is refreshed.
mutable struct ReactantCache
  batch::Any
  st_ra::Any
  batch_ra::Any
  compiled::Dict{Any, Any}
end

function _cache!(m)
  c = m.backend_cache
  if c === nothing
    c = ReactantCache(m.batch, Reactant.to_rarray(m.st), Reactant.to_rarray(m.batch), Dict{Any, Any}())
    m.backend_cache = c
  elseif c.batch !== m.batch
    c.batch = m.batch
    c.batch_ra = Reactant.to_rarray(m.batch)
  end
  return c
end

# device copy of `batch`: the cached one for the current batch, a fresh one otherwise
_batch_ra(c, m, batch) = batch === m.batch ? c.batch_ra : Reactant.to_rarray(batch)

function _compiled!(c, key, f, args)
  return get!(c.compiled, key) do
    Reactant.compile(f, args)
  end
end

_to_host(y::Reactant.ConcreteRNumber) = Reactant.to_number(y)
_to_host(y) = Array(y)

# ---------------------------------------------------------------------------------------
# Traced functions (arguments: flat parameters, [tangent / cotangent], then constants:
# model, residual or loss, ComponentArray axes, state, batch)
# ---------------------------------------------------------------------------------------

_jvp_fn(x, v, model, R, ax, st, b) =
  only(Enzyme.autodiff(Enzyme.Forward, _residual, Enzyme.Duplicated(x, v), Enzyme.Const(model),
                       Enzyme.Const(R), Enzyme.Const(ax), Enzyme.Const(st), Enzyme.Const(b)))

_dot_residual(x, w, model, R, ax, st, b) = dot(_residual(x, model, R, ax, st, b), w)

function _vjp_fn(x, w, model, R, ax, st, b)
  dx = zero(x)
  Enzyme.autodiff(Enzyme.Reverse, _dot_residual, Enzyme.Active, Enzyme.Duplicated(x, dx),
                  Enzyme.Const(w), Enzyme.Const(model), Enzyme.Const(R), Enzyme.Const(ax),
                  Enzyme.Const(st), Enzyme.Const(b))
  return dx
end

function _value_and_grad_fn(f, x, model, L, ax, st, b)
  dx = zero(x)
  _, val = Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(x, dx),
                           Enzyme.Const(model), Enzyme.Const(L), Enzyme.Const(ax),
                           Enzyme.Const(st), Enzyme.Const(b))
  return val, dx
end
_nls_value_and_grad_fn(x, model, R, ax, st, b) = _value_and_grad_fn(_nls_objective, x, model, R, ax, st, b)
_nlp_value_and_grad_fn(x, model, L, ax, st, b) = _value_and_grad_fn(_objective, x, model, L, ax, st, b)

# ---------------------------------------------------------------------------------------
# LuxNLSModel
# ---------------------------------------------------------------------------------------

function _eval_residual_on(nls::LuxNLSModel, ::ReactantBackend, x, batch)
  c = _cache!(nls)
  args = (Reactant.to_rarray(x), nls.model, nls.residual, nls.axes, c.st_ra, _batch_ra(c, nls, batch))
  f = _compiled!(c, (:residual, LuxNLPModels._shape(batch)), _residual, args)
  return _to_host(f(args...))
end

function _eval_jvp!(nls::LuxNLSModel, ::ReactantBackend, x, v, Jv)
  c = _cache!(nls)
  args = (Reactant.to_rarray(x), Reactant.to_rarray(v), nls.model, nls.residual, nls.axes, c.st_ra, c.batch_ra)
  f = _compiled!(c, :jvp, _jvp_fn, args)
  copyto!(Jv, _to_host(f(args...)))
  return Jv
end

function _eval_vjp!(nls::LuxNLSModel, ::ReactantBackend, x, w, Jtw)
  c = _cache!(nls)
  args = (Reactant.to_rarray(x), Reactant.to_rarray(w), nls.model, nls.residual, nls.axes, c.st_ra, c.batch_ra)
  f = _compiled!(c, :vjp, _vjp_fn, args)
  copyto!(Jtw, _to_host(f(args...)))
  return Jtw
end

function _eval_value_and_grad!(nls::LuxNLSModel, ::ReactantBackend, x, g)
  c = _cache!(nls)
  args = (Reactant.to_rarray(x), nls.model, nls.residual, nls.axes, c.st_ra, c.batch_ra)
  f = _compiled!(c, :value_and_grad, _nls_value_and_grad_fn, args)
  val, dx = f(args...)
  copyto!(g, _to_host(dx))
  return eltype(g)(_to_host(val))
end

# ---------------------------------------------------------------------------------------
# LuxNLPModel
# ---------------------------------------------------------------------------------------

function _eval_obj_on(nlp::LuxNLPModel, ::ReactantBackend, x, batch)
  c = _cache!(nlp)
  args = (Reactant.to_rarray(x), nlp.model, nlp.loss, nlp.axes, c.st_ra, _batch_ra(c, nlp, batch))
  f = _compiled!(c, (:objective, LuxNLPModels._shape(batch)), _objective, args)
  return eltype(x)(_to_host(f(args...)))
end

function _eval_value_and_grad!(nlp::LuxNLPModel, ::ReactantBackend, x, g)
  c = _cache!(nlp)
  args = (Reactant.to_rarray(x), nlp.model, nlp.loss, nlp.axes, c.st_ra, c.batch_ra)
  f = _compiled!(c, :value_and_grad, _nlp_value_and_grad_fn, args)
  val, dx = f(args...)
  copyto!(g, _to_host(dx))
  return eltype(g)(_to_host(val))
end

end # module

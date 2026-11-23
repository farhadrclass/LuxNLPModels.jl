# NLPModels API for unconstrained problems: obj, grad!, objgrad!, hprod!.

_lossvalue(l::Number) = l
_lossvalue(l::Tuple) = first(l)
_lossvalue(l) = throw(ArgumentError(
  "the loss must return a scalar or a tuple `(value, st, stats)`, got $(typeof(l))"))

# The objective as a function of the flat parameter vector; everything else is a constant
# context for DifferentiationInterface.
_objective(x, model, loss, ax, st, batch) =
  _lossvalue(loss(model, ComponentArray(x, ax), st, batch))

_contexts(nlp::LuxNLPModel) =
  (Constant(nlp.model), Constant(nlp.loss), Constant(nlp.axes), Constant(nlp.st), Constant(nlp.batch))

# AD preparations are cached and keyed on the argument types (DI checks them strictly) and
# invalidated by `set_batch!` when the batch shape changes.
function _grad_prep(nlp::LuxNLPModel, x, ctx)
  p = nlp.grad_prep
  if p === nothing || p[1] !== typeof(x)
    p = (typeof(x), DI.prepare_gradient(_objective, nlp.backend, x, ctx...))
    nlp.grad_prep = p
  end
  return p[2]
end

function _hvp_prep(nlp::LuxNLPModel, x, v, ctx)
  p = nlp.hvp_prep
  if p === nothing || p[1] !== (typeof(x), typeof(v))
    p = ((typeof(x), typeof(v)), DI.prepare_hvp(_objective, nlp.hvp_backend, x, (v,), ctx...))
    nlp.hvp_prep = p
  end
  return p[2]
end

_value_and_gradient!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector) =
  _eval_value_and_grad!(nlp, nlp.backend, x, g)

# ---- evaluation hooks (methods for ReactantBackend are in the Reactant extension)
_eval_obj_on(nlp::LuxNLPModel, backend, x, batch) =
  _objective(x, nlp.model, nlp.loss, nlp.axes, nlp.st, batch)

function _eval_value_and_grad!(nlp::LuxNLPModel, backend, x, g)
  ctx = _contexts(nlp)
  prep = _grad_prep(nlp, x, ctx)
  f, _ = DI.value_and_gradient!(_objective, g, prep, backend, x, ctx...)
  return f
end

function NLPModels.obj(nlp::LuxNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  _account!(nlp; forward = 1)
  return _eval_obj_on(nlp, nlp.backend, x, nlp.batch)
end

function NLPModels.grad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  _account!(nlp; forward = 1, backward = 1)
  _value_and_gradient!(nlp, x, g)
  return g
end

# One forward + one backward pass; NLPModels convention counts it as one obj and one grad.
function NLPModels.objgrad!(nlp::LuxNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  _account!(nlp; forward = 1, backward = 1)
  f = _value_and_gradient!(nlp, x, g)
  return f, g
end

function NLPModels.hprod!(nlp::LuxNLPModel, x::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight::Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x v Hv
  nlp.hvp_backend === nothing && throw(ArgumentError("hprod! is not available with this backend"))
  increment!(nlp, :neval_hprod)
  _account!(nlp; hvp = 1)
  ctx = _contexts(nlp)
  prep = _hvp_prep(nlp, x, v, ctx)
  DI.hvp!(_objective, (Hv,), prep, nlp.hvp_backend, x, (v,), ctx...)
  isone(obj_weight) || (Hv .*= obj_weight)
  return Hv
end

# Lagrangian version (no constraints, so y is empty).
function NLPModels.hprod!(nlp::LuxNLPModel, x::AbstractVector, y::AbstractVector,
                          v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(eltype(x)))
  @lencheck 0 y
  return NLPModels.hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

"""
    full_objective(nlp, x; data = nlp.data, st = nlp.st)

Loss on `data` (the full training set by default) at `x`, *without* touching the counters or
the current batch. Intended for monitoring; use `data = test_data` for held-out loss.
"""
full_objective(nlp::LuxNLPModel, x::AbstractVector; data = nlp.data, st = nlp.st) =
  st === nlp.st ? _eval_obj_on(nlp, nlp.backend, x, data) :
                  _objective(x, nlp.model, nlp.loss, nlp.axes, st, data)

"""
    check_derivatives(nlp, x = nlp.meta.x0; ndirs = 3, h = cbrt(eps(T)), hprod = true, rng = Xoshiro(0))

Compare the AD derivatives with central finite differences along `ndirs` random unit
directions `d`: the directional derivative `∇f(x)ᵀd` with `(f(x+hd) - f(x-hd)) / 2h`, and
(if `hprod`) `∇²f(x)d` with `(∇f(x+hd) - ∇f(x-hd)) / 2h`.
Returns the largest relative errors `(grad = ..., hprod = ...)`; counters are left untouched.

Run it in Float64 (`ps = f64(ps)`) for every new loss, in particular losses that themselves
differentiate the network (PINN residuals): some nested-AD combinations return wrong
parameter gradients without any error or warning. Expect errors around `h^2` (truncation)
plus `eps/h` (rounding), i.e. ~1e-10 in Float64 for smooth losses.
"""
function check_derivatives(nlp::LuxNLPModel{T}, x::AbstractVector = nlp.meta.x0;
                           ndirs::Integer = 3, h::Real = cbrt(eps(T)), hprod::Bool = true,
                           rng::AbstractRNG = Random.Xoshiro(0)) where {T}
  counters = deepcopy(nlp.counters)
  wk = deepcopy(nlp.work)
  n = nlp.meta.nvar
  g = similar(x); gp = similar(x); gm = similar(x); Hd = similar(x); d = similar(x)
  grad!(nlp, x, g)
  err_g = zero(T)
  err_h = zero(T)
  for _ in 1:ndirs
    dh = randn(rng, T, n)
    copyto!(d, dh ./ norm(dh))
    fp = obj(nlp, x .+ T(h) .* d)
    fm = obj(nlp, x .- T(h) .* d)
    fd = (fp - fm) / (2 * T(h))
    ad = dot(g, d)
    err_g = max(err_g, abs(fd - ad) / max(abs(fd), abs(ad), eps(T)))
    if hprod
      NLPModels.hprod!(nlp, x, d, Hd)
      grad!(nlp, x .+ T(h) .* d, gp)
      grad!(nlp, x .- T(h) .* d, gm)
      fdH = (gp .- gm) ./ (2 * T(h))
      err_h = max(err_h, norm(fdH .- Hd) / max(norm(fdH), norm(Hd), eps(T)))
    end
  end
  nlp.counters = counters
  nlp.work = wk
  return (grad = err_g, hprod = hprod ? err_h : T(NaN))
end

# ---------------------------------------------------------------------------------------
# Explicit Hessian: NOT RECOMMENDED for neural networks.
#
# The Hessian of a network loss is dense. Storing its lower triangle takes n(n+1)/2 entries
# and every evaluation costs n Hessian-vector products (one per column), i.e. roughly n times
# the cost of a gradient. It is provided only so that solvers that factorize the Hessian
# (e.g. `R2N` with `MA57R2NSubsolver` / `MA97R2NSubsolver`, which need HSL) can be tried on
# *small* networks, as a reference point. Prefer `hprod!` / `hess_op` (R2N's default Krylov
# subsolvers CG, CR, MINRES; `trunk`) or quasi-Newton models, which never form the matrix.
# ---------------------------------------------------------------------------------------

function _check_hess_size(nlp::LuxNLPModel)
  n = nlp.meta.nvar
  n ≤ nlp.max_hess_nvar || throw(ArgumentError(
    "the explicit Hessian of this model would be dense with $n variables " *
    "($(nlp.meta.nnzh) stored entries, $n Hessian-vector products per evaluation). " *
    "This is not recommended; use `hprod!`/`hess_op` (e.g. R2N with CGR2NSubsolver) instead, " *
    "or raise the limit with `LuxNLPModel(...; max_hess_nvar = $n)`."))
  return nothing
end

"""
    hess_structure!(nlp::LuxNLPModel, rows, cols)

Structure of the explicit Hessian: the dense lower triangle, column by column
(`nlp.meta.nnzh = n(n+1)/2` entries).

!!! warning "Not recommended"
    See [`hess_coord!`](@ref). Throws an `ArgumentError` when `nvar > max_hess_nvar`.
"""
function NLPModels.hess_structure!(nlp::LuxNLPModel, rows::AbstractVector{<:Integer},
                                   cols::AbstractVector{<:Integer})
  @lencheck nlp.meta.nnzh rows cols
  _check_hess_size(nlp)
  n = nlp.meta.nvar
  k = 0
  for j in 1:n, i in j:n
    k += 1
    rows[k] = i
    cols[k] = j
  end
  return rows, cols
end

"""
    hess_coord!(nlp::LuxNLPModel, x, y, vals; obj_weight = 1)

Values of the explicit Hessian (dense lower triangle, ordered as in `hess_structure!`),
computed column by column with `n` Hessian-vector products. `y` must be empty (no
constraints). `hess_coord!(nlp, x, vals)`, `hess_coord` and `hess` from NLPModels call this method.

!!! warning "Not recommended"
    Neural-network Hessians are dense: `n(n+1)/2` entries and `n` Hessian-vector products
    per evaluation. Only for small networks (e.g. to try R2N with a direct HSL subsolver as
    a reference). Prefer `hprod!` / `hess_op`. Throws an `ArgumentError` when
    `nvar > max_hess_nvar` (default 5_000, set in the constructor).
"""
function NLPModels.hess_coord!(nlp::LuxNLPModel, x::AbstractVector, y::AbstractVector,
                               vals::AbstractVector; obj_weight::Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck 0 y
  @lencheck nlp.meta.nnzh vals
  _check_hess_size(nlp)
  increment!(nlp, :neval_hess)
  n = nlp.meta.nvar
  _account!(nlp; hvp = n)
  ctx = _contexts(nlp)
  e = fill!(similar(x), 0)
  Hcol = similar(x)
  prep = _hvp_prep(nlp, x, e, ctx)
  k = 1
  for j in 1:n
    e[j:j] .= 1                       # range broadcast: no scalar indexing on GPU arrays
    DI.hvp!(_objective, (Hcol,), prep, nlp.hvp_backend, x, (e,), ctx...)
    e[j:j] .= 0
    len = n - j + 1
    copyto!(vals, k, Hcol, j, len)    # lower part of column j
    k += len
  end
  isone(obj_weight) || (vals .*= obj_weight)
  return vals
end

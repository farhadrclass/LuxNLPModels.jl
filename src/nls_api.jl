# NLSModels API: residual, J(x)v, J(x)ᵀw (used by Gauss-Newton type solvers such as R2NLS
# with LSMR, trunk and tron on NLS models), plus a fused gradient and hprod for NLP solvers.

_nls_contexts(nls::LuxNLSModel) =
  (Constant(nls.model), Constant(nls.residual), Constant(nls.axes), Constant(nls.st), Constant(nls.batch))

# Cached DI preparation stored in `field`, rebuilt when the argument types change
# (`set_batch!` clears all preparations when the batch shape changes).
function _cached_prep!(nls::LuxNLSModel, field::Symbol, key, build)
  p = getfield(nls, field)
  if p === nothing || p[1] != key
    p = (key, build())
    setfield!(nls, field, p)
  end
  return p[2]
end

function NLPModels.residual!(nls::LuxNLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  _account!(nls; forward = 1)
  copyto!(Fx, _eval_residual_on(nls, nls.backend, x, nls.batch))
  return Fx
end

# ---- evaluation hooks: DifferentiationInterface backends (methods for ReactantBackend are
# ---- defined in the Reactant extension)

_eval_residual_on(nls::LuxNLSModel, backend, x, batch) =
  _residual(x, nls.model, nls.residual, nls.axes, nls.st, batch)

function _eval_jvp!(nls::LuxNLSModel, backend, x, v, Jv)
  ctx = _nls_contexts(nls)
  prep = _cached_prep!(nls, :jvp_prep, (typeof(x), typeof(v)),
                       () -> DI.prepare_pushforward(_residual, backend, x, (v,), ctx...))
  DI.pushforward!(_residual, (Jv,), prep, backend, x, (v,), ctx...)
  return Jv
end

function _eval_vjp!(nls::LuxNLSModel, backend, x, w, Jtw)
  ctx = _nls_contexts(nls)
  prep = _cached_prep!(nls, :vjp_prep, (typeof(x), typeof(w)),
                       () -> DI.prepare_pullback(_residual, backend, x, (w,), ctx...))
  DI.pullback!(_residual, (Jtw,), prep, backend, x, (w,), ctx...)
  return Jtw
end

function _eval_value_and_grad!(nls::LuxNLSModel, backend, x, g)
  ctx = _nls_contexts(nls)
  prep = _cached_prep!(nls, :grad_prep, typeof(x),
                       () -> DI.prepare_gradient(_nls_objective, backend, x, ctx...))
  f, _ = DI.value_and_gradient!(_nls_objective, g, prep, backend, x, ctx...)
  return f
end

function NLPModels.jprod_residual!(nls::LuxNLSModel, x::AbstractVector, v::AbstractVector,
                                   Jv::AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  _account!(nls; jvp = 1)
  return _eval_jvp!(nls, nls.fwd_backend, x, v, Jv)
end

function NLPModels.jtprod_residual!(nls::LuxNLSModel, x::AbstractVector, w::AbstractVector,
                                    Jtw::AbstractVector)
  @lencheck nls.meta.nvar x Jtw
  @lencheck nls.nls_meta.nequ w
  increment!(nls, :neval_jtprod_residual)
  _account!(nls; forward = 1, backward = 1)
  return _eval_vjp!(nls, nls.backend, x, w, Jtw)
end

# ∇(½‖F‖²) = J(x)ᵀF(x) in one forward and one reverse pass (the generic NLSModels
# implementation evaluates the residual, then calls jtprod_residual!: two forward passes).
_nls_value_and_gradient!(nls::LuxNLSModel, x, g) = _eval_value_and_grad!(nls, nls.backend, x, g)

function NLPModels.grad!(nls::LuxNLSModel, x::AbstractVector, g::AbstractVector)
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_grad)
  _account!(nls; forward = 1, backward = 1)
  _nls_value_and_gradient!(nls, x, g)
  return g
end

function NLPModels.objgrad!(nls::LuxNLSModel, x::AbstractVector, g::AbstractVector)
  @lencheck nls.meta.nvar x g
  increment!(nls, :neval_obj)
  increment!(nls, :neval_grad)
  _account!(nls; forward = 1, backward = 1)
  f = _nls_value_and_gradient!(nls, x, g)
  return f, g
end

# Exact Hessian-vector product of ½‖F‖² (Gauss-Newton term plus residual curvature), so that
# general NLP solvers (R2N, trunk on the NLP view) can also be used. For PINN residuals this
# is a derivative of order (input order + 2); Gauss-Newton solvers do not need it.
function NLPModels.hprod!(nls::LuxNLSModel, x::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight::Real = one(eltype(x)))
  @lencheck nls.meta.nvar x v Hv
  nls.hvp_backend === nothing &&
    throw(ArgumentError("hprod! is not available with this backend; use a Gauss-Newton solver (R2NLS, trunk, tron on the NLS model)"))
  increment!(nls, :neval_hprod)
  _account!(nls; hvp = 1)
  ctx = _nls_contexts(nls)
  prep = _cached_prep!(nls, :hvp_prep, (typeof(x), typeof(v)),
                       () -> DI.prepare_hvp(_nls_objective, nls.hvp_backend, x, (v,), ctx...))
  DI.hvp!(_nls_objective, (Hv,), prep, nls.hvp_backend, x, (v,), ctx...)
  isone(obj_weight) || (Hv .*= obj_weight)
  return Hv
end

function NLPModels.hprod!(nls::LuxNLSModel, x::AbstractVector, y::AbstractVector,
                          v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(eltype(x)))
  @lencheck 0 y
  return NLPModels.hprod!(nls, x, v, Hv; obj_weight = obj_weight)
end

full_objective(nls::LuxNLSModel, x::AbstractVector; data = nls.data, st = nls.st) =
  sum(abs2, full_residual(nls, x; data, st)) / 2

"""
    full_residual(nls, x; data = nls.data, st = nls.st)

Residual on `data` (the full training data by default), without touching counters or batch.
"""
full_residual(nls::LuxNLSModel, x::AbstractVector; data = nls.data, st = nls.st) =
  st === nls.st ? _eval_residual_on(nls, nls.backend, x, data) :
                  _residual(x, nls.model, nls.residual, nls.axes, st, data)

# ---------------------------------------------------------------------------------------
# Explicit Jacobian: NOT RECOMMENDED for neural networks.
#
# The residual Jacobian of a network is dense (nequ × nvar entries) and forming it costs
# min(nequ, nvar) Jacobian-vector products. It is provided only so that R2NLS's default
# `QRMumpsSubsolver` (a sparse QR factorization) runs on *small* problems. Prefer the
# operator subsolvers `LSMRSubsolver`, `LSQRSubsolver`, `CGLSSubsolver`, which only use
# J(x)v and J(x)ᵀw.
# ---------------------------------------------------------------------------------------

function _check_jac_size(nls::LuxNLSModel)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  m * n ≤ nls.max_jac_entries || throw(ArgumentError(
    "the explicit residual Jacobian would be dense with $m × $n = $(m * n) entries. This is " *
    "not recommended; use an operator subsolver (e.g. `R2NLS(nls; subsolver = LSMRSubsolver)`) " *
    "or raise the limit with `LuxNLSModel(...; max_jac_entries = $(m * n))`."))
  return nothing
end

"""
    jac_structure_residual!(nls::LuxNLSModel, rows, cols)

Dense structure (column-major) of the residual Jacobian.

!!! warning "Not recommended"
    See [`jac_coord_residual!`](@ref).
"""
function NLPModels.jac_structure_residual!(nls::LuxNLSModel, rows::AbstractVector{<:Integer},
                                           cols::AbstractVector{<:Integer})
  @lencheck nls.nls_meta.nnzj rows cols
  _check_jac_size(nls)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  k = 0
  for j in 1:n, i in 1:m
    k += 1
    rows[k] = i
    cols[k] = j
  end
  return rows, cols
end

"""
    jac_coord_residual!(nls::LuxNLSModel, x, vals)

Values of the dense residual Jacobian (column-major), computed with `min(nequ, nvar)`
reverse or forward passes.

!!! warning "Not recommended"
    The Jacobian of a network residual is dense: `nequ × nvar` entries. Use it only for small
    problems (R2NLS's default `QRMumpsSubsolver`); prefer `LSMRSubsolver`, `LSQRSubsolver` or
    `CGLSSubsolver`. Throws an `ArgumentError` above `max_jac_entries` entries.
"""
function NLPModels.jac_coord_residual!(nls::LuxNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  _check_jac_size(nls)
  increment!(nls, :neval_jac_residual)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  rowwise = m ≤ n
  rowwise ? _account!(nls; forward = m, backward = m) : _account!(nls; jvp = n)
  backend = rowwise ? nls.backend : nls.fwd_backend
  ctx = _nls_contexts(nls)
  prep = _cached_prep!(nls, :jac_prep, (typeof(x), rowwise),
                       () -> DI.prepare_jacobian(_residual, backend, x, ctx...))
  J = DI.jacobian(_residual, prep, backend, x, ctx...)
  copyto!(vals, vec(J))
  return vals
end

# ---------------------------------------------------------------------------------------
# Derivative checks
# ---------------------------------------------------------------------------------------

"""
    check_derivatives(nls::LuxNLSModel, x = nls.meta.x0; ndirs = 3, h = cbrt(eps(T)), hprod = false, rng = Xoshiro(0))

Checks the residual derivatives along `ndirs` random unit directions `v ∈ ℝⁿ`, `w ∈ ℝᵐ`:

- `jprod`:   `J(x)v` against `(F(x+hv) - F(x-hv)) / 2h`;
- `adjoint`: `wᵀ(J(x)v)` against `(J(x)ᵀw)ᵀv`, relative to `‖w‖‖Jv‖`. This identity holds
  exactly, so forward- and reverse-mode results that disagree (the typical failure of an
  unsupported nested-AD combination) show up at round-off level independently of `h`;
- `grad`:    the fused gradient against `J(x)ᵀF(x)` (the definition of ∇½‖F‖²);
- `hprod`:   optionally, `hprod!` against finite differences of the gradient.

Returns the largest relative errors; counters and work are left untouched. Run it in Float64.
"""
function check_derivatives(nls::LuxNLSModel{T}, x::AbstractVector = nls.meta.x0;
                           ndirs::Integer = 3, h::Real = cbrt(eps(T)), hprod::Bool = false,
                           rng::AbstractRNG = Random.Xoshiro(0)) where {T}
  counters = deepcopy(nls.counters)
  wk = deepcopy(nls.work)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  Fx = similar(x, m); Fp = similar(x, m); Fm = similar(x, m); Jv = similar(x, m)
  w = similar(x, m); v = similar(x); Jtw = similar(x); g = similar(x); JtF = similar(x)
  Hv = similar(x); gp = similar(x); gm = similar(x)
  hT = T(h)
  residual!(nls, x, Fx)
  grad!(nls, x, g)
  jtprod_residual!(nls, x, Fx, JtF)
  err_grad = norm(g .- JtF) / max(norm(JtF), eps(T))
  err_jprod = zero(T); err_adj = zero(T); err_h = zero(T)
  for _ in 1:ndirs
    vh = randn(rng, T, n); copyto!(v, vh ./ norm(vh))
    wh = randn(rng, T, m); copyto!(w, wh ./ norm(wh))
    jprod_residual!(nls, x, v, Jv)
    residual!(nls, x .+ hT .* v, Fp)
    residual!(nls, x .- hT .* v, Fm)
    fd = (Fp .- Fm) ./ (2hT)
    err_jprod = max(err_jprod, norm(fd .- Jv) / max(norm(fd), norm(Jv), eps(T)))
    jtprod_residual!(nls, x, w, Jtw)
    a, b = dot(w, Jv), dot(Jtw, v)
    err_adj = max(err_adj, abs(a - b) / max(norm(Jv), norm(Jtw), eps(T)))
    if hprod
      NLPModels.hprod!(nls, x, v, Hv)
      grad!(nls, x .+ hT .* v, gp)
      grad!(nls, x .- hT .* v, gm)
      fdH = (gp .- gm) ./ (2hT)
      err_h = max(err_h, norm(fdH .- Hv) / max(norm(fdH), norm(Hv), eps(T)))
    end
  end
  nls.counters = counters
  nls.work = wk
  return (jprod = err_jprod, adjoint = err_adj, grad = err_grad, hprod = hprod ? err_h : T(NaN))
end

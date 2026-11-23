"""
    LuxNLSModel{T, S} <: AbstractNLSModel{T, S}

Nonlinear least-squares model `min_x ½‖F(x)‖²` with residual
`F(x) = vec(residual(model, ps(x), st, batch))`, where `ps(x)` is the `ComponentArray` view of
the flat parameter vector `x`.

Least-squares solvers (`R2NLS` with `LSMRSubsolver`/`LSQRSubsolver`/`CGLSSubsolver`, `trunk`
and `tron` on NLS models, i.e. Gauss-Newton type methods) use only the residual and
Jacobian-vector products `J(x)v` (forward mode) and `J(x)ᵀw` (reverse mode). General NLP
solvers and Optimisers.jl rules can use the same model through `obj`, `grad!` and `hprod!`.

See [`LuxNLSModel(model, residual, data; kwargs...)`](@ref).
"""
mutable struct LuxNLSModel{T, S <: AbstractVector{T}, M, R, AX, ST, D, B, AD, FAD, HAD} <:
               AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  work::WorkCounters
  model::M
  residual::R
  axes::AX
  st::ST
  data::D
  batch::B
  nobs::Int
  batch_nobs::Int
  sampler::Union{Nothing, AbstractBatchSampler}
  backend::AD        # reverse mode: Jᵀw, gradient
  fwd_backend::FAD   # forward mode: Jv
  hvp_backend::HAD
  grad_prep::Any
  jvp_prep::Any
  vjp_prep::Any
  hvp_prep::Any
  jac_prep::Any
  prep_shape::Any
  backend_cache::Any   # used by non-DI backends (ReactantBackend)
  max_jac_entries::Int
  max_hess_nvar::Int
end

_resvalue(r::AbstractArray) = r
_resvalue(r::Tuple) = first(r)
_resvalue(r) = throw(ArgumentError(
  "the residual must return an array or a tuple `(array, st, stats)`, got $(typeof(r))"))

# Residual as a function of the flat parameter vector (all other arguments are constants).
_residual(x, model, residual, ax, st, batch) =
  vec(_resvalue(residual(model, ComponentArray(x, ax), st, batch)))

_nls_objective(x, model, residual, ax, st, batch) =
  sum(abs2, _residual(x, model, residual, ax, st, batch)) / 2

"""
    LuxNLSModel(model, residual, data; kwargs...)

Build a nonlinear least-squares model `min ½‖F(x)‖²` from a Lux `model`.

# Arguments
- `residual(model, ps, st, batch)`: returns an array (or a tuple `(array, st, stats)`) whose
  entries are the residuals; it is flattened with `vec`. Include any weights and
  normalizations in it: with `F = [r_interior * √(2/N_i); r_boundary * √(2λ/N_b)]`, the
  objective `½‖F‖²` equals the usual PINN loss `mean(r_i²) + λ mean(r_b²)`.
- `data`: full data, passed to `residual` as `batch` in full-batch mode.

# Keyword arguments
Same as [`LuxNLPModel`](@ref) (`rng`, `ps`, `st`, `device`, `batchsize`, `shuffle`, `max_hess_nvar`,
`name`), plus:
- `backend = AutoZygote()`: reverse-mode backend for `J(x)ᵀw` and the gradient, or
  `ReactantBackend()` (requires `using Reactant, Enzyme`) to compile the residual and all its
  derivatives with Reactant/Enzyme, which supports nested AD of any order inside the residual.
- `fwd_backend = AutoForwardDiff()` (`ReactantBackend()` when `backend` is): forward mode for `J(x)v`.
- `hvp_backend = SecondOrder(fwd_backend, backend)`: for `hprod!` of `½‖F‖²` (general NLP solvers
  only; not available with `ReactantBackend`).
- `max_jac_entries = 10_000_000`: size guard for the explicit Jacobian (`jac_coord_residual!`,
  needed by R2NLS's default `QRMumpsSubsolver`). Not recommended for neural networks.

# Notes
- The number of residuals `nequ` is fixed. In minibatch mode every batch must produce the
  same number of residuals, hence `partial = false` is required.
- If the residual differentiates the network with respect to its inputs (PINNs), `J(x)v` and
  `J(x)ᵀw` are mixed derivatives of order (input order + 1). Check them with
  [`check_derivatives`](@ref) before training.
"""
function LuxNLSModel(model, residual, data;
                     rng::AbstractRNG = Random.default_rng(),
                     ps = nothing,
                     st = nothing,
                     device = identity,
                     batchsize = nothing,
                     shuffle::Bool = true,
                     partial::Bool = false,
                     backend = AutoZygote(),
                     fwd_backend = backend isa ReactantBackend ? backend : AutoForwardDiff(),
                     hvp_backend = backend isa ReactantBackend ? nothing : DI.SecondOrder(fwd_backend, backend),
                     max_jac_entries::Integer = 10_000_000,
                     max_hess_nvar::Integer = 5_000,
                     name::AbstractString = "LuxNLSModel")
  (partial && batchsize !== nothing) && throw(ArgumentError(
    "LuxNLSModel has a fixed number of residuals: minibatches must all have the same size (partial = false)"))
  _check_backend_loaded(backend)
  _check_backend_loaded(fwd_backend)
  c = _lux_setup(model, data; rng, ps, st, device, batchsize, shuffle, partial)
  T = eltype(c.x0)
  S = typeof(c.x0)
  n = length(c.x0)
  meta = _unconstrained_meta(c.x0, name)
  # nequ is found by evaluating the residual once (through the backend, so that compiled
  # backends reuse the compilation); the metadata is completed afterwards.
  nls = LuxNLSModel(meta, NLSMeta{T, S}(1, n; x0 = c.x0), NLSCounters(), WorkCounters(),
                    model, residual, c.ax, c.st, c.data, c.batch, c.nobs, c.batch_nobs,
                    c.sampler, backend, fwd_backend, hvp_backend, nothing, nothing, nothing,
                    nothing, nothing, _shape(c.batch), nothing, Int(max_jac_entries),
                    Int(max_hess_nvar))
  nequ = length(_eval_residual_on(nls, backend, c.x0, c.batch))
  nls.nls_meta = NLSMeta{T, S}(nequ, n; x0 = c.x0, nnzj = nequ * n, nnzh = div(n * (n + 1), 2))
  return nls
end

function Base.show(io::IO, nls::LuxNLSModel)
  print(io, "LuxNLSModel: ", nls.nls_meta.nequ, " residuals, ", nls.meta.nvar,
        " variables of type ", eltype(nls.meta.x0), " in ", typeof(nls.meta.x0).name.name,
        ", ", _mode_string(nls))
end

function _clear_preps!(nls::LuxNLSModel)
  nls.grad_prep = nothing
  nls.jvp_prep = nothing
  nls.vjp_prep = nothing
  nls.hvp_prep = nothing
  nls.jac_prep = nothing
  nls.backend_cache = nothing
  return nls
end

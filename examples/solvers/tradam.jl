# TRAdam: Adam's direction with a trust region in the norm defined by Adam's diagonal scaling.
# PROTOTYPE, not part of LuxNLPModels: it needs only the NLPModels API (obj, grad!, objgrad!) and
# the `on_accept!` hook of the stochastic contract, so it runs on any AbstractNLPModel. When it
# matures it belongs in its own package or in JSOSolvers (align with src/fomo.jl first:
# parameter-set struct as in FOMOParameterSet, docstring layout, allocation tests).
# A second-order variant would add hprod! calls at x_k, on the same batch; the contract covers it.
#
# With H = diag(h), h = sqrt(v̂) + ϵv (bias-corrected second moment), the step is
#     s = -τ H⁻¹ d,  τ = min(1, Δ / ‖d‖_{H⁻¹}),  so ‖s‖_H ≤ Δ,
# where d = (1-β) g + β b̂ is Adam's bias-corrected first moment when β = βeff, and β is reduced
# (damping) so that gᵀH⁻¹d ≥ λ1 gᵀH⁻¹g and ‖d‖_{H⁻¹} ≤ λ2 ‖g‖_{H⁻¹}. Then
#     pred = -gᵀs - sᵀHs/2 ≥ τ gᵀH⁻¹g (λ1 - λ2²/2) > 0   when λ1 > λ2²/2.
#
# GPU: each scalar reduction blocks the host. Per iteration this version does two fused
# reductions and one `obj`, plus `objgrad!` and `norm(g)` after an accepted step.
#
# STATUS: prototype, not executed. Lines marked CHECK depend on package behaviour.
using LinearAlgebra, NLPModels, SolverCore

mutable struct TRAdamSolver{T, V} <: AbstractOptimizationSolver
  x::V
  g::V
  d::V
  s::V
  xt::V
  bhat::V
  v::V
  vplus::V
  h::V
end

function TRAdamSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x0 = nlp.meta.x0
  return TRAdamSolver{T, V}(ntuple(_ -> similar(x0), 9)...)
end

SolverCore.reset!(solver::TRAdamSolver) = solver                 # moments are reset in solve!
SolverCore.reset!(solver::TRAdamSolver, ::AbstractNLPModel) = solver

function TRAdam(nlp::AbstractNLPModel; kwargs...)
  solver = TRAdamSolver(nlp)
  stats = GenericExecutionStats(nlp)
  return SolverCore.solve!(solver, nlp, stats; kwargs...)
end

# N weighted sums in one pass: one kernel and one host synchronization on GPU.
_tadd(p::NTuple{N, T}, q::NTuple{N, T}) where {N, T} = ntuple(i -> p[i] + q[i], Val(N))
function _fused_sums(f, ::Val{N}, xs::AbstractVector{T}...) where {N, T}
  bc = Broadcast.instantiate(Broadcast.broadcasted(f, xs...))
  return mapreduce(identity, _tadd, bc; init = ntuple(_ -> zero(T), Val(N)))   # CHECK on CuArray
end

"""
    damping(a, c, A, βeff, λ1, λ2)

Largest β ∈ [0, βeff] such that d = (1-β) g + β b satisfies gᵀH⁻¹d ≥ λ1 a and
‖d‖²_{H⁻¹} ≤ λ2² a, with a = gᵀH⁻¹g, c = gᵀH⁻¹b, A = ‖g - b‖²_{H⁻¹} (computed directly, not as
a - 2c + bᵀH⁻¹b, to avoid cancellation when b ≈ g).
"""
function damping(a::T, c::T, A::T, βeff::T, λ1::T, λ2::T) where {T}
  B = c - a
  βI = c >= a ? T(Inf) : (1 - λ1) * a / (a - c)
  βII = A <= 0 ? T(Inf) : (-B + sqrt(B^2 + A * (λ2^2 - 1) * a)) / A
  return min(βeff, βI, βII)
end

function SolverCore.solve!(
  solver::TRAdamSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_time::Float64 = 30.0,
  max_iter::Int = typemax(Int),
  max_eval::Int = -1,
  Δ0::T = T(1e-3),
  Δmax::T = T(1),
  η1::T = eps(T)^(1 // 4),
  η2::T = T(95 // 100),
  γ1::T = T(1 // 2),
  γ2::T = T(2),
  β1::T = T(9 // 10),
  β2::T = T(999 // 1000),
  ϵv::T = T(1e-8),
  λ1::T = T(3 // 4),
  λ2::T = T(6 // 5),
  stochastic::Bool = false,
  on_accept! = nlp -> nothing,
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("TRAdam should only be called on unconstrained problems.")
  λ1 > λ2^2 / 2 && λ1 <= 1 <= λ2 || error("need λ1 ≤ 1 ≤ λ2 and λ1 > λ2²/2")
  SolverCore.reset!(stats)
  start_time = time()
  xk, g, d, s, xt = solver.x, solver.g, solver.d, solver.s, solver.xt
  bhat, v, vplus, h = solver.bhat, solver.v, solver.vplus, solver.h
  xk .= x
  fill!(bhat, zero(T))
  fill!(v, zero(T))
  j = 1
  Δ = Δ0
  fk, _ = objgrad!(nlp, xk, g)
  gnorm = norm(g)
  ϵ = atol + rtol * gnorm
  ρ = zero(T)

  verbose > 0 && @info log_header([:iter, :f, :dual, :radius, :ratio], [Int, T, T, T, T])
  set_iter!(stats, 0)
  set_objective!(stats, fk)
  set_dual_residual!(stats, gnorm)
  set_time!(stats, time() - start_time)
  set_status!(stats, get_status(nlp; elapsed_time = stats.elapsed_time, optimal = gnorm <= ϵ,
                                max_eval = max_eval, iter = 0, max_iter = max_iter, max_time = max_time))
  callback(nlp, solver, stats)
  done = stats.status != :unknown

  while !done
    # diagonal scaling and damped momentum
    @. vplus = β2 * v + (1 - β2) * g^2
    bc2 = 1 - β2^j
    @. h = sqrt(vplus / bc2) + ϵv
    a, c, A = _fused_sums((gi, bi, hi) -> (gi^2 / hi, gi * bi / hi, (gi - bi)^2 / hi), Val(3), g, bhat, h)
    βeff = β1 * (1 - β1^(j - 1)) / (1 - β1^j)
    β̃ = damping(a, c, A, βeff, λ1, λ2)
    @. d = (1 - β̃) * g + β̃ * bhat

    # trust-region step in the H-norm and predicted decrease
    gd, dd = _fused_sums((gi, di, hi) -> (gi * di / hi, di^2 / hi), Val(2), g, d, h)
    if !(dd > 0)                             # d = 0: no step can be taken
      set_status!(stats, :small_step)
      break
    end
    τ = min(one(T), Δ / sqrt(dd))
    @. s = -τ * d / h
    pred = τ * gd - τ^2 * dd / 2             # = -gᵀs - sᵀHs/2
    @. xt = xk + s
    ft = obj(nlp, xt)                        # same batch as fk (contract, item 1)
    ρ = pred > 0 ? (fk - ft) / pred : -T(Inf)

    if ρ >= η1
      xk .= xt
      bhat .= d
      v .= vplus
      j += 1
      if stochastic
        on_accept!(nlp)                      # commit state, switch batch (contract, item 2)
        fk, _ = objgrad!(nlp, xk, g)
      else
        fk = ft
        grad!(nlp, xk, g)
      end
      gnorm = norm(g)
    end
    Δ = ρ < η1 ? γ1 * Δ : (ρ >= η2 ? min(γ2 * Δ, Δmax) : Δ)

    set_iter!(stats, stats.iter + 1)
    set_objective!(stats, fk)
    set_dual_residual!(stats, gnorm)
    set_time!(stats, time() - start_time)
    set_solver_specific!(stats, :radius, Δ)
    verbose > 0 && mod(stats.iter, verbose) == 0 && @info log_row(Any[stats.iter, fk, gnorm, Δ, ρ])
    set_status!(stats, get_status(nlp; elapsed_time = stats.elapsed_time,
                                  optimal = !stochastic && gnorm <= ϵ, max_eval = max_eval,
                                  iter = stats.iter, max_iter = max_iter, max_time = max_time))
    callback(nlp, solver, stats)             # never switches the batch (contract, item 3)
    done = stats.status != :unknown
  end
  set_solution!(stats, xk)
  return stats
end

# Baseline: Optimisers.jl rules (Adam, Descent, Momentum, ...) driven through the same
# NLPModel, counters, callbacks and stopping rules as the JSO solvers.

"""
    OptimisersSolver

Minimal solver object exposing `x`, `gx` (like JSOSolvers solvers), the Optimisers.jl `rule`
and its `state`, so that callbacks written for JSO solvers work unchanged.
"""
mutable struct OptimisersSolver{S, R, ST}
  x::S
  gx::S
  rule::R
  state::ST
end

"""
    stats = train_optimisers(nlp, rule; kwargs...)

Run the Optimisers.jl `rule` (e.g. `Adam(1f-3)`, `Descent(0.1f0)`) on `nlp`, one
`objgrad!` per iteration. Returns a `SolverCore.GenericExecutionStats`, like JSO solvers.

In minibatch mode the batch is advanced after every update (one pass over the data per
epoch). Stopping rules mirror JSOSolvers (`SolverCore.get_status`) plus `max_epochs`:

- `x = nlp.meta.x0`, `atol = √eps(T)`, `rtol = √eps(T)` (on the gradient norm of the current batch),
- `max_iter = typemax(Int)`, `max_eval = -1`, `max_time = 30.0` (same default as JSOSolvers),
- `max_epochs = Inf`,
- `callback = (nlp, solver, stats) -> nothing` (set `stats.status = :user` to stop),
- `verbose = 0` (print every `verbose` iterations).
"""
function train_optimisers(nlp::LuxModels{T, S}, rule::Optimisers.AbstractRule;
                          x::S = nlp.meta.x0,
                          atol::Real = √eps(T),
                          rtol::Real = √eps(T),
                          max_iter::Integer = typemax(Int),
                          max_eval::Integer = -1,
                          max_time::Real = 30.0,
                          max_epochs::Real = Inf,
                          minibatch::Bool = nlp.sampler !== nothing,
                          callback = (args...) -> nothing,
                          verbose::Integer = 0) where {T, S}
  xk = copy(x)
  solver = OptimisersSolver(xk, similar(xk), rule, Optimisers.setup(rule, xk))
  stats = GenericExecutionStats(nlp)
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  f, _ = objgrad!(nlp, solver.x, solver.gx)
  gnorm = norm(solver.gx)
  set_objective!(stats, f)
  set_dual_residual!(stats, gnorm)
  ϵ = atol + rtol * gnorm

  _status!(stats, nlp, gnorm ≤ ϵ, f, max_eval, max_iter, max_time)
  verbose > 0 && @info "iter = 0  f = $f  ‖g‖ = $gnorm"
  callback(nlp, solver, stats)

  while stats.status == :unknown
    # iteration k: update with the gradient of batch k, then evaluate on batch k + 1
    solver.state, solver.x = Optimisers.update!(solver.state, solver.x, solver.gx)
    set_iter!(stats, stats.iter + 1)
    minibatch && next_batch!(nlp)
    if epoch(nlp) > max_epochs
      # every gradient of `max_epochs` epochs has been used; `stats.objective` refers to the
      # last evaluated batch, i.e. to the iterate before this final update.
      set_time!(stats, time() - start_time)
      set_status!(stats, :user)
      break
    end
    f, _ = objgrad!(nlp, solver.x, solver.gx)
    gnorm = norm(solver.gx)
    set_objective!(stats, f)
    set_dual_residual!(stats, gnorm)
    set_time!(stats, time() - start_time)
    _status!(stats, nlp, gnorm ≤ ϵ, f, max_eval, max_iter, max_time)
    verbose > 0 && stats.iter % verbose == 0 &&
      @info "iter = $(stats.iter)  f = $f  ‖g‖ = $gnorm  epoch = $(epoch(nlp))"
    callback(nlp, solver, stats)
  end

  set_time!(stats, time() - start_time)
  set_solution!(stats, solver.x)
  return stats
end

function _status!(stats, nlp, optimal, f, max_eval, max_iter, max_time)
  status = SolverCore.get_status(nlp;
                                 elapsed_time = stats.elapsed_time,
                                 optimal = optimal,
                                 unbounded = f == -Inf,
                                 exception = isnan(f) || f == Inf,
                                 max_eval = max_eval,
                                 iter = stats.iter,
                                 max_iter = max_iter,
                                 max_time = Float64(max_time))
  # The callback of the previous iteration may have requested a stop.
  stats.status == :unknown && set_status!(stats, status)
  return stats
end

"""
    BatchSolver

Outer "solver" object of [`train_by_batches`](@ref): `x` (current iterate), `inner` (stats
of the last inner solve) and `inner_statuses` (count of inner termination statuses).
"""
mutable struct BatchSolver{S}
  x::S
  inner::Any
  inner_statuses::Dict{Symbol, Int}
end

"""
    stats = train_by_batches(nlp, solve; kwargs...)

Sample-then-solve training for minibatch mode: for each minibatch, call
`inner = solve(nlp, x)` (e.g. `(nlp, x) -> R2(nlp; x, max_iter = 10)`), warm-start from
`inner.solution`, then advance to the next batch.

Every inner solve sees a fixed batch, hence a *deterministic* objective, so the solver's
acceptance tests, line searches and quasi-Newton updates are used as designed. Solver
state (step size, L-BFGS memory, ...) is not carried across batches unless `solve`
closes over it.

One outer iteration = one batch. `callback(nlp, solver, stats)` is called after every outer
iteration with `solver.x` and with `stats.objective`/`stats.dual_feas` taken from the inner
solve, so [`logging_callback`](@ref) produces logs comparable to `train_optimisers`.

Keywords: `x = nlp.meta.x0`, `max_epochs = 1`, `max_iter = typemax(Int)` (outer iterations),
`max_time = 30.0`, `callback`, `verbose = 0`.
"""
function train_by_batches(nlp::LuxModels{T, S}, solve;
                          x::S = nlp.meta.x0,
                          max_epochs::Real = 1,
                          max_iter::Integer = typemax(Int),
                          max_time::Real = 30.0,
                          callback = (args...) -> nothing,
                          verbose::Integer = 0) where {T, S}
  nlp.sampler === nothing &&
    throw(ArgumentError("train_by_batches requires minibatch mode (construct with `batchsize`)"))
  xk = copy(x)
  stats = GenericExecutionStats(nlp)
  solver = BatchSolver(xk, nothing, Dict{Symbol, Int}())
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)
  # iteration-0 record: monitoring value, not counted as work
  set_objective!(stats, T(full_objective(nlp, xk; data = nlp.batch)))
  set_dual_residual!(stats, T(NaN))
  set_status!(stats, :unknown)
  callback(nlp, solver, stats)

  while stats.status == :unknown
    inner = solve(nlp, xk)
    solver.inner = inner
    solver.inner_statuses[inner.status] = get(solver.inner_statuses, inner.status, 0) + 1
    xk .= inner.solution
    set_iter!(stats, stats.iter + 1)
    set_objective!(stats, inner.objective)
    set_dual_residual!(stats, inner.dual_feas)
    set_time!(stats, time() - start_time)
    verbose > 0 && stats.iter % verbose == 0 &&
      @info "batch $(stats.iter)  epoch $(epoch(nlp))  f = $(inner.objective)  inner: $(inner.status), $(inner.iter) its"
    if !isfinite(inner.objective)
      set_status!(stats, :exception)
    elseif stats.iter ≥ max_iter
      set_status!(stats, :max_iter)
    elseif stats.elapsed_time > max_time
      set_status!(stats, :max_time)
    end
    callback(nlp, solver, stats)
    stats.status == :unknown || break
    next_batch!(nlp)
    epoch(nlp) > max_epochs && set_status!(stats, :user)
  end

  set_time!(stats, time() - start_time)
  set_solution!(stats, xk)
  set_solver_specific!(stats, :inner_statuses, solver.inner_statuses)
  return stats
end

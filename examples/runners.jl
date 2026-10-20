# Training runners on a LuxNLPModel. STATUS: written, not executed.
#
# Every runner uses the same model, data, batch order and objective through the NLPModels API,
# logs once per epoch on the VALIDATION set (the test set is evaluated once, in run.jl), records
# oracle counts and forward/backward pass counts, excludes evaluation time from solver time,
# and returns (history::DataFrame, x_final).
#
# Minibatch modes for existing JSO solvers (checked against JSOSolvers 0.14, src/fomo.jl):
#   batchwise  `inner_iters` iterations per batch, warm-started. Ratio tests never mix batches,
#              but solver memory restarts at every call, and an epoch costs several times
#              Adam's passes: compare on pass counts or time, not on epochs.
#   callback   batch switched in the callback. This ALWAYS mixes batches: the step is built from
#              the gradient on B_k, the trial point is evaluated on B_{k+1}, ρ compares
#              f_{B_{k+1}}(c) with f_{B_k}(x_k), and the batch also advances after rejected
#              steps. A baseline only, not evidence about the solver.
#   fullbatch  deterministic; the training set is one batch (small problems or --subset).
# Solvers that follow the stochastic contract, first or second order, go through `run_contract`.
using LinearAlgebra, Printf, DataFrames
using NLPModels, SolverCore, JSOSolvers, Optimisers, Lux
using LuxNLPModels

_newhistory() = DataFrame(method = String[], epoch = Int[], train_loss = Float64[],
                          val_metric = Float64[], nobj = Int[], ngrad = Int[], nhprod = Int[],
                          nforward = Int[], nbackward = Int[], nhvp_ad = Int[],
                          solver_time = Float64[])

_metric(nlp, x, data, metric) =
  metric === :accuracy ? accuracy(nlp, x, data) : mean_objective(nlp, x, data)

"Evaluate at `x`, append one row, print it; returns the evaluation time (excluded from solver time)."
function _log!(history, method, nlp, x, val, metric, solver_time)
  t0 = time()
  recalibrate_state!(nlp, x)                 # same BatchNorm treatment for every method
  train_loss = mean_objective(nlp, x)        # test-mode loss on the whole training set
  m = _metric(nlp, x, val, metric)
  p = pass_counts(nlp)
  push!(history, (method = method, epoch = epoch(nlp), train_loss = train_loss, val_metric = m,
                  nobj = neval_obj(nlp), ngrad = neval_grad(nlp), nhprod = neval_hprod(nlp),
                  nforward = p.forward, nbackward = p.backward, nhvp_ad = p.hvp_ad,
                  solver_time = solver_time))
  @printf("%-16s epoch %3d  train %.4f  val %.4f  fwd %9d  bwd %9d  time %8.1fs\n",
          method, epoch(nlp), train_loss, m, p.forward, p.backward, solver_time)
  return time() - t0
end

_jso_solver(name::Symbol, nlp) = name === :fomo ? FomoSolver(nlp) : FoSolver(nlp)   # FoSolver = R2

# ---------------- Adam (Optimisers.jl) on the flat parameter vector ----------------
function run_adam(nlp, val, metric; epochs::Int, η = 1.0f-3, sync! = () -> nothing)
  x = copy(nlp.meta.x0)
  g = similar(x)
  state = Optimisers.setup(Optimisers.Adam(η), x)
  history = _newhistory()
  solver_time = 0.0
  _log!(history, "adam", nlp, x, val, metric, solver_time)
  e = epoch(nlp)
  while epoch(nlp) < epochs
    t0 = time()
    objgrad!(nlp, x, g)
    state, x = Optimisers.update!(state, x, g)
    next_batch!(nlp)
    sync!()                                  # GPU work is asynchronous: wait before reading time
    solver_time += time() - t0
    if epoch(nlp) > e
      e = epoch(nlp)
      _log!(history, "adam", nlp, x, val, metric, solver_time)
    end
  end
  return history, x
end

# ---------------- existing JSO first-order solvers in minibatch modes ----------------
"Mode batchwise: `inner_iters` iterations of fomo or fo/R2 per batch, warm-started."
function run_jso_batchwise(name::Symbol, nlp, val, metric; epochs::Int, inner_iters::Int = 5,
                           sync! = () -> nothing, kwargs...)
  T = eltype(nlp.meta.x0)
  solver = _jso_solver(name, nlp)
  stats = GenericExecutionStats(nlp)
  x = copy(nlp.meta.x0)
  label = "$(name)_batchwise"
  history = _newhistory()
  solver_time = 0.0
  _log!(history, label, nlp, x, val, metric, solver_time)
  e = epoch(nlp)
  while epoch(nlp) < epochs
    t0 = time()
    SolverCore.solve!(solver, nlp, stats; x = x, max_iter = inner_iters, max_time = Inf,
                      atol = zero(T), rtol = zero(T), kwargs...)
    x .= stats.solution
    next_batch!(nlp)
    sync!()
    solver_time += time() - t0
    if epoch(nlp) > e
      e = epoch(nlp)
      _log!(history, label, nlp, x, val, metric, solver_time)
    end
  end
  return history, x
end

"Mode callback: one solve!, batch switched after every iteration. Mixes batches (see top)."
function run_jso_callback(name::Symbol, nlp, val, metric; epochs::Int, sync! = () -> nothing, kwargs...)
  T = eltype(nlp.meta.x0)
  solver = _jso_solver(name, nlp)
  stats = GenericExecutionStats(nlp)
  label = "$(name)_callback"
  history = _newhistory()
  _log!(history, label, nlp, nlp.meta.x0, val, metric, 0.0)
  e = Ref(epoch(nlp))
  t_eval = Ref(0.0)
  t_start = time()
  function cb(nlp, solver, stats)
    next_batch!(nlp)
    if epoch(nlp) > e[]
      e[] = epoch(nlp)
      sync!()
      t_eval[] += _log!(history, label, nlp, solver.x, val, metric, time() - t_start - t_eval[])
    end
    epoch(nlp) >= epochs && set_status!(stats, :user)
  end
  SolverCore.solve!(solver, nlp, stats; callback = cb, max_time = Inf, atol = zero(T), rtol = zero(T),
                    kwargs...)
  return history, copy(stats.solution)
end

"""
Mode fullbatch: deterministic run of any JSO solver (fomo, R2, lbfgs, trunk, ...) on the whole
training set. One forward/backward pass covers all observations: use `--subset` on large data.
"""
function run_jso_fullbatch(solve_fn, label::String, nlp, val, metric; max_time = 600.0,
                           sync! = () -> nothing, kwargs...)
  T = eltype(nlp.meta.x0)
  use_full_batch!(nlp)
  history = _newhistory()
  t0 = time()
  stats = solve_fn(nlp; max_time = max_time, atol = T(1e-6), rtol = T(1e-6), kwargs...)
  sync!()
  solver_time = time() - t0
  _log!(history, label, nlp, stats.solution, val, metric, solver_time)
  return history, copy(stats.solution)
end

# ---------------- solvers that follow the stochastic contract ----------------
"""
    run_contract(solve_fn, label, nlp, val, metric; epochs, sync!, kwargs...)

For any solver, first or second order, that follows the stochastic contract through an
`on_accept!` hook. Requirements on `solve_fn(nlp; on_accept!, callback, max_time, atol, rtol, kwargs...)`:
it returns a `GenericExecutionStats`, calls `on_accept!(nlp)` after each accepted step and then
recomputes f and g, and exposes the iterate as `solver.x` in `callback(nlp, solver, stats)`.
The callback here only logs; the batch changes only in `on_accept!`.
"""
function run_contract(solve_fn, label::String, nlp, val, metric; epochs::Int,
                      sync! = () -> nothing, kwargs...)
  T = eltype(nlp.meta.x0)
  history = _newhistory()
  _log!(history, label, nlp, nlp.meta.x0, val, metric, 0.0)
  e = Ref(epoch(nlp))
  t_eval = Ref(0.0)
  t_start = time()
  function cb(nlp, solver, stats)
    if epoch(nlp) > e[]
      e[] = epoch(nlp)
      sync!()
      t_eval[] += _log!(history, label, nlp, solver.x, val, metric, time() - t_start - t_eval[])
    end
    epoch(nlp) >= epochs && set_status!(stats, :user)
  end
  stats = solve_fn(nlp; callback = cb, max_time = Inf, atol = zero(T), rtol = zero(T),
                   on_accept! = m -> (commit_state!(m); next_batch!(m)), kwargs...)
  return history, copy(stats.solution)
end

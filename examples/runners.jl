# Training runners. Every runner trains through the NLPModels API on a PassCounter-wrapped
# LuxNLPModel, logs once per epoch on the model's `data_test` (pass the VALIDATION split there;
# run.jl evaluates the test set once, at the end), records oracle and pass counts, excludes
# evaluation time from solver time, and returns (history::DataFrame, x_final).
#
# Minibatch modes for existing JSO solvers (checked against JSOSolvers 0.14, src/fomo.jl):
#   batchwise  `inner_iters` iterations per minibatch, warm-started. Ratio tests never mix
#              minibatches, but solver memory restarts at every call and an epoch costs several
#              times Adam's passes: compare on passes or time, not epochs.
#   callback   minibatch switched in the callback (the FluxNLPModels tutorial pattern). This
#              always mixes minibatches: the step comes from the gradient on B_k, the trial point
#              is evaluated on B_{k+1}, and the minibatch also advances after rejected steps.
#   fullbatch  deterministic; build the model with batchsize = number of observations.
# Solvers that switch minibatches only after accepted steps (on_accept!) use `run_contract`.
using LinearAlgebra, Printf, DataFrames
using NLPModels, SolverCore, JSOSolvers, Optimisers, Lux
using LuxNLPModels
include(joinpath(@__DIR__, "instrumentation.jl"))

_newhistory() = DataFrame(method = String[], epoch = Int[], train_loss = Float64[],
                          val_metric = Float64[], nobj = Int[], ngrad = Int[], nhprod = Int[],
                          nforward = Int[], nbackward = Int[], nhvp_ad = Int[],
                          solver_time = Float64[])

"Test-mode metric at `x` on `data`: accuracy, or mean loss otherwise."
evaluate_metric(nlp, x, data, metric; st = testmode_state(nlp, x)) =
  metric === :accuracy ? accuracy(nlp, x; data = data, st = st) : mean_loss(nlp, x, data; st = st)

"Evaluate at `x`, append a row, print it; returns the evaluation time."
function _log!(history, method, m::PassCounter, x, metric, solver_time)
  t0 = time()
  nlp = m.nlp
  st = testmode_state(nlp, x)                # same BatchNorm treatment for every method
  train_loss = mean_loss(nlp, x, nlp.data_train; st = st)
  val = evaluate_metric(nlp, x, nlp.data_test, metric; st = st)
  push!(history, (method = method, epoch = epoch(m), train_loss = train_loss, val_metric = val,
                  nobj = neval_obj(m), ngrad = neval_grad(m), nhprod = neval_hprod(m),
                  nforward = m.forward, nbackward = m.backward, nhvp_ad = m.hvp_ad,
                  solver_time = solver_time))
  @printf("%-16s epoch %3d  train %.4f  val %.4f  fwd %9d  bwd %9d  time %8.1fs\n",
          method, epoch(m), train_loss, val, m.forward, m.backward, solver_time)
  return time() - t0
end

_jso_solver(name::Symbol, m) = name === :fomo ? FomoSolver(m) : FoSolver(m)   # FoSolver = R2

# ---------------- Adam (Optimisers.jl) on the flat parameter vector ----------------
function run_adam(nlp, metric; epochs::Int, η = 1.0f-3, sync! = () -> nothing)
  m = PassCounter(nlp)
  x = copy(m.meta.x0)
  g = similar(x)
  state = Optimisers.setup(Optimisers.Adam(η), x)
  history = _newhistory()
  solver_time = 0.0
  _log!(history, "adam", m, x, metric, solver_time)
  e = epoch(m)
  while epoch(m) < epochs
    t0 = time()
    objgrad!(m, x, g)
    state, x = Optimisers.update!(state, x, g)
    minibatch_next_train!(m)
    sync!()                                  # GPU work is asynchronous: wait before reading time
    solver_time += time() - t0
    if epoch(m) > e
      e = epoch(m)
      _log!(history, "adam", m, x, metric, solver_time)
    end
  end
  return history, x
end

# ---------------- existing JSO first-order solvers on minibatches ----------------
"Mode batchwise: `inner_iters` iterations of fomo or fo/R2 per minibatch, warm-started."
function run_jso_batchwise(name::Symbol, nlp, metric; epochs::Int, inner_iters::Int = 5,
                           sync! = () -> nothing, kwargs...)
  m = PassCounter(nlp)
  T = eltype(m.meta.x0)
  solver = _jso_solver(name, m)
  stats = GenericExecutionStats(m)
  x = copy(m.meta.x0)
  label = "$(name)_batchwise"
  history = _newhistory()
  solver_time = 0.0
  _log!(history, label, m, x, metric, solver_time)
  e = epoch(m)
  while epoch(m) < epochs
    t0 = time()
    SolverCore.solve!(solver, m, stats; x = x, max_iter = inner_iters, max_time = Inf,
                      atol = zero(T), rtol = zero(T), kwargs...)
    x .= stats.solution
    minibatch_next_train!(m)
    sync!()
    solver_time += time() - t0
    if epoch(m) > e
      e = epoch(m)
      _log!(history, label, m, x, metric, solver_time)
    end
  end
  return history, x
end

"Mode callback: one solve!, minibatch switched after every iteration. Mixes minibatches."
function run_jso_callback(name::Symbol, nlp, metric; epochs::Int, sync! = () -> nothing, kwargs...)
  m = PassCounter(nlp)
  T = eltype(m.meta.x0)
  solver = _jso_solver(name, m)
  stats = GenericExecutionStats(m)
  label = "$(name)_callback"
  history = _newhistory()
  _log!(history, label, m, m.meta.x0, metric, 0.0)
  e = Ref(epoch(m))
  t_eval = Ref(0.0)
  t_start = time()
  function cb(m, solver, stats)
    minibatch_next_train!(m)
    if epoch(m) > e[]
      e[] = epoch(m)
      sync!()
      t_eval[] += _log!(history, label, m, solver.x, metric, time() - t_start - t_eval[])
    end
    epoch(m) >= epochs && set_status!(stats, :user)
  end
  SolverCore.solve!(solver, m, stats; callback = cb, max_time = Inf, atol = zero(T), rtol = zero(T),
                    kwargs...)
  return history, copy(stats.solution)
end

"Mode fullbatch: any JSO solver (fomo, R2, lbfgs, trunk, ...) on a model built with batchsize = n."
function run_jso_fullbatch(solve_fn, label::String, nlp, metric; max_time = 600.0,
                           sync! = () -> nothing, kwargs...)
  m = PassCounter(nlp)
  T = eltype(m.meta.x0)
  history = _newhistory()
  t0 = time()
  stats = solve_fn(m; max_time = max_time, atol = T(1e-6), rtol = T(1e-6), kwargs...)
  sync!()
  _log!(history, label, m, stats.solution, metric, time() - t0)
  return history, copy(stats.solution)
end

# ---------------- solvers that switch minibatches only after accepted steps ----------------
"""
    run_contract(solve_fn, label, nlp, metric; epochs, sync!, kwargs...)

For any solver, first or second order, that takes an `on_accept!` hook:
`solve_fn(m; on_accept!, callback, max_time, atol, rtol, kwargs...)` must return a
`GenericExecutionStats`, call `on_accept!(m)` after each accepted step and then recompute f and
g, and expose the iterate as `solver.x`. The callback here only logs.
"""
function run_contract(solve_fn, label::String, nlp, metric; epochs::Int,
                      sync! = () -> nothing, kwargs...)
  m = PassCounter(nlp)
  T = eltype(m.meta.x0)
  history = _newhistory()
  _log!(history, label, m, m.meta.x0, metric, 0.0)
  e = Ref(epoch(m))
  t_eval = Ref(0.0)
  t_start = time()
  function cb(m, solver, stats)
    if epoch(m) > e[]
      e[] = epoch(m)
      sync!()
      t_eval[] += _log!(history, label, m, solver.x, metric, time() - t_start - t_eval[])
    end
    epoch(m) >= epochs && set_status!(stats, :user)
  end
  stats = solve_fn(m; callback = cb, max_time = Inf, atol = zero(T), rtol = zero(T),
                   on_accept! = minibatch_next_train!, kwargs...)
  return history, copy(stats.solution)
end

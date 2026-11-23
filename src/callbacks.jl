# Callbacks with the JSOSolvers signature `callback(nlp, solver, stats)`.
# The same callbacks are used by `train_optimisers`, so JSO solvers and Optimisers.jl rules
# produce identical logs.

"""
    TrainingLog()

Per-iteration record filled by [`logging_callback`](@ref). All fields are vectors of equal length:

- `iter`, `time` (seconds from solver start, *excluding* time spent in logging/metrics),
- `objective` (value reported by the solver, i.e. on the current batch), `gnorm`,
- `full_objective` (loss on the full training data; `NaN` when not evaluated),
- `metric` (user-defined, e.g. test accuracy or PINN relative L2 error; `nothing` when not evaluated),
- `neval_obj`, `neval_grad`, `neval_hprod` (NLPModels counters),
- `neval_residual`, `neval_jprod_residual`, `neval_jtprod_residual` (NLS counters; 0 for NLP models),
- `forward`, `backward`, `jvp`, `hvp`, `epochs` (see [`work`](@ref)).
"""
Base.@kwdef mutable struct TrainingLog
  iter::Vector{Int} = Int[]
  time::Vector{Float64} = Float64[]
  objective::Vector{Float64} = Float64[]
  gnorm::Vector{Float64} = Float64[]
  full_objective::Vector{Float64} = Float64[]
  metric::Vector{Any} = Any[]
  neval_obj::Vector{Int} = Int[]
  neval_grad::Vector{Int} = Int[]
  neval_hprod::Vector{Int} = Int[]
  neval_residual::Vector{Int} = Int[]
  neval_jprod_residual::Vector{Int} = Int[]
  neval_jtprod_residual::Vector{Int} = Int[]
  forward::Vector{Int} = Int[]
  backward::Vector{Int} = Int[]
  jvp::Vector{Int} = Int[]
  hvp::Vector{Int} = Int[]
  epochs::Vector{Float64} = Float64[]
  overhead::Float64 = 0.0
end

Base.length(log::TrainingLog) = length(log.iter)

function Base.show(io::IO, log::TrainingLog)
  n = length(log)
  print(io, "TrainingLog with ", n, " records")
  n > 0 && print(io, ", last: iter = ", log.iter[end], ", objective = ", log.objective[end],
                 ", epochs = ", round(log.epochs[end]; digits = 3))
end

"""
    logging_callback(log::TrainingLog; every = 1, full = false, metric = nothing)

Return a callback `(nlp, solver, stats) -> nothing` that appends to `log` every `every`
iterations (and always at iteration 0).

- `full = true` also records [`full_objective`](@ref) on the training data (not counted in
  the NLPModels counters nor in the reported time).
- `metric = (nlp, x) -> value` records any additional quantity (test accuracy, PINN error, ...).

The iterate is read from `solver.x`, which all JSOSolvers solvers and `train_optimisers` provide.
"""
function logging_callback(log::TrainingLog; every::Integer = 1, full::Bool = false, metric = nothing)
  return function (nlp, solver, stats)
    (stats.iter == 0 || stats.iter % every == 0) || return nothing
    t_start = time()
    x = solver.x
    push!(log.iter, stats.iter)
    push!(log.objective, Float64(stats.objective))
    push!(log.gnorm, Float64(stats.dual_feas))
    push!(log.full_objective, full ? Float64(full_objective(nlp, x)) : NaN)
    push!(log.metric, metric === nothing ? nothing : metric(nlp, x))
    push!(log.neval_obj, neval_obj(nlp))
    push!(log.neval_grad, neval_grad(nlp))
    push!(log.neval_hprod, neval_hprod(nlp))
    nls = nlp isa AbstractNLSModel
    push!(log.neval_residual, nls ? neval_residual(nlp) : 0)
    push!(log.neval_jprod_residual, nls ? neval_jprod_residual(nlp) : 0)
    push!(log.neval_jtprod_residual, nls ? neval_jtprod_residual(nlp) : 0)
    w = work(nlp)
    push!(log.forward, w.forward)
    push!(log.backward, w.backward)
    push!(log.jvp, w.jvp)
    push!(log.hvp, w.hvp)
    push!(log.epochs, w.epochs)
    # The solver's clock includes the time spent in previous callbacks: subtract it.
    push!(log.time, stats.elapsed_time - log.overhead)
    log.overhead += time() - t_start
    return nothing
  end
end

"""
    minibatch_callback(; every = 1, max_epochs = Inf)

Return a callback that advances `nlp` to the next minibatch every `every` iterations and
stops the solver (status `:user`) once a batch of epoch `max_epochs + 1` would be used.

!!! warning
    JSO solvers assume a deterministic objective. After a batch switch, the next iteration
    compares objective values computed on different batches (e.g. the acceptance ratio of
    `R2`, the curvature pairs of `lbfgs`). This is a heuristic stochastic variant, not the
    method covered by the solver's convergence theory. For a deterministic comparison use
    full-batch mode (`batchsize = nothing`).
"""
function minibatch_callback(; every::Integer = 1, max_epochs::Real = Inf)
  return function (nlp, solver, stats)
    if stats.iter > 0 && stats.iter % every == 0
      next_batch!(nlp)
    end
    if epoch(nlp) > max_epochs && stats.status == :unknown
      set_status!(stats, :user)
    end
    return nothing
  end
end

"""
    combine_callbacks(callbacks...)

Call `callbacks` in order. Put the logging callback before the minibatch callback so that
the logged objective refers to the batch used in that iteration.
"""
combine_callbacks(cbs...) = (nlp, solver, stats) -> (foreach(cb -> cb(nlp, solver, stats), cbs); nothing)

"""
    budget_callback(; max_passes = Inf, max_backward = typemax(Int))

Return a callback that stops the solver (status `:user`) once the work spent on `nlp`
exceeds `max_passes` data passes (`work(nlp).epochs`: observations processed by forward
passes and Hessian-vector products, divided by the data set size) or `max_backward`
reverse passes. Use it to compare methods at equal computational budget rather than at
equal iteration count, which favours methods with expensive iterations.
"""
function budget_callback(; max_passes::Real = Inf, max_backward::Integer = typemax(Int))
  return function (nlp, solver, stats)
    w = work(nlp)
    if (w.epochs > max_passes || w.backward > max_backward) && stats.status == :unknown
      set_status!(stats, :user)
    end
    return nothing
  end
end

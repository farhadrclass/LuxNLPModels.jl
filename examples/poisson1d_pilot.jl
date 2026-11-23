# Differentiation-order pilot on 1D Poisson: formulation × AD route × solver.
#
#   julia --project=<env with Reactant, Enzyme, Lux, JSOSolvers, Optimisers> examples/poisson1d_pilot.jl
#
# Formulations / AD routes (same collocation points, λ = 1, Float64):
#   F2-closed   second-order residual, u_xx propagated in closed form; DI: ForwardDiff (Jv), Zygote (Jᵀw)
#   F2-Reactant second-order residual, u_xx by nested Enzyme; compiled with Reactant (3rd order overall)
#   F1-Lux      first-order system (u, q = u'), Lux nested AD (batched_jacobian) under Zygote/ForwardDiff
# F2-closed and F2-Reactant minimize the same objective (the standard PINN loss); F1 is a
# different objective (an extra network output and residual), compared on the solution error.
#
# Every run starts from the same initialization per seed, has the same wall-clock cap and an
# iteration cap per solver; compilation is excluded (one warm-up per formulation and solver).

const HAS_REACTANT = try
  @eval using Reactant, Enzyme
  Reactant.set_default_backend("cpu")
  true
catch
  false
end
using LuxNLPModels, Lux, ADTypes, NLPModels, JSOSolvers, Optimisers, ComponentArrays
using Random, LinearAlgebra, Statistics, Printf, DelimitedFiles
include(joinpath(@__DIR__, "poisson1d_problem.jl"))

const WIDTH, DEPTH, N_INTERIOR, LAMBDA = 20, 2, 128, 1.0
const SEEDS = 1:3
const MAX_TIME = 60.0
const z = 0.0
data = poisson_data(; n_interior = N_INTERIOR)

formulations = Any[("F2-closed", residual_taylor, 1, (;))]
HAS_REACTANT && push!(formulations, ("F2-Reactant", residual_enzyme, 1, (backend = ReactantBackend(),)))
push!(formulations, ("F1-Lux", residual_firstorder, 2, (;)))

solvers = [
  ("R2NLS+LSMR", (nls, x, it) -> R2NLS(nls; x, subsolver = LSMRSubsolver, max_iter = it, max_time = MAX_TIME, atol = z, rtol = z, Fatol = z, Frtol = z), 300),
  ("trunk-NLS", (nls, x, it) -> trunk(nls; x, max_iter = it, max_time = MAX_TIME, atol = z, rtol = z), 300),
  ("L-BFGS", (nls, x, it) -> lbfgs(nls; x, max_iter = it, max_time = MAX_TIME, atol = z, rtol = z), 3_000),
  ("Adam(1e-3)", (nls, x, it) -> train_optimisers(nls, Adam(1e-3); x, max_iter = it, max_time = MAX_TIME, atol = z, rtol = z), 20_000),
]

x0_for(model, seed) = copy(getdata(ComponentArray(f64(first(Lux.setup(Xoshiro(seed), model))))))

rows = Any[["formulation" "solver" "seed" "rel_l2" "objective" "status" "iter" "time_s" "neval_residual" "neval_jprod" "neval_jtprod" "neval_grad" "neval_obj"]]
summary = Dict{Tuple{String, String}, Vector{NamedTuple}}()
for (fname, R, nout, kw) in formulations
  model = mlp(WIDTH, DEPTH; nout)
  nls = LuxNLSModel(model, (m, p, s, b) -> R(m, p, s, b; λ = LAMBDA), data;
                    ps = f64(first(Lux.setup(Xoshiro(1), model))), st = last(Lux.setup(Xoshiro(1), model)), kw...)
  println("== ", fname, ": ", nls)
  for (sname, solve, maxit) in solvers
    solve(nls, x0_for(model, 0), 2)                         # warm-up / compilation
    for seed in SEEDS
      NLPModels.reset!(nls)
      stats = solve(nls, x0_for(model, seed), maxit)
      e = rel_l2(nls, stats.solution)
      r = (rel_l2 = e, objective = stats.objective, status = stats.status, iter = stats.iter, time = stats.elapsed_time)
      push!(get!(summary, (fname, sname), NamedTuple[]), r)
      push!(rows, [fname sname seed e stats.objective string(stats.status) stats.iter stats.elapsed_time neval_residual(nls) neval_jprod_residual(nls) neval_jtprod_residual(nls) neval_grad(nls) neval_obj(nls)])
      @printf "  %-11s seed %d: rel L2 = %.2e  ½‖F‖² = %.2e  %-9s iter = %5d  time = %5.1f s\n" sname seed e stats.objective stats.status stats.iter stats.elapsed_time
      flush(stdout)
    end
  end
end
writedlm(joinpath(@__DIR__, "results_poisson1d_pilot.csv"), reduce(vcat, rows), ',')

println("\nRelative L2 error of u, median [min, max] over seeds $(SEEDS); time cap $(MAX_TIME) s per run")
@printf "%-12s %-11s %-34s %-10s %-9s %s\n" "formulation" "solver" "rel. L2" "time (s)" "iter" "statuses"
for (fname, _, _, _) in formulations, (sname, _, _) in solvers
  v = summary[(fname, sname)]
  e = getfield.(v, :rel_l2)
  @printf "%-12s %-11s %.2e [%.2e, %.2e]   %-10.1f %-9d %s\n" fname sname median(e) minimum(e) maximum(e) median(getfield.(v, :time)) round(Int, median(getfield.(v, :iter))) join(unique(string.(getfield.(v, :status))), ",")
end

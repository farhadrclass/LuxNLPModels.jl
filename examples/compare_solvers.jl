# Compare JSO solvers with Optimisers.jl rules on the same LuxNLPModel.
#
#   julia --project=examples examples/compare_solvers.jl [cpu|gpu] [fullbatch|minibatch]
#
# Protocol (see README, "Comparing methods fairly"):
#   * identical model, data, initialization and loss for every method (one LuxNLPModel each);
#   * equal compute budget in data passes (`budget_callback`), plus a wall-clock cap;
#   * several initializations (seeds); results summarised by median and range;
#   * Adam/SGD learning rates chosen on a validation split, never on the test split;
#   * one warm-up run per method before timing (Julia compiles on first call);
#   * logs written to CSV for plotting and further analysis.
#
# The target is a toy regression problem; replace `make_data` and `make_model` with your task.

using LuxNLPModels, Lux, NLPModels, JSOSolvers, Optimisers, SolverCore
using Random, Statistics, Printf, DelimitedFiles

const DEVICE_NAME = length(ARGS) ≥ 1 ? ARGS[1] : "cpu"
const MODE = length(ARGS) ≥ 2 ? Symbol(ARGS[2]) : :fullbatch
const T = Float32

# JSOSolvers' R2/fomo index a small buffer stored with the iterate's array type (see README);
# `@allowscalar` permits those few scalar accesses on GPU arrays and is a no-op on the CPU.
using GPUArraysCore: @allowscalar
if DEVICE_NAME == "gpu"
  using LuxCUDA          # `] add LuxCUDA` in this environment on a CUDA machine
  const dev = gpu_device()
else
  const dev = cpu_device()
end

# ---------------------------------------------------------------- problem
function make_data(rng; n = 4096, noise = T(0.05))
  X = 2 .* rand(rng, T, 2, n) .- 1
  y = sin.(T(π) .* X[1:1, :]) .* cos.(T(π) .* X[2:2, :]) .+ noise .* randn(rng, T, 1, n)
  return X, y
end
make_model() = Chain(Dense(2 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 1))

const BUDGET_PASSES = MODE == :fullbatch ? 2_000 : 200   # data passes per run
const MAX_TIME = 120.0                                  # seconds per run (safety cap)
const SEEDS = 1:5
const BATCHSIZE = 128

data_rng = Xoshiro(2026)
Xtr, ytr = make_data(data_rng)
Xval, yval = make_data(data_rng; n = 1024)
Xte, yte = make_data(data_rng; n = 4096)
val_data = dev((Xval, yval))
test_data = dev((Xte, yte))

function make_nlp(seed)
  ps, st = Lux.setup(Xoshiro(seed), make_model())
  return LuxNLPModel(make_model(), MSELoss(), (Xtr, ytr); ps, st, device = dev,
                     batchsize = MODE == :fullbatch ? nothing : BATCHSIZE,
                     rng = Xoshiro(10_000 + seed))
end

# ---------------------------------------------------------------- methods
# Each method: (nlp, callback) -> GenericExecutionStats. The callback carries the budget and logging.
optimisers_method(rule) = (nlp, cb) -> train_optimisers(nlp, rule; callback = cb, max_time = MAX_TIME,
                                                         atol = zero(T), rtol = zero(T))
jso_fullbatch = Dict(
  "R2"    => (nlp, cb) -> @allowscalar(R2(nlp; callback = cb, max_time = MAX_TIME, max_iter = typemax(Int), atol = zero(T), rtol = zero(T))),
  "fomo"  => (nlp, cb) -> @allowscalar(fomo(nlp; callback = cb, max_time = MAX_TIME, atol = zero(T), rtol = zero(T))),
  "lbfgs" => (nlp, cb) -> lbfgs(nlp; callback = cb, max_time = MAX_TIME, atol = zero(T), rtol = zero(T)),
  "trunk" => (nlp, cb) -> trunk(nlp; callback = cb, max_time = MAX_TIME, atol = zero(T), rtol = zero(T)),
)
# Minibatch: sample-then-solve with a few inner iterations on each fixed batch.
inner_iters = 5
jso_minibatch = Dict(
  "R2×$inner_iters"    => (nlp, cb) -> train_by_batches(nlp, (nlp, x) -> @allowscalar(R2(nlp; x, max_iter = inner_iters, atol = zero(T), rtol = zero(T)));
                                                     max_epochs = Inf, max_time = MAX_TIME, callback = cb),
  "lbfgs×$inner_iters" => (nlp, cb) -> train_by_batches(nlp, (nlp, x) -> lbfgs(nlp; x, max_iter = inner_iters, atol = zero(T), rtol = zero(T));
                                                     max_epochs = Inf, max_time = MAX_TIME, callback = cb),
)
jso_methods = MODE == :fullbatch ? jso_fullbatch : jso_minibatch

function run_method(method, seed; budget = BUDGET_PASSES, every = MODE == :fullbatch ? 10 : 20)
  nlp = make_nlp(seed)
  log = TrainingLog()
  metric = (nlp, x) -> full_objective(nlp, x; data = test_data)
  cb = combine_callbacks(logging_callback(log; every, full = true, metric), budget_callback(max_passes = budget))
  stats = method(nlp, cb)
  return (; stats, log, nlp)
end

val_loss(r) = full_objective(r.nlp, r.stats.solution; data = val_data)

# ---------------------------------------------------------------- learning-rate selection
# Adam and SGD get a learning-rate grid; the rate is chosen on the validation loss at seed 0
# (not one of the evaluation seeds), with the full budget.
function tune(make_rule, grid)
  scores = map(grid) do η
    r = run_method(optimisers_method(make_rule(η)), 0)
    v = val_loss(r)
    isfinite(v) ? v : T(Inf)
  end
  best = grid[argmin(scores)]
  @printf "  %-8s grid %s -> val %s; selected η = %g\n" string(make_rule(best)) grid round.(scores; sigdigits = 3) best
  return make_rule(best)
end

println("device = $DEVICE_NAME, mode = $MODE, budget = $BUDGET_PASSES data passes, seeds = $SEEDS")
println("warming up (compilation) ...")
for m in values(jso_methods); run_method(m, 0; budget = 2); end
run_method(optimisers_method(Adam(T(1e-3))), 0; budget = 2)

println("tuning learning rates on validation data ...")
lr_grid = T[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
methods = Dict{String, Any}(jso_methods)
methods["Adam"] = optimisers_method(tune(η -> Adam(η), lr_grid))
methods["SGD"] = optimisers_method(tune(η -> Descent(η), T[1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]))

# ---------------------------------------------------------------- runs
rows = Any[["method" "seed" "iter" "time_s" "batch_objective" "train_objective" "test_objective" "data_passes" "neval_obj" "neval_grad" "neval_hprod" "forward" "backward" "hvp"]]
final = Dict{String, Vector{NamedTuple}}()
for name in sort(collect(keys(methods))), seed in SEEDS
  r = run_method(methods[name], seed)
  L = r.log
  for k in 1:length(L)
    push!(rows, [name seed L.iter[k] L.time[k] L.objective[k] L.full_objective[k] L.metric[k] L.epochs[k] L.neval_obj[k] L.neval_grad[k] L.neval_hprod[k] L.forward[k] L.backward[k] L.hvp[k]])
  end
  test = full_objective(r.nlp, r.stats.solution; data = test_data)
  push!(get!(final, name, NamedTuple[]), (; seed, test, status = r.stats.status, passes = work(r.nlp).epochs, time = r.stats.elapsed_time - r.log.overhead))
end
outfile = joinpath(@__DIR__, "results_$(DEVICE_NAME)_$(MODE).csv")
writedlm(outfile, reduce(vcat, rows), ',')

println("\nFinal test MSE over seeds $(SEEDS) (median [min, max]); budget $(BUDGET_PASSES) data passes:")
@printf "%-10s %-30s %-14s %-12s %s\n" "method" "test MSE" "data passes" "time (s)" "statuses"
for name in sort(collect(keys(final)); by = n -> median(getfield.(final[n], :test)))
  v = final[name]
  te = getfield.(v, :test)
  @printf "%-10s %.3e [%.3e, %.3e]   %-14.1f %-12.2f %s\n" name median(te) minimum(te) maximum(te) median(getfield.(v, :passes)) median(getfield.(v, :time)) join(unique(string.(getfield.(v, :status))), ",")
end
println("\nper-iteration logs: ", outfile)

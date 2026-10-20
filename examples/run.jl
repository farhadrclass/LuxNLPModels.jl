# One run: a problem from models.jl with one optimizer. STATUS: written, not executed.
#
#   julia --project=examples examples/run.jl PROBLEM OPTIMIZER [EPOCHS] [SEED] [options]
#
#   PROBLEM    mlp_mnist | lenet_mnist | lenet_fashion | resnet_cifar10 | ae_mnist
#   OPTIMIZER  adam | fomo_batchwise | r2_batchwise | fomo_callback | r2_callback | tradam
#              fomo_full | r2_full | lbfgs_full | trunk_full     (deterministic; use --subset)
#   options    --gpu  --act=relu|gelu|tanh  --pool=max|mean  --lr=1e-3 (adam)  --batchsize=128
#              --subset=N  --nval=5000  --valseed=0
#
# SEED sets the initialization and the batch order (paired across methods); the validation split
# depends only on --valseed. Validation is held out of the training data (at most 20% of it).
# Output: results/<PROBLEM>_<OPTIMIZER>_s<SEED>.csv       one row per epoch (validation metric)
#         results/<PROBLEM>_<OPTIMIZER>_s<SEED>_test.csv  test metric of the final iterate
using Random, CSV, DataFrames, Lux, LuxNLPModels, JSOSolvers
using MLDataDevices: cpu_device, gpu_device

const USE_GPU = "--gpu" in ARGS
if USE_GPU
  using CUDA, LuxCUDA
end
sync!() = USE_GPU ? CUDA.synchronize() : nothing

include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "runners.jl"))
include(joinpath(@__DIR__, "solvers", "tradam.jl"))

function option(name, default)
  i = findfirst(a -> startswith(a, "--$name="), ARGS)
  return i === nothing ? default : String(split(ARGS[i], "="; limit = 2)[2])
end
pos = filter(a -> !startswith(a, "--"), ARGS)
length(pos) >= 2 || error("usage: run.jl PROBLEM OPTIMIZER [EPOCHS] [SEED] [options]")
pname, oname = Symbol(pos[1]), pos[2]
epochs = length(pos) >= 3 ? parse(Int, pos[3]) : 5
seed = length(pos) >= 4 ? parse(Int, pos[4]) : 1
acts = Dict("relu" => relu, "gelu" => gelu, "tanh" => tanh)
pools = Dict("max" => MaxPool, "mean" => MeanPool)
act = (a = option("act", nothing)) === nothing ? nothing : acts[a]
pool = (p = option("pool", nothing)) === nothing ? nothing : pools[p]
lr = parse(Float32, option("lr", "1e-3"))
bs = parse(Int, option("batchsize", "128"))
nsub = parse(Int, option("subset", "0"))
nval = parse(Int, option("nval", "5000"))
valseed = parse(Int, option("valseed", "0"))

# ---------------- data, model ----------------
dev = USE_GPU ? gpu_device() : cpu_device()
model, train, test, objective, metric = problem(pname; act, pool)
nsub > 0 && (train = select_obs(train, 1:nsub))
train, val = split_validation(train, min(nval, count_obs(train) ÷ 5); seed = valseed)
source = ArrayBatchSource(train; batchsize = min(bs, count_obs(train)), seed = seed)
hvp_kw = USE_GPU ? (; hvp_backend = :finite_difference) : (;)
nlp = LuxNLPModel(model, source; objective, seed, device = dev, hvp_kw...)

# ---------------- optimizer ----------------
runners = Dict(
  "adam" => () -> run_adam(nlp, val, metric; epochs, η = lr, sync!),
  "fomo_batchwise" => () -> run_jso_batchwise(:fomo, nlp, val, metric; epochs, sync!),
  "r2_batchwise" => () -> run_jso_batchwise(:fo, nlp, val, metric; epochs, sync!),
  "fomo_callback" => () -> run_jso_callback(:fomo, nlp, val, metric; epochs, sync!),
  "r2_callback" => () -> run_jso_callback(:fo, nlp, val, metric; epochs, sync!),
  "tradam" => () -> run_contract(TRAdam, "tradam", nlp, val, metric; epochs, sync!, stochastic = true),
  "fomo_full" => () -> run_jso_fullbatch(fomo, "fomo_full", nlp, val, metric; sync!),
  "r2_full" => () -> run_jso_fullbatch(R2, "r2_full", nlp, val, metric; sync!),
  "lbfgs_full" => () -> run_jso_fullbatch(lbfgs, "lbfgs_full", nlp, val, metric; sync!),
  "trunk_full" => () -> run_jso_fullbatch(trunk, "trunk_full", nlp, val, metric; sync!),
)
haskey(runners, oname) || error("unknown optimizer $oname; choose from $(join(sort!(collect(keys(runners))), ", "))")
history, x = runners[oname]()

# ---------------- final test evaluation (once) ----------------
sync!()
recalibrate_state!(nlp, x)
test_metric = _metric(nlp, x, test, metric)
println("test $metric: $test_metric")

mkpath("results")
base = joinpath("results", "$(pname)_$(oname)_s$(seed)")
CSV.write(base * ".csv", history)
CSV.write(base * "_test.csv",
          DataFrame(problem = [string(pname)], method = [oname], seed = [seed], epochs = [epochs],
                    test_metric = [test_metric]))

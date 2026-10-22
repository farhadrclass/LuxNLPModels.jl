# Train LeNet-5 on MNIST with a JSOSolvers solver on minibatches.
#
#   julia --project=examples examples/lenet_mnist.jl
#   julia --project=examples examples/lenet_mnist.jl --gpu --batchsize=256 --epochs=5
#   julia --project=examples examples/lenet_mnist.jl --solver=trunk --trunk_iters=5
#
# From the REPL, edit the settings below and `include("examples/lenet_mnist.jl")`.
using Lux, LuxNLPModels, JSOSolvers, SolverCore, MLDatasets

# ---------------- settings ----------------
function option(name, default)        # `--name=value` on the command line, otherwise `default`
  i = findfirst(startswith("--$name="), ARGS)
  i === nothing && return default
  value = split(ARGS[i], "=")[2]
  return default isa String ? String(value) : parse(typeof(default), value)
end
use_gpu = "--gpu" in ARGS             # or `true`: the GPU if one works, otherwise the CPU
batchsize = option("batchsize", 128)
epochs = option("epochs", 3)
solver_name = option("solver", "R2")  # "R2" (first order) or "trunk" (second order)
trunk_iters = option("trunk_iters", 5) # trunk iterations per minibatch

if use_gpu
  using LuxCUDA                       # CUDA backend for gpu_device()
end
device = use_gpu ? gpu_device() : cpu_device()   # gpu_device() warns and falls back to the CPU
println("device: $device, solver: $solver_name, batch size: $batchsize, epochs: $epochs")

# ---------------- data: 28×28×1×N images, 10×N one-hot labels ----------------
ENV["DATADEPS_ALWAYS_ACCEPT"] = true  # download MNIST without asking
function load_mnist(split)
  d = MNIST(split)
  X = reshape(d.features, 28, 28, 1, :)
  Y = Float32.((0:9) .== permutedims(d.targets))
  return X, Y
end
train, test = load_mnist(:train), load_mnist(:test)

# ---------------- model: LeNet-5 ----------------
model = Chain(
  Conv((5, 5), 1 => 6, relu; pad = 2), MaxPool((2, 2)),
  Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)),
  FlattenLayer(),
  Dense(16 * 5 * 5 => 120, relu), Dense(120 => 84, relu), Dense(84 => 10))

nlp = LuxNLPModel(model, train; batchsize, device)   # cross-entropy loss by default

# ---------------- evaluation ----------------
# Fraction of correct predictions at x, in test mode, 1000 images at a time on the device.
# With BatchNorm, recompute its running statistics at x first: obj and grad! never update them.
function accuracy(nlp, x, (X, Y))
  ps, st = params_view(nlp, x), Lux.testmode(nlp.st)
  correct = 0
  for idx in Iterators.partition(axes(Y, 2), 1_000)
    ŷ = cpu_device()(first(nlp.model(nlp.device(X[:, :, :, idx]), ps, st)))
    correct += count(argmax(ŷ; dims = 1) .== argmax(Y[:, idx]; dims = 1))
  end
  return correct / size(Y, 2)
end
report(nlp, x) = println("epoch $(epoch(nlp))  test accuracy $(accuracy(nlp, x, test))")

# ---------------- training ----------------
if solver_name == "R2"
  # First order: move to the next minibatch in the callback, after every iteration, and report
  # when a pass over the training data ends.
  callback(nlp, solver, _) = minibatch_next_train!(nlp) || report(nlp, solver.x)
  iters_per_epoch = size(train[1], 4) ÷ batchsize
  stats = R2(nlp; callback, max_iter = epochs * iters_per_epoch, max_time = Inf,
             atol = 0.0f0, rtol = 0.0f0)   # stop on max_iter only: stochastic gradients never vanish
else
  # trunk accepts a step by comparing f before and after it, so f, ∇f and Hv must all come from
  # one minibatch: a few iterations per minibatch, warm-started, reusing one solver. Each CG
  # iteration inside trunk costs one hprod (two gradients).
  solver, stats = TrunkSolver(nlp), GenericExecutionStats(nlp)
  x = copy(nlp.meta.x0)
  while epoch(nlp) < epochs
    solve!(solver, nlp, stats; x, max_iter = trunk_iters, max_time = Inf, atol = 0.0f0,
           rtol = 0.0f0)
    x .= stats.solution
    minibatch_next_train!(nlp) || report(nlp, x)   # also: set_batchsize!(nlp, ...) to grow it
  end
end
println("status: $(stats.status), final test accuracy: $(accuracy(nlp, stats.solution, test))")

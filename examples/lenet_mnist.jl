# Train LeNet-5 on MNIST with a JSOSolvers solver, switching the minibatch at every iteration.
#
#   julia --project=examples examples/lenet_mnist.jl
#   julia --project=examples examples/lenet_mnist.jl --gpu --batchsize=256 --epochs=5
#
# From the REPL, edit the settings below and `include("examples/lenet_mnist.jl")`.
using Lux, LuxNLPModels, JSOSolvers, MLDatasets

# ---------------- settings ----------------
function option(name, default)        # `--name=value` on the command line, otherwise `default`
  i = findfirst(startswith("--$name="), ARGS)
  return i === nothing ? default : parse(typeof(default), split(ARGS[i], "=")[2])
end
use_gpu = "--gpu" in ARGS             # or `true`: the GPU if one works, otherwise the CPU
batchsize = option("batchsize", 128)
epochs = option("epochs", 3)

if use_gpu
  using LuxCUDA                       # CUDA backend for gpu_device()
end
device = use_gpu ? gpu_device() : cpu_device()   # gpu_device() warns and falls back to the CPU
println("device: $device, batch size: $batchsize, epochs: $epochs")

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

nlp = LuxNLPModel(model, train, test; batchsize, device)   # cross-entropy loss by default

# ---------------- training ----------------
# The callback runs after every iteration: move to the next minibatch, report after each epoch.
function callback(nlp, solver, _)
  if !minibatch_next_train!(nlp)      # false: a pass over the training data just ended
    println("epoch $(epoch(nlp))  test accuracy $(accuracy(nlp, solver.x))")
  end
end

iters_per_epoch = size(train[1], 4) ÷ batchsize
stats = R2(nlp; callback, max_iter = epochs * iters_per_epoch, max_time = Inf,
           atol = 0.0f0, rtol = 0.0f0)   # stop on max_iter only: stochastic gradients never vanish
println("status: $(stats.status), final test accuracy: $(accuracy(nlp, stats.solution))")

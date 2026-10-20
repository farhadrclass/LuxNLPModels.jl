# Model zoo and data for the examples. STATUS: written, not executed.
# Data are Float32 with observations along the last dimension.
# Smoothness: f is C² only with smooth activations (`act = gelu` or `tanh`) AND mean pooling
# (`pool = MeanPool`); ReLU and MaxPool are piecewise smooth. Second-order / NPC experiments that
# assume C² should use `--act=gelu --pool=mean`.
using Lux, MLDatasets, Random

# ---------------- data helpers ----------------
count_obs(A::AbstractArray) = size(A, ndims(A))
count_obs(t::Tuple) = count_obs(first(t))
select_obs(A::AbstractArray, idx) = copy(selectdim(A, ndims(A), idx))
select_obs(t::Tuple, idx) = map(a -> select_obs(a, idx), t)

"""
    split_validation(data, nval; seed = 0) -> (train, val)

Hold out `nval` random training observations for validation. Use the same `seed` for every
method so all methods are tuned and stopped on the same validation set; the test set is
evaluated once, at the end.
"""
function split_validation(data, nval::Int; seed::Int = 0)
  n = count_obs(data)
  1 <= nval < n || throw(ArgumentError("nval must be in 1:$(n - 1)"))
  perm = randperm(Xoshiro(seed), n)
  return select_obs(data, perm[(nval + 1):end]), select_obs(data, perm[1:nval])
end

function _onehot(labels, nclasses)
  Y = zeros(Float32, nclasses, length(labels))
  for (i, l) in enumerate(labels)
    Y[l + 1, i] = 1                                   # labels 0:(nclasses-1)
  end
  return Y
end

"MNIST or FashionMNIST as WHCN images (28×28×1×N) and one-hot labels."
function load_images(name::Symbol)
  D = name === :mnist ? MNIST : name === :fashion ? FashionMNIST : error("unknown $name")
  tr, te = D(:train), D(:test)                        # CHECK: MLDatasets 0.7 API
  img(d) = reshape(Float32.(d.features), 28, 28, 1, :)
  return (img(tr), _onehot(tr.targets, 10)), (img(te), _onehot(te.targets, 10))
end

"CIFAR-10 as 32×32×3×N images and one-hot labels."
function load_cifar10()
  tr, te = CIFAR10(:train), CIFAR10(:test)            # CHECK
  img(d) = Float32.(d.features)
  return (img(tr), _onehot(tr.targets, 10)), (img(te), _onehot(te.targets, 10))
end

"Flatten WHCN images to (W*H*C)×N, for MLPs and autoencoders."
flat(X) = reshape(X, :, size(X, ndims(X)))

# ---------------- architectures ----------------
"Fully connected network: input -> width -> width -> nout."
mlp(nin, nout; width = 128, act = gelu) =
  Chain(Dense(nin => width, act), Dense(width => width, act), Dense(width => nout))

"LeNet-5 for 28×28 inputs."
lenet5(; inchannels = 1, nclasses = 10, act = relu, pool = MaxPool) = Chain(
  Conv((5, 5), inchannels => 6, act; pad = 2), pool((2, 2)),
  Conv((5, 5), 6 => 16, act), pool((2, 2)),
  FlattenLayer(),
  Dense(16 * 5 * 5 => 120, act), Dense(120 => 84, act), Dense(84 => nclasses))

_resblock(c, act) = Chain(
  SkipConnection(Chain(Conv((3, 3), c => c; pad = 1), BatchNorm(c, act),
                       Conv((3, 3), c => c; pad = 1), BatchNorm(c)), +),
  WrappedFunction(x -> act.(x)))                      # CHECK: WrappedFunction signature in Lux v1

"Small ResNet for 32×32×3 inputs (BatchNorm: evaluation uses testmode_state)."
small_resnet(; nclasses = 10, width = 32, act = relu, pool = MaxPool) = Chain(
  Conv((3, 3), 3 => width; pad = 1), BatchNorm(width, act),
  _resblock(width, act), pool((2, 2)),
  Conv((3, 3), width => 2width; pad = 1), BatchNorm(2width, act),
  _resblock(2width, act), pool((2, 2)),
  GlobalMeanPool(), FlattenLayer(), Dense(2width => nclasses))

"Autoencoder on flattened inputs; train with MSELoss() and data (x, x)."
autoencoder(nin; latent = 32, width = 256, act = gelu) = Chain(
  Dense(nin => width, act), Dense(width => latent, act),
  Dense(latent => width, act), Dense(width => nin))

# ---------------- registry ----------------
"""
    problem(name; act = nothing, pool = nothing) -> (model, train, test, loss_f, metric)

`train`/`test` are data tuples; `metric` is `:accuracy` or `:mse`. `nothing` keeps the
architecture's default activation / pooling.
"""
function problem(name::Symbol; act = nothing, pool = nothing)
  if name === :mlp_mnist
    (tr, te) = load_images(:mnist)
    return mlp(784, 10; act = something(act, gelu)), (flat(tr[1]), tr[2]), (flat(te[1]), te[2]),
           CrossEntropyLoss(; logits = Val(true)), :accuracy
  elseif name === :lenet_mnist || name === :lenet_fashion
    (tr, te) = load_images(name === :lenet_mnist ? :mnist : :fashion)
    return lenet5(; act = something(act, relu), pool = something(pool, MaxPool)), tr, te,
           CrossEntropyLoss(; logits = Val(true)), :accuracy
  elseif name === :resnet_cifar10
    (tr, te) = load_cifar10()
    return small_resnet(; act = something(act, relu), pool = something(pool, MaxPool)), tr, te,
           CrossEntropyLoss(; logits = Val(true)), :accuracy
  elseif name === :ae_mnist
    (tr, te) = load_images(:mnist)
    Xtr, Xte = flat(tr[1]), flat(te[1])
    return autoencoder(784; act = something(act, gelu)), (Xtr, Xtr), (Xte, Xte), MSELoss(), :mse
  else
    error("unknown problem $name")
  end
end

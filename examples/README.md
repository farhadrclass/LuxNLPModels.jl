# Example: LeNet-5 on MNIST

`lenet_mnist.jl` loads MNIST, builds LeNet-5, wraps it in a `LuxNLPModel` and trains it on
minibatches, with `R2` (first order) or `trunk` (second order) from JSOSolvers.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'    # once (Julia 1.10: Pkg.develop(path = ".."); Pkg.instantiate())
julia --project=examples examples/lenet_mnist.jl                                  # CPU, R2
julia --project=examples examples/lenet_mnist.jl --gpu --batchsize=256 --epochs=5 # GPU if available
julia --project=examples examples/lenet_mnist.jl --solver=trunk --trunk_iters=5   # second order
```

| Setting | Default | |
|---|---|---|
| `--gpu` | off | use the GPU if one works; otherwise a warning and the CPU |
| `--batchsize=N` | 128 | minibatch size |
| `--epochs=N` | 3 | passes over the training data |
| `--solver=NAME` | `R2` | `R2`: next minibatch after every iteration; `trunk`: `trunk_iters` iterations per minibatch |
| `--trunk_iters=N` | 5 | trunk iterations on each minibatch |

The same settings are variables at the top of the script, for running it from the REPL.
Any other JSOSolvers first-order solver takes the same keywords as `R2`: replace it with `fomo`.

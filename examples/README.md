# Example: LeNet-5 on MNIST

`lenet_mnist.jl` loads MNIST, builds LeNet-5, wraps it in a `LuxNLPModel` and trains it with
`R2` from JSOSolvers, moving to the next minibatch after every iteration.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'    # once (Julia 1.10: Pkg.develop(path = ".."); Pkg.instantiate())
julia --project=examples examples/lenet_mnist.jl                                  # CPU
julia --project=examples examples/lenet_mnist.jl --gpu --batchsize=256 --epochs=5 # GPU if available
```

| Setting | Default | |
|---|---|---|
| `--gpu` | off | use the GPU if one works; otherwise a warning and the CPU |
| `--batchsize=N` | 128 | minibatch size |
| `--epochs=N` | 3 | passes over the training data |

The same settings are variables at the top of the script, for running it from the REPL.
Any other JSOSolvers first-order solver takes the same keywords: replace `R2` with `fomo`.

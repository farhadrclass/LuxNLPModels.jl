# LuxNLPModels.jl

`LuxNLPModels` exposes the training problem of a [Lux.jl](https://lux.csail.mit.edu) model as an
[`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl), so that solvers from
JuliaSmoothOptimizers (JSO), e.g. `R2`, `fomo`, `lbfgs`, `trunk` and `R2N` from JSOSolvers.jl, can train it,
and so that they can be compared with Optimisers.jl rules (`Adam`, `Descent`, ...) on *the same*
objective, data, initialization, counters and stopping rules.

It plays the role of FluxNLPModels.jl for Lux, with three differences: the objective is any loss in
Lux's training convention (so physics-informed losses fit without changes), derivatives go through
DifferentiationInterface with a configurable backend (gradients and Hessian-vector products), and the
package records implementation-independent work measures for fair comparisons.

```julia
using LuxNLPModels, Lux, JSOSolvers, Optimisers, Random

model = Chain(Dense(2 => 32, tanh), Dense(32 => 1))
X = randn(Float32, 2, 1000); y = sin.(X[1:1, :])
ps, st = Lux.setup(Xoshiro(0), model)

nlp = LuxNLPModel(model, MSELoss(), (X, y); ps, st)            # full batch, Float32, CPU
stats = trunk(nlp; max_time = 60.0)                            # any JSO solver
x = stats.solution                                             # flat parameter vector
predict(nlp, x, X)                                             # network output
unflatten(nlp, x).layer_1.weight                               # structured view, no copy

nlp = LuxNLPModel(model, MSELoss(), (X, y); ps, st)            # fresh counters
stats_adam = train_optimisers(nlp, Adam(1f-3); max_iter = 5000) # same NLPModel, same stats type
```

## The model

`LuxNLPModel(model, loss, data; kwargs...)` defines `f(x) = loss(model, ps(x), st, batch)`, where

- `x` is the flat vector behind `ComponentArray(ps)`, wrapped back without copying at each evaluation.
  It lives on the parameters' device (`Vector` on CPU, `CuVector` with `device = gpu_device()`);
- `loss(model, ps, st, batch)` follows Lux's training convention and returns a scalar or `(value, st, stats)`.
  All Lux losses (`MSELoss()`, `CrossEntropyLoss(; logits = Val(true))`, ...) work with `batch = (x, y)`;
  custom losses can use any batch object (e.g. collocation points);
- `batch` is the full `data` (default, deterministic objective) or the current minibatch (`batchsize = b`).

Implemented API: `obj`, `grad!`, `objgrad!` (one forward and one backward pass), `hprod!` (so `hess_op`), for
unconstrained problems. Main keywords: `ps`, `st`, `rng`, `device`, `batchsize`, `shuffle`, `partial`,
`backend = AutoZygote()`, `hvp_backend = SecondOrder(AutoForwardDiff(), backend)`.

Design choices to be aware of:

- **The Lux state is not updated by evaluations.** For a fixed batch the objective is a deterministic
  function of `x`, as JSO solvers assume. With dropout this means a fixed mask; with BatchNorm in
  training mode, batch statistics are used but running statistics are not accumulated. Use
  `st = Lux.testmode(st)` to train in inference mode.
- **Precision follows the parameters.** Lux initializes in Float32; JSO default tolerances are then
  `√eps(Float32) ≈ 3e-4`. Use `ps = f64(ps)` and Float64 data for derivative checks and for PINNs.
- **Counters.** NLPModels counters count calls (`objgrad!` increments both `neval_obj` and `neval_grad`).
  `work(nlp)` additionally reports forward passes, backward passes, Hessian-vector products and processed
  observations; `work(nlp).epochs` is the number of data passes. Data without a single observation
  count (e.g. PINN interior and boundary sets of different sizes) count as one observation, so one
  data pass is one evaluation on all the data.

## Comparing methods fairly

| need | tool |
|---|---|
| per-iteration record (objective, full-data loss, any metric, counters, work, time) | `TrainingLog`, `logging_callback(log; every, full, metric)` |
| equal compute budget | `budget_callback(; max_passes, max_backward)` |
| held-out evaluation without counting work | `full_objective(nlp, x; data = test_data)` |
| several callbacks | `combine_callbacks(cb1, cb2, ...)` |
| Adam/SGD driven like a JSO solver | `train_optimisers(nlp, rule; max_iter, max_eval, max_time, max_epochs, callback)` |

The callbacks have the JSOSolvers signature `(nlp, solver, stats)` and work identically for JSO solvers,
`train_optimisers` and `train_by_batches`. Logged times exclude the time spent inside logging callbacks.

`examples/compare_solvers.jl` implements a protocol that addresses the usual pitfalls:
equal budgets in data passes (not iterations, which favour methods with expensive iterations), five
initializations reported as median [min, max], learning rates of Adam/SGD chosen on a validation split
(not on the test split, and not on the evaluation seeds), a compilation warm-up before timing, and CSV
logs for plotting. Two points still need judgment:

- **Data passes undercount Hessian-vector products.** On the toy problem in the example, trunk used the
  same 2,000 data passes as lbfgs and Adam but took ~37 s instead of ~4 s. Report both an equal-work and an
  equal-time view, or weight `work(nlp).hvp` by its measured cost.
- **Toy problems saturate.** In that example trunk, lbfgs and Adam all reach the noise floor
  (test MSE ≈ σ² = 2.5e-3), so it validates the pipeline but does not separate the methods. Use the task you
  care about, report spread across seeds, and treat differences within that spread as unresolved.

## Minibatch training

JSO solvers assume a deterministic objective; this matters once the batch changes during a run.

1. **`train_optimisers`** advances the batch after every update (standard Adam/SGD epochs).
2. **`train_by_batches(nlp, (nlp, x) -> R2(nlp; x, max_iter = k); max_epochs)`** (sample-then-solve):
   each inner solve runs on one fixed batch, so acceptance tests, line searches and L-BFGS pairs are used
   as designed; solver state is not carried across batches unless the closure keeps it.
3. **`minibatch_callback(; every, max_epochs)`** switches the batch inside a running JSO solver (the
   FluxNLPModels approach). This is a heuristic. In R2 the reference objective and gradient norm are
   local variables of the solver, so after a switch the acceptance ratio compares `f` on the old batch at
   `x` with `f` on the new batch at the trial point. In our test (2,000 points, batches of 100) R2 accepted
   its last step at iteration 7, its frozen reference loss (0.063) was lower than the loss at the same `x`
   on the next five batches (0.070-0.110), it rejected 25 consecutive steps and stopped with
   `:small_step` after 1.9 epochs.

## GPU

```julia
using Lux, LuxCUDA
nlp = LuxNLPModel(model, MSELoss(), (X, y); ps, st, device = gpu_device(), batchsize = 256)
```

The package itself avoids scalar indexing. **It has not been run on a CUDA device.** GPU compatibility was
checked with JLArrays, a CPU-backed GPU array type that forbids scalar indexing (`test/runtests.jl`):

| component (JLArray, scalar indexing disallowed) | status |
|---|---|
| `obj`, `grad!`, `objgrad!`, `hprod!`, minibatch sampling | works; matches CPU |
| `train_optimisers` (Adam, Descent), `train_by_batches` | works |
| `trunk` (exact `hprod`) | works |
| `R2`, `TR`, `fomo` | need `@allowscalar` (see below); iterates then match CPU to ~2e-7 |
| `lbfgs`, `trunk` on an `LBFGSModel` | fail: LinearOperators' `LBFGSOperator` stores `Vector{Vector{T}}` and `push!` requires `Vector{T}` |

The default `hprod` backend (ForwardDiff over Zygote) runs on JLArray; its efficiency on CUDA is untested.
`NLPModelMeta` is built with `variable_bounds_analysis = false` because the bounds analysis calls `findall`
on the bound vectors, which not every GPU array type implements.

## Physics-informed losses (towards `PINNProblems`)

A PINN residual is just another `loss(model, ps, st, batch)` with collocation points in `batch`; the
NLPModel does not change. The difficulty is differentiating, with respect to the parameters, a loss that
already differentiates the network with respect to its inputs. Checks with Lux 1.32 / Zygote 0.7,
gradient vs central finite differences in Float64:

| input derivatives inside the loss | parameter gradient |
|---|---|
| `u''` by finite differences in `x` | correct (relative error 4e-7, set by the inner FD step) |
| `u'` with `Lux.batched_jacobian(sm, AutoForwardDiff(), x)` (Lux's nested-AD rule) | correct (2e-11) |
| `u''` by nesting `batched_jacobian` twice | **wrong (relative error 0.98), no error or warning** |
| `u, u', u''` propagated through the tanh MLP in closed form (`examples/pinn_poisson1d.jl`) | correct (`check_derivatives`) |

On that example (1D Poisson, 128 collocation points, tanh MLP 1-20-20-1, Float64, budget of 20,000 data
passes, 3 initializations), the relative L2 error against `sin(πx)` was, median [min, max]: lbfgs 2.7e-7
[1.5e-7, 9.5e-7], trunk 3.6e-7 [1.7e-7, 4.3e-7], Adam→lbfgs 4.7e-7 [4.2e-7, 3.5e-6], Adam(1e-3) 8.9e-5
[3.2e-5, 5.0e-4], R2 2.6e-4 [3.4e-5, 3.4e-4]. This is one small problem, three seeds and an untuned Adam
learning rate; it shows the pipeline works and is consistent with the literature below, but it is not
evidence on its own.

So **run `check_derivatives(nlp)` in Float64 on every new PINN loss** before training. The closed-form
propagation in the example is exact, Zygote-differentiable and GPU-friendly, but specific to MLPs with a
known activation; for general architectures, Enzyme or Reactant backends (`backend = AutoEnzyme()`) are
the next thing to evaluate. They are untested here.

PINN losses are sums of squared residuals. A `LuxNLSModel` (an `AbstractNLSModel` with `residual!`,
`jprod_residual!` via forward mode, `jtprod_residual!` via reverse mode) would give access to Gauss-Newton
and Levenberg-Marquardt type solvers (`trunkls`, `tronls`, `R2NLS` in JSOSolvers) using only first-order
parameter derivatives of the residuals. That avoids the third-order derivatives that `hprod` of a
second-order PDE residual requires. This is the most promising direction for the PINN comparisons:
curvature information is reported to matter for PINN training (Rathore et al., ICML 2024: Adam+L-BFGS
and a Newton-CG method outperform Adam; Müller & Zeinhofer, ICML 2023: energy natural gradient, a
Gauss-Newton-type method, reaches errors orders of magnitude below Adam).

## Issues found in JSOSolvers 0.14.9 (worth reporting upstream)

1. **`TR` without momentum is identical to `R2`.** In `fomo.jl`, `norm_d` is set once before the loop
   (line 509) and only updated in the momentum branch (line 553), so the step `α / norm_d` uses
   `‖∇f(x₀)‖` throughout. Combined with TR's initial `α` scaled by `‖∇f(x₀)‖`, R2 and TR produce
   bit-identical iterates (checked over 30 iterations).
2. **`FoSolver` / `FomoSolver` declare `o::V`** (the objective memory of length `M`) although the
   constructor builds a `Vector{T}`; with GPU arrays the buffer is converted and `obj_mem[mem_ind + 1] = ...`
   (line 450) is scalar indexing. Declaring `o::Vector{T}` would fix it.
3. **`lbfgs` is CPU-only** through LinearOperators' `LBFGSOperator` (see GPU table).

## Development status

Verified here: Julia 1.12.7, Lux 1.32.2, NLPModels 0.21.12, JSOSolvers 0.14.9, Optimisers 0.4.11,
DifferentiationInterface 0.7.22; 120 tests (`] test`) on CPU and the JLArray GPU proxy.
Not verified: CUDA/AMD devices, Enzyme/Reactant backends, NLPModels 0.22 (compat currently `0.21.12`),
Julia 1.10/1.11, models with BatchNorm/Dropout beyond the deterministic-state design above.

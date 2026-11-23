# LuxNLPModels.jl

`LuxNLPModels` exposes the training problem of a [Lux.jl](https://lux.csail.mit.edu) model to
JuliaSmoothOptimizers (JSO) solvers, in two forms:

- `LuxNLPModel <: AbstractNLPModel`, `min f(x)` for any loss (`R2`, `fomo`, `lbfgs`, `trunk`, `R2N`, ...);
- `LuxNLSModel <: AbstractNLSModel`, `min ½‖F(x)‖²` for least-squares losses such as PINN residuals,
  which gives Gauss-Newton type solvers (`R2NLS` with LSMR/LSQR/CGLS, `trunk` and `tron` on NLS models)
  the residual and Jacobian-vector products `J(x)v`, `J(x)ᵀw`.

Both can be trained with Optimisers.jl rules (`Adam`, `Descent`, ...) on *the same* objective, data,
initialization, counters and stopping rules, so that methods are compared on equal terms.
Derivatives go through DifferentiationInterface with configurable backends, or through Reactant.jl and
Enzyme (`ReactantBackend()`), which handles the higher-order nested derivatives of PINN residuals.

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

# least squares: ½‖F‖² equals the MSE when F = (ŷ - y)·√(2/m)
res(model, ps, st, (x, y)) = (first(model(x, ps, st)) .- y) .* sqrt(2f0 / size(y, 2))
nls = LuxNLSModel(model, res, (X, y); ps, st)
stats_gn = R2NLS(nls; subsolver = LSMRSubsolver)               # regularized Gauss-Newton
```

## The model

`LuxNLPModel(model, loss, data; kwargs...)` defines `f(x) = loss(model, ps(x), st, batch)`, where

- `x` is the flat vector behind `ComponentArray(ps)`, wrapped back without copying at each evaluation.
  It lives on the parameters' device (`Vector` on CPU, `CuVector` with `device = gpu_device()`);
- `loss(model, ps, st, batch)` follows Lux's training convention and returns a scalar or `(value, st, stats)`.
  All Lux losses (`MSELoss()`, `CrossEntropyLoss(; logits = Val(true))`, ...) work with `batch = (x, y)`;
  custom losses can use any batch object (e.g. collocation points);
- `batch` is the full `data` (default, deterministic objective) or the current minibatch (`batchsize = b`).

Implemented API: `obj`, `grad!`, `objgrad!` (one forward and one backward pass), `hprod!` (so `hess_op`), and
the explicit Hessian `hess_structure!` / `hess_coord!` (so `hess`), for unconstrained problems. Main keywords:
`ps`, `st`, `rng`, `device`, `batchsize`, `shuffle`, `partial`, `backend = AutoZygote()`,
`hvp_backend = SecondOrder(AutoForwardDiff(), backend)`, `max_hess_nvar = 5_000`.

Second-order solvers should use the Hessian as an operator. `R2N`'s Krylov subsolvers (`CGR2NSubsolver`, the
default, `CRR2NSubsolver`, `MinresR2NSubsolver`, `MinresQlpR2NSubsolver`) and `trunk` only call `hprod!`.

> **The explicit Hessian is not recommended for neural networks.** It is dense: `n(n+1)/2` stored entries
> and `n` Hessian-vector products per evaluation. For a 481-parameter MLP one `hess_coord!` took 0.21 s,
> about 1,700 gradients (one `hprod!` costs about 3.4 gradients). It exists only to try factorization-based
> subsolvers (`R2N` with `MA57R2NSubsolver` / `MA97R2NSubsolver`, which need HSL) on small networks as a
> reference. Above `max_hess_nvar` parameters it throws an `ArgumentError`. The HSL path is untested here
> (HSL is not available in this environment).

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

## Least-squares models

`LuxNLSModel(model, residual, data; kwargs...)` defines `F(x) = vec(residual(model, ps(x), st, batch))`
and the objective `½‖F(x)‖²`. Put weights and normalizations in the residual: with
`F = [r_interior·√(2/N_i); r_boundary·√(2λ/N_b)]`, `½‖F‖²` is the usual PINN loss `mean(r_i²) + λ·mean(r_b²)`.

| operation | implementation | used by |
|---|---|---|
| `residual!` | one forward pass | all NLS solvers |
| `jprod_residual!` (`J(x)v`) | forward mode, `fwd_backend = AutoForwardDiff()` | LSMR/LSQR/CGLS via `jac_op_residual` |
| `jtprod_residual!` (`J(x)ᵀw`) | reverse mode, `backend = AutoZygote()` | LSMR/LSQR/CGLS, gradient |
| `grad!`, `objgrad!` | one fused forward + reverse pass of `½‖F‖²` | first-order solvers, Adam |
| `hprod!` | exact Hessian-vector product of `½‖F‖²` | general NLP solvers (`R2N`, `trunk` on the NLP view) |
| `jac_coord_residual!` | dense Jacobian, `min(m, n)` passes; **not recommended** | R2NLS's default `QRMumpsSubsolver` |

Solvers checked on `LuxNLSModel`: `R2NLS` with `LSMRSubsolver`, `LSQRSubsolver`, `CGLSSubsolver` (operator
only, no explicit Jacobian) and with the default `QRMumpsSubsolver` (explicit Jacobian, guarded by
`max_jac_entries = 10_000_000`); `trunk` and `tron` on NLS models (Gauss-Newton, LSMR); `R2N`, `lbfgs`
and `train_optimisers` through the NLP view. Pass `subsolver = LSMRSubsolver` to R2NLS for networks:
the Jacobian is dense and forming it costs `min(m, n)` passes.

`check_derivatives(nls)` compares `J(x)v` with finite differences, `J(x)ᵀw` with `J(x)v` through the
exact identity `wᵀ(Jv) = (Jᵀw)ᵀv` (the *adjoint test*, at round-off level and independent of a
finite-difference step), and the fused gradient with `J(x)ᵀF(x)`. A forward/reverse disagreement, the
typical failure of an unsupported nested-AD combination, shows up in the adjoint test.

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

Build a model with `batchsize = b` (observations along the last dimension of every data array) or, for
NamedTuple data with fields of different sizes such as PINN collocation sets,
`batchsize = (interior = 64, boundary = nothing)`: each integer field is minibatched with its own
permutation, `nothing` fields are passed whole, and epochs follow the field with the most batches per
epoch. A `LuxNLSModel` has a fixed number of residuals, so its minibatches must all have the same size
(`partial = false`); `set_batch!` and `use_full_batch!` throw if a batch would change `nequ`.

Switching batches, as in FluxNLPModels:

- `minibatch_next_train!(nlp)` advances to the next batch and returns `true`, or `false` when the epoch is
  exhausted (the batch is then the first of the next epoch); `reset_minibatch_train!(nlp)` restarts at epoch 1.
  These match FluxNLPModels' semantics, so callbacks written for FluxNLPModels port directly.
- `minibatch_callback(; every, max_epochs)` switches inside a running JSO solver (e.g.
  `R2NLS(nls; subsolver = LSMRSubsolver, callback = minibatch_callback(max_epochs = 5))`).

JSO solvers assume a deterministic objective, which matters once the batch changes during a run:

1. **`train_optimisers`** advances the batch after every update (standard Adam/SGD epochs).
2. **`train_by_batches(nlp, (nlp, x) -> R2(nlp; x, max_iter = k); max_epochs)`** (sample-then-solve):
   each inner solve runs on one fixed batch, so acceptance tests, line searches and L-BFGS pairs are used
   as designed; solver state is not carried across batches unless the closure keeps it.
3. **Switching inside a running solver** (the two functions above) is a heuristic. In R2 the reference
   objective and gradient norm are local variables of the solver, so after a switch the acceptance ratio
   compares `f` on the old batch at `x` with `f` on the new batch at the trial point. In our test (2,000
   points, batches of 100) R2 accepted its last step at iteration 7, its frozen reference loss (0.063) was
   lower than the loss at the same `x` on the next five batches (0.070-0.110), it rejected 25 consecutive
   steps and stopped with `:small_step` after 1.9 epochs. Report which of the three schemes an
   experiment uses.

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

## Physics-informed losses: the differentiation-order problem

A PINN residual already contains input derivatives (`u_xx` for a second-order PDE). Its Jacobian with
respect to the weights is then a mixed derivative of order 3, and `hprod` of the loss would be of order 4.
Lux's documentation advises against nested AD beyond second order and rewrites its PINN tutorial as a
first-order system for that reason; it states that Reactant.jl supports nested AD of higher order.

**Correctness** (1D Poisson residual, tanh MLP, Float64, `check_derivatives`: `J v` against finite
differences, `Jᵀw` against `J v` through the adjoint test):

| residual formulation and AD route | `J v` | `Jᵀw` (adjoint test) |
|---|---|---|
| (a) second order, `u_xx` propagated through the MLP in closed form; ForwardDiff / Zygote, no nested AD | 3e-10 | 3e-17 |
| (b) first-order system `(u, q = u′)`, Lux `batched_jacobian` (nested AD of order 2) | 3e-10 | 3e-17 |
| (c) second order by nesting `batched_jacobian` twice | 4e-10 | **1.5e-1: wrong, no error or warning** |
| (d) second order with nested `Enzyme.gradient` (as in Lux's PINN example), `ReactantBackend()` | 1e-15 relative to (a) | 7e-16 relative to (a) |

Forward-over-forward nesting is fine; the failure in (c) is the reverse pass over doubly nested
ForwardDiff. `check_derivatives` detects it in one call.

**Cost per call** (CPU, 481 parameters, 130 residuals; (b) has 502 parameters and 258 residuals):

| route | residual | `J v` | `Jᵀw` | gradient | first compilation |
|---|---|---|---|---|---|
| (a) closed form, Zygote / ForwardDiff | 0.11 ms | 0.35 ms | 0.95 ms | 0.96 ms | none |
| (d) nested Enzyme, Reactant | 0.10 ms | 0.20 ms | 0.22 ms | 0.23 ms | ≈ 50 s |
| (b) first-order system, Lux nested AD under Zygote | 0.17 ms | 0.45 ms | 70 ms | 65 ms | none |

**Pilot** (`examples/poisson1d_pilot.jl`; −u″ = π² sin(πx) on (0, 1), u(0) = u(1) = 0; tanh MLP 1-20-20-1
(two outputs for F1); 128 uniform interior points; λ = 1; Float64; Glorot initialization, seeds 1-3;
iteration caps R2NLS+LSMR 300, trunk-NLS 300, L-BFGS 3,000, Adam 20,000; 60 s wall-clock cap per run;
compilation excluded). Relative L2 error of `u` against `sin(πx)` on 1,001 points, median [min, max]:

| formulation | R2NLS+LSMR | trunk-NLS | L-BFGS | Adam(1e-3) |
|---|---|---|---|---|
| F2-closed (a) | 1.8e-7 [6.5e-8, 1.8e-7], 16 s | 3.6e-8 [1.9e-8, 5.1e-7], 15 s | 5.2e-6 [5.0e-6, 7.7e-6], 4 s | 1.1e-4 [3.1e-5, 6.3e-4], 23 s |
| F2-Reactant (d) | 1.8e-7 [6.5e-8, 4.9e-7], 4 s | 8.0e-8 [1.3e-8, 2.5e-7], 5 s | 2.9e-6 [2.4e-6, 1.5e-5], 1 s | 2.3e-5 [9.3e-6, 9.1e-5], 5 s |
| F1-Lux (b) | 1.9e-4 [1.2e-4, 1.9e-4], 60 s cap (38 its) | 4.4e-5 [2.2e-5, 5.1e-5], 60 s cap (37 its) | 1.6e-4 [1.4e-4, 2.4e-4], 60 s cap (693 its) | 2.3e-2 [1.8e-2, 2.9e-2], 60 s cap (721 its) |

How to read it. Rows F2-closed and F2-Reactant minimize the same objective from the same initializations;
after 20 R2NLS iterations their iterates agree to a relative 1e-6 (`test/reactant_tests.jl`), then they
drift apart through round-off, as expected on a nonconvex problem, so their seed-level differences are
not meaningful. Iteration caps differ between columns, so compare formulations within a column, not
solvers across columns. Every F1 run stopped at the time cap: its result measures the cost of Lux's nested
AD under Zygote (`Jᵀw` ≈ 300× slower than Reactant), not the quality of the first-order formulation, which
was not tested at equal iteration counts or under Reactant.

**Decision.** Use the standard second-order residual with `ReactantBackend()` (nested Enzyme): it is
correct to round-off, it is the fastest route measured (3-4× faster than (a) for the Gauss-Newton
solvers), it applies to any architecture, and it minimizes the same objective as a standard JAX PINN, so
a comparison with JAX numbers isolates implementation and solver effects. Keep route (a) as an
independent cross-check (different AD systems) for MLPs. The first-order reformulation changes the
objective (an extra output and residual, and a relative weight between the two residuals), so a gap
against a second-order JAX baseline would mix formulation and implementation; it is worth studying as a
formulation, ideally under Reactant, not as a workaround.

Caveats: one 1D problem, three seeds, CPU only, untuned Adam learning rate, iteration caps rather than
equal work. Reactant on GPU is untested here, and Reactant's first compilation per model and batch shape
costs tens of seconds. The comparison with your JAX numbers is still open.

PINN losses fit the least-squares form, and Gauss-Newton type solvers reached the smallest errors here,
consistent with reports that curvature information matters for PINN training (Rathore et al., ICML 2024;
Müller & Zeinhofer, ICML 2023). Gauss-Newton only needs `J v` and `Jᵀw` (order 3 for a second-order PDE),
whereas `hprod` of the loss needs order 4.

## Reactant backend

```julia
using Reactant, Enzyme, LuxNLPModels      # loads the extension
Reactant.set_default_backend("cpu")       # or "gpu"
nls = LuxNLSModel(model, residual, data; ps, st, backend = ReactantBackend())
stats = R2NLS(nls; subsolver = LSMRSubsolver)
```

The residual (or loss), `J v` (Enzyme forward mode), `Jᵀw` and the gradient (Enzyme reverse mode) are
compiled on first use and cached per batch shape; minibatches of the same shape reuse them. The flat
parameter vector stays on the host and is copied to Reactant's device at each call. Inside the residual,
take input derivatives with `Enzyme.gradient` (see `residual_enzyme` in `examples/poisson1d_problem.jl`;
for pointwise networks `∂(Σu)/∂xᵢ = u′(xᵢ)`). Not available with this backend: `hprod!` and the
explicit Jacobian. Tests: `test/reactant_tests.jl` (19 tests, separate from `] test`).

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
DifferentiationInterface 0.7.22, Reactant 0.2.308 with Enzyme 0.13.220 (CPU); 189 tests (`] test`:
general models, least-squares models, JLArray GPU proxy) and 19 Reactant tests (`test/reactant_tests.jl`).

Not verified: CUDA/AMD devices (with or without Reactant), R2N with HSL subsolvers, NLPModels 0.22
(compat currently `0.21.12`), Julia 1.10/1.11, models with BatchNorm/Dropout beyond the
deterministic-state design above, and the agreement of the Poisson pilot with JAX results.

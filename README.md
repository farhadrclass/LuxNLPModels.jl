# LuxNLPModels.jl

Lux networks as `AbstractNLPModel`s, so JuliaSmoothOptimizers solvers, first or second order,
can train them on CPU or GPU, with minibatches, supervised or RL objectives.

**Status: written, never executed.** Run the tests first. Lines marked `CHECK` depend on package
behaviour to confirm against the installed versions. Nothing here is "ready" until the tests pass.

## Layout

```
LuxNLPModels.jl/
├── Project.toml              package deps and compat
├── README.md
├── .JuliaFormatter.toml      JSO style
├── .gitattributes  .gitignore
├── src/                      the package: models only, no solvers
│   ├── LuxNLPModels.jl       module, exports, includes
│   ├── batch_sources.jl      AbstractBatchSource, ArrayBatchSource (supervised and RL data)
│   ├── objectives.jl         classification and MSE objectives
│   ├── model.jl              LuxNLPModel, constructor, PassCounts
│   ├── stochastic_interface.jl  the contract: next_batch!, set_batch!, commit_state!, ...
│   ├── api.jl                obj, grad!, objgrad!
│   ├── hvp.jl                hprod! (finite differences or forward-over-reverse)
│   ├── evaluation.jl         recalibrate_state!, mean_objective, accuracy
│   └── rl.jl                 PPO objective
├── test/
│   ├── Project.toml
│   ├── runtests.jl           shared helpers, includes the files below
│   ├── test_batch_sources.jl
│   ├── test_api.jl
│   ├── test_hvp.jl
│   ├── test_state.jl
│   ├── test_rl.jl
│   ├── test_solvers.jl       fomo, R2, lbfgs (first order) and trunk (second order)
│   └── test_gpu.jl           opt-in: LUXNLPMODELS_TEST_GPU=true
└── examples/                 everything that is not the package
    ├── Project.toml
    ├── README.md
    ├── models.jl  runners.jl  run.jl  ppo.jl
    └── solvers/
        └── tradam.jl         solver prototype
```

To add from the JSO template (BestieTemplate.jl) before publishing: `LICENSE`, `.github/workflows`
(CI, TagBot, CompatHelper), `docs/` (Documenter).

## The stochastic contract

For any solver, first or second order:

1. f(x_k), f(x_k + s_k), g(x_k) and any Hessian-vector product at x_k use the same batch.
2. The batch changes only after an accepted step: the solver calls `commit_state!` and
   `next_batch!` (e.g. through an `on_accept!` hook), then recomputes f and g on the new batch.
3. Callbacks and evaluation code never switch the batch.

Stochastic layers (Dropout) are reseeded at every batch switch: all evaluations on one batch
use the same masks. The contract makes ratio tests compare values of the same function f_B; it
does not by itself make an accepted step decrease the full objective f.

## Changes in this version

| Issue | Fix |
|---|---|
| `nlp.hvp_backend = :finite_difference` failed (typed field) | field untyped; `set_hvp_backend!` also resets the preparation |
| `recalibrate_state!` used the current batch size (whole data set after a full-batch switch) | uses `nominal_batchsize(source)` |
| metadata advertised a dense Hessian (`nnzh = n(n+1)/2`, `hess_available = true`) | `nnzh = 0`, `hess_available = false` |
| BatchNorm state stayed Float32 when `T = Float64` | parameters and state converted with `f32`/`f64` |
| Dropout masks frozen for solvers that never commit; changed by evaluation | RNGs reseeded at batch switches; evaluation restores them |
| evaluation chunks of 10 000, whole test set at once | chunks of 1 000; host-side argmax |
| `next_batch!` after a size change broke the buffers | sources return to the nominal size, AD re-prepared |
| `RolloutBuffer` duplicated `ArrayBatchSource` | one source; `replace_data!` for new rollouts |
| no cost measure beyond oracle counts | `pass_counts(nlp)`: forward and backward passes |
| examples tuned and logged on the test set | validation split; test evaluated once at the end |
| TRAdam: about eight host synchronizations per iteration | two fused reductions + `obj` |

## GPU

Designed for GPU, untested. Parameters, workspaces and the current batch live on `device`; data
stay on the host and one batch is copied per switch (keeping the data on the device is a
possible improvement). Use `hvp_backend = :finite_difference` on GPU until forward-over-reverse
has been tested. In Float32, ratio tests lose meaning when the predicted decrease falls below
about `1e-7 |f|`.

## Setup

1. Julia ≥ 1.10.
2. `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
3. Examples: see `examples/README.md`.

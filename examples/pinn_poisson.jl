# Physics-informed neural network (PINN) for a 1-D Poisson problem, trained two ways from the
# same initial point and compared:
#   1. Adam, then L-BFGS from its result: the usual PINN recipe;
#   2. trunk: a second-order trust-region method using Hessian-vector products.
#
#   julia --project=examples examples/pinn_poisson.jl
#
# Problem: -u'' = f on (0, 1), u(0) = u(1) = 0, with f chosen so that the solution is
# u(x) = sin(πx) + sin(4πx)/4.
using Lux, LuxNLPModels, NLPModels, JSOSolvers, Optimisers, Printf
using LinearAlgebra: norm

# ---------------- settings ----------------
n_points = 200                     # collocation points
adam_iters, adam_lr = 2_000, 1e-3  # Adam, then L-BFGS
lbfgs_iters = 1_000
trunk_iters = 100
max_time = 300.0                   # seconds, cap per solver call
const δ = 1e-3                     # step of the finite-difference stencil for u''

u_exact(x) = sin(π * x) + sin(4π * x) / 4
f(x) = π^2 * sin(π * x) + 4π^2 * sin(4π * x)          # f = -u_exact''

# ---------------- PINN: network, solution and loss ----------------
model = Chain(Dense(1 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 1))

# u(x) = x(1 - x) N(x) satisfies the boundary conditions exactly, so the loss is the residual of
# the equation only.
u_net(model, ps, st, x) = x .* (1 .- x) .* first(model(x, ps, st))

# Mean squared residual of -u'' = f at the collocation points x (1×N), with fx = f.(x).
# u'' is a 3-point finite difference of the network, evaluated in one forward pass: there is no
# nested AD, so grad! (Zygote) and hprod! work as for any other model, on CPU or GPU.
function pinn_loss(model, ps, st, (x, fx))
  N = size(x, 2)
  u = u_net(model, ps, st, hcat(x .- δ, x, x .+ δ))
  uxx = (u[:, 1:N] .- 2 .* u[:, (N + 1):(2N)] .+ u[:, (2N + 1):(3N)]) ./ δ^2
  return sum(abs2, uxx .+ fx) / N, st, (;)
end

X = reshape(collect(range(0, 1; length = n_points + 2)[2:(end - 1)]), 1, :)   # interior grid
# A single batch holding every point: a deterministic problem, as trunk and L-BFGS expect.
# Float64 because the stencil divides small differences by δ².
nlp = LuxNLPModel(model, (X, f.(X)); loss_f = pinn_loss, T = Float64, batchsize = n_points,
                  shuffle = false)

# Relative L2 error against u_exact on a fine grid.
xs = reshape(collect(range(0, 1; length = 1_001)), 1, :)
rel_error(x) = norm(u_net(model, params_view(nlp, x), nlp.st, xs) .- u_exact.(xs)) /
               norm(u_exact.(xs))

# ---------------- the two methods ----------------
# 1. Adam (Optimisers.jl) on the gradients of the NLPModel, then L-BFGS from its result.
function adam_lbfgs(nlp; adam_iters, lbfgs_iters)
  x, g = copy(nlp.meta.x0), similar(nlp.meta.x0)
  state = Optimisers.setup(Adam(adam_lr), x)
  for _ in 1:adam_iters
    grad!(nlp, x, g)
    state, x = Optimisers.update!(state, x, g)
  end
  stats = lbfgs(nlp; x, mem = 20, max_iter = lbfgs_iters, max_time, atol = 1e-10, rtol = 0.0)
  return stats.solution
end

# 2. trunk: each iteration of its inner CG costs one hprod, i.e. two gradients.
trunk_solve(nlp; iters) = trunk(nlp; max_iter = iters, max_time, atol = 1e-10, rtol = 0.0).solution

# ---------------- comparison ----------------
runs = [
  "Adam + L-BFGS" => nlp -> adam_lbfgs(nlp; adam_iters, lbfgs_iters),
  "trunk" => nlp -> trunk_solve(nlp; iters = trunk_iters),
]

# Compile both methods first, so that the timings measure training only.
adam_lbfgs(nlp; adam_iters = 2, lbfgs_iters = 2)
trunk_solve(nlp; iters = 2)

@printf("%-14s %10s %10s %7s %7s %7s %9s\n",
        "method", "loss", "L2 error", "obj", "grad", "hprod", "time (s)")
@printf("%-14s %10.2e %10.2e\n", "initial", obj(nlp, nlp.meta.x0), rel_error(nlp.meta.x0))
for (name, train) in runs
  NLPModels.reset!(nlp)
  t = @elapsed x = train(nlp)
  counts = (neval_obj(nlp), neval_grad(nlp), neval_hprod(nlp))
  @printf("%-14s %10.2e %10.2e %7d %7d %7d %9.1f\n", name, obj(nlp, x), rel_error(x), counts..., t)
end

# Prototype PINN problem: 1D Poisson, -u''(x) = π² sin(πx) on (0, 1), u(0) = u(1) = 0,
# exact solution u(x) = sin(πx).
#
#   julia --project=examples examples/pinn_poisson1d.jl
#
# The second derivative u_xx is obtained by propagating (h, ∂h/∂x, ∂²h/∂x²) through the tanh
# MLP layer by layer (forward Taylor mode written with array operations). The result is exact,
# differentiable by Zygote with respect to the parameters, and GPU friendly. We do *not* nest
# `Lux.batched_jacobian` twice: in our tests (Lux 1.32, Zygote 0.7) that returns wrong
# parameter gradients without warning; `check_derivatives` detects it.

using LuxNLPModels, Lux, NLPModels, JSOSolvers, Optimisers, SolverCore
using Random, Statistics, Printf, LinearAlgebra

const T = Float64            # PINN accuracy is limited by Float32 round-off (~1e-4 relative)
const WIDTH, DEPTH = 20, 2   # hidden layers

make_model() = Chain(Dense(1 => WIDTH, tanh), [Dense(WIDTH => WIDTH, tanh) for _ in 2:DEPTH]..., Dense(WIDTH => 1))

"""
    taylor2(ps, x) -> (u, u_x, u_xx)

Values and first two input derivatives of the MLP built by `make_model` (tanh hidden layers,
identity output) at the points `x` (1×N), from its parameters `ps` (ComponentArray).
"""
function taylor2(ps, x::AbstractMatrix)
  h, h1, h2 = x, one.(x), zero.(x)
  names = propertynames(ps)
  for (i, name) in enumerate(names)
    p = getproperty(ps, name)
    z = p.weight * h .+ p.bias
    z1 = p.weight * h1
    z2 = p.weight * h2
    if i < length(names)
      a = tanh.(z)
      s = 1 .- a .^ 2                       # tanh'(z);  tanh''(z) = -2 a s
      h, h1, h2 = a, s .* z1, s .* z2 .- 2 .* a .* s .* z1 .^ 2
    else
      h, h1, h2 = z, z1, z2
    end
  end
  return h, h1, h2
end

rhs(x) = T(π)^2 * sinpi(x)
exact(x) = sinpi(x)

# Lux training convention: loss(model, ps, st, batch). The batch holds collocation points.
function poisson_loss(model, ps, st, b)
  _, _, uxx = taylor2(ps, b.interior)
  ub, _, _ = taylor2(ps, b.boundary)
  return mean(abs2, uxx .+ rhs.(b.interior)) + mean(abs2, ub)
end

const N_INTERIOR = 128
data = (interior = reshape(collect(range(T(0), T(1); length = N_INTERIOR + 2))[2:end-1], 1, :),
        boundary = T[0 1])
xtest = reshape(collect(range(T(0), T(1); length = 1001)), 1, :)
uexact = exact.(xtest)

function make_nlp(seed)
  ps, st = Lux.setup(Xoshiro(seed), make_model())
  return LuxNLPModel(make_model(), poisson_loss, data; ps = f64(ps), st)
end

rel_l2(nlp, x) = norm(first(taylor2(unflatten(nlp, x), xtest)) .- uexact) / norm(uexact)

# ---------------------------------------------------------------- 1. validate the loss
nlp = make_nlp(1)
u, ux, uxx = taylor2(unflatten(nlp, nlp.meta.x0), xtest)
# compare u_x, u_xx with finite differences of the network output
hx = 1e-4
up = first(taylor2(unflatten(nlp, nlp.meta.x0), xtest .+ hx))
um = first(taylor2(unflatten(nlp, nlp.meta.x0), xtest .- hx))
@printf "input derivatives vs FD:   max|u_x - FD| = %.1e,  max|u_xx - FD| = %.1e\n" maximum(abs.(ux .- (up .- um) ./ 2hx)) maximum(abs.(uxx .- (up .- 2u .+ um) ./ hx^2))
errs = check_derivatives(nlp; ndirs = 5)
@printf "parameter derivatives:     rel. error grad = %.1e,  hprod = %.1e\n" errs.grad errs.hprod

# ---------------------------------------------------------------- 2. train
const BUDGET_PASSES = 20_000     # one pass = loss on all collocation points
const MAX_TIME = 60.0
const SEEDS = 1:3

z = zero(T)
# Each method: (nlp, budget in data passes, time cap) -> stats
methods = [
  "Adam(1e-3)"  => (nlp, B, tmax) -> train_optimisers(nlp, Adam(T(1e-3)); callback = budget_callback(max_passes = B), max_time = tmax, atol = z, rtol = z),
  "lbfgs"       => (nlp, B, tmax) -> lbfgs(nlp; callback = budget_callback(max_passes = B), max_time = tmax, atol = z, rtol = z),
  "trunk"       => (nlp, B, tmax) -> trunk(nlp; callback = budget_callback(max_passes = B), max_time = tmax, atol = z, rtol = z),
  "R2"          => (nlp, B, tmax) -> R2(nlp; callback = budget_callback(max_passes = B), max_time = tmax, atol = z, rtol = z),
  # common PINN protocol: Adam first, then L-BFGS (half of the budget each)
  "Adam→lbfgs"  => (nlp, B, tmax) -> begin
                     s = train_optimisers(nlp, Adam(T(1e-3)); callback = budget_callback(max_passes = B / 2),
                                          max_time = tmax / 2, atol = z, rtol = z)
                     lbfgs(nlp; x = s.solution, callback = budget_callback(max_passes = B), max_time = tmax / 2, atol = z, rtol = z)
                   end,
]

for (_, m) in methods  # compile with a tiny budget
  m(make_nlp(0), 5, MAX_TIME)
end

@printf "\n%-12s %-32s %-26s %-10s %-10s %s\n" "method" "rel. L2 error: median [min, max]" "final loss: median" "passes" "time (s)" "statuses"
for (name, m) in methods
  res = map(SEEDS) do seed
    nlp = make_nlp(seed)
    s = m(nlp, BUDGET_PASSES, MAX_TIME)
    (err = rel_l2(nlp, s.solution), loss = s.objective, passes = work(nlp).epochs, time = s.elapsed_time, status = s.status)
  end
  e = getfield.(res, :err)
  @printf "%-12s %.2e [%.2e, %.2e]       %.2e                   %-10.0f %-10.1f %s\n" name median(e) minimum(e) maximum(e) median(getfield.(res, :loss)) median(getfield.(res, :passes)) median(getfield.(res, :time)) join(unique(string.(getfield.(res, :status))), ",")
end
println("\nNote: `time` for Adam→lbfgs is the L-BFGS phase only; `passes` counts both phases.")

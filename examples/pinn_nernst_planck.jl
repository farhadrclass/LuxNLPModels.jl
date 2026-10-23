# Nernst-Planck equation PINN: optimizer benchmark.
#
# Adapted from the SciMLBenchmarks "Nernst-Planck Equation Physics-Informed Neural Network (PINN)
# Optimizer Benchmarks", itself adapted from "NeuralPDE: Automating Physics-Informed Neural
# Networks (PINNs) with Error Approximations" (Zubov et al., 2021, arXiv:2107.09443), which builds
# the PINN with NeuralPDE.jl. Same equations, constants, conditions, networks and iteration
# budgets; here the loss is written directly as a Lux loss, which LuxNLPModels hands to the
# solvers. Differences from the original:
# - fixed grids of collocation points instead of adaptive quadrature (QuadratureTraining);
# - derivatives by finite differences of the networks, evaluated in one forward pass;
# - JSOSolvers has no dense BFGS: L-BFGS replaces it (also after Adam); R2 and trunk are added.
#
#   julia --project=examples examples/pinn_nernst_planck.jl
#   julia --project=examples examples/pinn_nernst_planck.jl --gpu
using Lux, LuxNLPModels, NLPModels, JSOSolvers, Optimisers, Printf

# ---------------- settings ----------------
use_gpu = "--gpu" in ARGS
iters = 200                       # per method, as in the original (Adam + L-BFGS: 50 + 150)
max_time = 120.0                  # seconds, cap per method
const nt, nx = 20, 50             # collocation grid: nt times × nx positions

if use_gpu
  using LuxCUDA                   # CUDA backend for gpu_device()
end
device = use_gpu ? gpu_device() : cpu_device()

# ---------------- problem, non-dimensional, as in the original ----------------
const t_ref, x_ref, C_ref, Phi_ref = 1.0, 0.38, 0.16, 1.0  # s, dm, mol dm⁻³, V
const epsilon, F = 78.5, 96485.3415                         # permittivity, Faraday (A s mol⁻¹)
const R, Temp = 831.0, 298.0                                # kg dm² s⁻² K⁻¹ mol⁻¹, K
const z_Na, z_Cl = 1.0, -1.0
const D_Na, D_Cl = 0.89e-7, 1.36e-7                         # dm² s⁻¹
const u_Na = D_Na * abs(z_Na) * F / (R * Temp)              # mobilities
const u_Cl = D_Cl * abs(z_Cl) * F / (R * Temp)
const Pe_Na = x_ref^2 / (t_ref * D_Na)
const Pe_Cl = x_ref^2 / (t_ref * D_Cl)
const M_Na = x_ref^2 / (t_ref * Phi_ref * u_Na)
const M_Cl = x_ref^2 / (t_ref * Phi_ref * u_Cl)
const Po_1 = (epsilon * Phi_ref) / (F * x_ref * C_ref)
const t_max, x_max = 0.01 / t_ref, 0.38 / x_ref
const Na_0, Cl_0, Phi_0 = 0.16 / C_ref, 0.16 / C_ref, 4.0 / Phi_ref
const Na_anode, Na_cathode = 0.0, 2.0 * Na_0
const Cl_anode, Cl_cathode = 1.37 * Cl_0, 0.0

# For (t, x) ∈ (0, t_max) × (0, x_max):
#   Φ_xx = (z_Na Na + z_Cl Cl) / Po_1
#   Na_t = Na_xx / Pe_Na + z_Na / (|z_Na| M_Na) (Na_x Φ_x + Na Φ_xx)
#   Cl_t = Cl_xx / Pe_Cl + z_Cl / (|z_Cl| M_Cl) (Cl_x Φ_x + Cl Φ_xx)
# with Φ(t, 0) = Phi_0, Φ(t, x_max) = 0,
#      Na(0, x) = Na_0, Na(t, 0) = Na_anode, Na(t, x_max) = Na_cathode,
#      Cl(0, x) = Cl_0, Cl(t, 0) = Cl_anode, Cl(t, x_max) = Cl_cathode.

# ---------------- networks: one per unknown, as in the original ----------------
mlp() = Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
model = BranchLayer(mlp(), mlp(), mlp())    # input (t, x); outputs (Φ, Na, Cl)

# ---------------- collocation points ----------------
# One 2×M matrix of (t, x): the interior grid, then the lines t = 0, x = 0 and x = x_max. It is
# the single batch of the NLP (no shuffling); pnp_loss splits it with these counts.
grid(ts, xs) = reduce(hcat, [[t, x] for t in ts for x in xs])
const n_in, n_ic, n_bc = nt * nx, nx, nt
tb, xb = range(0, t_max; length = nt), range(0, x_max; length = nx)
points = hcat(grid(range(0, t_max; length = nt + 2)[2:(end - 1)],
                   range(0, x_max; length = nx + 2)[2:(end - 1)]),
              grid([0.0], xb), grid(tb, [0.0]), grid(tb, [x_max]))

# ---------------- loss ----------------
const δt, δx = 1e-5, 1e-3         # finite-difference steps, 1e-3 × each side of the domain
msq(r) = sum(abs2, r) / length(r)

# u at the interior points and ∂u/∂x, ∂²u/∂x², ∂u/∂t, from the five stencil blocks of u.
function derivatives(u)
  n = n_in
  u0, uxm, uxp = u[:, 1:n], u[:, (n + 1):(2n)], u[:, (2n + 1):(3n)]
  utm, utp = u[:, (3n + 1):(4n)], u[:, (4n + 1):(5n)]
  return u0, (uxp .- uxm) ./ (2δx), (uxp .- 2 .* u0 .+ uxm) ./ δx^2, (utp .- utm) ./ (2δt)
end

# u on the lines t = 0 (initial), x = 0 (anode) and x = x_max (cathode).
initial(u) = u[:, (5n_in + 1):(5n_in + n_ic)]
anode(u) = u[:, (5n_in + n_ic + 1):(5n_in + n_ic + n_bc)]
cathode(u) = u[:, (5n_in + n_ic + n_bc + 1):end]

# NeuralPDE's loss with unit weights: the mean squared residual of each equation plus the mean
# squared error of each condition. One forward pass evaluates all three networks at the
# interior points, their four stencil neighbours, and the boundary points.
function pnp_loss(model, ps, st, pts)
  t, x = pts[1:1, 1:n_in], pts[2:2, 1:n_in]
  stencil = hcat(vcat(t, x), vcat(t, x .- δx), vcat(t, x .+ δx), vcat(t .- δt, x),
                 vcat(t .+ δt, x), pts[:, (n_in + 1):end])
  (Φo, Nao, Clo), _ = model(stencil, ps, st)
  Φ, Φx, Φxx, _ = derivatives(Φo)
  Na, Nax, Naxx, Nat = derivatives(Nao)
  Cl, Clx, Clxx, Clt = derivatives(Clo)
  pde = msq(Φxx .- (z_Na .* Na .+ z_Cl .* Cl) ./ Po_1) +
        msq(Nat .- Naxx ./ Pe_Na .- z_Na / (abs(z_Na) * M_Na) .* (Nax .* Φx .+ Na .* Φxx)) +
        msq(Clt .- Clxx ./ Pe_Cl .- z_Cl / (abs(z_Cl) * M_Cl) .* (Clx .* Φx .+ Cl .* Φxx))
  bcs = msq(anode(Φo) .- Phi_0) + msq(cathode(Φo)) +
        msq(initial(Nao) .- Na_0) + msq(anode(Nao) .- Na_anode) +
        msq(cathode(Nao) .- Na_cathode) +
        msq(initial(Clo) .- Cl_0) + msq(anode(Clo) .- Cl_anode) +
        msq(cathode(Clo) .- Cl_cathode)
  return pde + bcs, st, (; pde, bcs)
end

# Float64 because the stencils divide small differences by δ².
nlp = LuxNLPModel(model, points; loss_f = pnp_loss, T = Float64, batchsize = size(points, 2),
                  shuffle = false, device)

# ---------------- methods ----------------
# Each method runs up to `n` iterations from x and calls record(f) at every iterate.

# Optimisers.jl rules (Adam, RMSProp) on the gradients of the NLPModel.
function optimisers_run(rule, nlp, x, n, record)
  g, state, t0 = similar(x), Optimisers.setup(rule, x), time()
  for _ in 1:n
    f, _ = objgrad!(nlp, x, g)
    record(f)
    state, x = Optimisers.update!(state, x, g)
    time() - t0 > max_time && break
  end
  return x
end

# JSOSolvers solvers; atol = rtol = 0 so that they stop on the budget, as the others do.
jso_run(solve, nlp, x, n, record; kwargs...) =
  solve(nlp; x, max_iter = n, max_time, atol = 0.0, rtol = 0.0,
        callback = (nlp, solver, stats) -> record(stats.objective), kwargs...).solution

runs = Pair{String, Function}[]
for η in (0.001, 0.005, 0.05)                 # the original: ADAM(), ADAM(0.005), ADAM(0.05)
  push!(runs, "Adam($η)" => (nlp, x, n, record) -> optimisers_run(Adam(η), nlp, x, n, record))
end
for η in (0.001, 0.005, 0.05)
  push!(runs, "RMSProp($η)" => (nlp, x, n, record) -> optimisers_run(RMSProp(η), nlp, x, n, record))
end
push!(runs, "R2" => (nlp, x, n, record) -> jso_run(R2, nlp, x, n, record))
push!(runs, "L-BFGS" => (nlp, x, n, record) -> jso_run(lbfgs, nlp, x, n, record; mem = 10))
push!(runs, "Adam + L-BFGS" => function (nlp, x, n, record)   # 50 + 150 for n = 200
  x = optimisers_run(Adam(0.001), nlp, x, n ÷ 4, record)
  return jso_run(lbfgs, nlp, x, n - n ÷ 4, record; mem = 10)
end)
push!(runs, "trunk" => (nlp, x, n, record) -> jso_run(trunk, nlp, x, n, record))

# ---------------- benchmark ----------------
for (_, method) in runs                       # compile everything, so timings measure training
  method(nlp, copy(nlp.meta.x0), 2, f -> nothing)
end

println("device: $device, $(nlp.meta.nvar) parameters, $(size(points, 2)) points, ",
        "initial loss $(obj(nlp, nlp.meta.x0))")
@printf("%-15s %11s %9s %7s %7s\n", "method", "final loss", "time (s)", "grad", "hprod")
histories = Dict{String, Vector{Tuple{Float64, Float64}}}()   # (time, loss) at each iterate
for (name, method) in runs
  NLPModels.reset!(nlp)
  history = Tuple{Float64, Float64}[]
  t0 = time()
  x = method(nlp, copy(nlp.meta.x0), iters, f -> push!(history, (time() - t0, f)))
  t = time() - t0
  counts = (neval_grad(nlp), neval_hprod(nlp))
  histories[name] = history
  @printf("%-15s %11.3e %9.2f %7d %7d\n", name, obj(nlp, x), t, counts...)
end

# Loss against time, as in the original's plots (with Plots.jl; the first time is 0, so shift
# it for a log axis):
#   using Plots
#   plot(; xscale = :log10, yscale = :log10, xlabel = "time (s)", ylabel = "loss")
#   for (name, h) in histories
#     plot!(max.(first.(h), 1e-3), last.(h); label = name)
#   end

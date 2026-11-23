using LuxNLPModels
using LuxNLPModels: MinibatchSampler, next_indices!, numobs, getobs
using Lux, NLPModels, NLPModelsTest, JSOSolvers, SolverCore, Optimisers
using ComponentArrays, LinearAlgebra, Random, Test
using ADTypes: AutoForwardDiff
using JLArrays, Adapt

const rng0 = Xoshiro(0)
const model = Chain(Dense(2 => 8, tanh), Dense(8 => 1))
const X64 = randn(rng0, 2, 120)
const Y64 = sin.(X64[1:1, :]) .+ 0.5 .* X64[2:2, :]
const X32, Y32 = Float32.(X64), Float32.(Y64)

function make_nlp(; T = Float64, kwargs...)
  ps, st = Lux.setup(Xoshiro(1), model)
  ps = T == Float64 ? f64(ps) : ps
  X, Y = T == Float64 ? (X64, Y64) : (X32, Y32)
  return LuxNLPModel(model, MSELoss(), (X, Y); ps, st, kwargs...)
end

@testset "LuxNLPModels" begin
  @testset "construction" begin
    ps, st = Lux.setup(Xoshiro(1), model)
    nlp = LuxNLPModel(model, MSELoss(), (X32, Y32); ps, st)
    @test nlp.meta.nvar == Lux.parameterlength(model)
    @test nlp.meta.x0 == getdata(ComponentArray(ps))
    @test eltype(nlp.meta.x0) == Float32
    @test unconstrained(nlp)
    @test unflatten(nlp, nlp.meta.x0).layer_1.weight == ps.layer_1.weight
    @test size(predict(nlp, nlp.meta.x0, X32)) == size(Y32)
    @test occursin("full batch", sprint(show, nlp))
    # parameters created by setup when not given; ComponentArray input accepted
    @test LuxNLPModel(model, MSELoss(), (X32, Y32); rng = Xoshiro(1)).meta.x0 == nlp.meta.x0
    @test LuxNLPModel(model, MSELoss(), (X32, Y32); ps = ComponentArray(ps), st).meta.x0 == nlp.meta.x0
  end

  @testset "objective and derivatives (Float64)" begin
    nlp = make_nlp()
    x = nlp.meta.x0
    yhat = predict(nlp, x, X64)
    @test obj(nlp, x) ≈ sum(abs2, yhat .- Y64) / length(Y64)
    f, g = objgrad(nlp, x)
    @test f ≈ obj(nlp, x) rtol = 1e-12   # AD forward pass may round differently
    @test g == grad(nlp, x)
    errs = check_derivatives(nlp; ndirs = 5)
    @test errs.grad < 1e-8
    @test errs.hprod < 1e-7
    v = randn(Xoshiro(2), nlp.meta.nvar)
    @test hprod(nlp, x, v; obj_weight = 2.5) ≈ 2.5 .* hprod(nlp, x, v)
    @test hprod(nlp, x, Float64[], v) ≈ hprod(nlp, x, v)
    # symmetry of the Hessian: uᵀHv = vᵀHu
    u = randn(Xoshiro(3), nlp.meta.nvar)
    @test dot(u, hprod(nlp, x, v)) ≈ dot(v, hprod(nlp, x, u)) rtol = 1e-10
    # hess_op from NLPModels is built on hprod!
    @test hess_op(nlp, x) * v ≈ hprod(nlp, x, v)
    # scalar-returning loss gives the same objective as Lux's tuple convention
    scalar_loss(m, ps, st, (xb, yb)) = sum(abs2, first(m(xb, ps, st)) .- yb) / length(yb)
    nlp2 = LuxNLPModel(model, scalar_loss, (X64, Y64); ps = unflatten(nlp, x), st = nlp.st)
    @test obj(nlp2, x) ≈ obj(nlp, x)
    @test grad(nlp2, x) ≈ grad(nlp, x)
  end

  @testset "explicit Hessian (not recommended, small models only)" begin
    nlp = make_nlp()
    n = nlp.meta.nvar
    x = nlp.meta.x0
    rows, cols = hess_structure(nlp)
    @test length(rows) == length(cols) == nlp.meta.nnzh == n * (n + 1) ÷ 2
    @test all(rows .≥ cols)
    @test allunique(zip(rows, cols))
    H = Matrix(hess(nlp, x))
    @test H ≈ H'
    for s in 1:3
      v = randn(Xoshiro(s), n)
      @test H * v ≈ hprod(nlp, x, v) rtol = 1e-10
    end
    @test hess_coord(nlp, x; obj_weight = 0.5) ≈ 0.5 .* hess_coord(nlp, x)
    @test hess_coord(nlp, x, Float64[]) ≈ hess_coord(nlp, x)
    # counters: one Hessian = n Hessian-vector products of work
    NLPModels.reset!(nlp)
    hess_coord(nlp, x)
    @test neval_hess(nlp) == 1
    @test work(nlp).hvp == n
    # closed form for linear least squares: MSE Hessian = (2/m) AᵀA with A = [X' 1]
    lin = Dense(2 => 1)
    ps, st = Lux.setup(Xoshiro(1), lin)
    nlpl = LuxNLPModel(lin, MSELoss(), (X64, Y64); ps = f64(ps), st)
    A = [X64' ones(120)]
    @test Matrix(hess(nlpl, nlpl.meta.x0)) ≈ 2 / 120 * (A' * A) rtol = 1e-12
    # size guard
    small = make_nlp(; max_hess_nvar = 10)
    @test_throws ArgumentError hess(small, small.meta.x0)
    @test_throws ArgumentError hess_structure(small)
    @test hprod(small, small.meta.x0, small.meta.x0) isa Vector   # operator form still available
  end

  @testset "NLPModelsTest dimensions" begin
    nlp = make_nlp()
    check_nlp_dimensions(nlp; exclude = [jth_hess, jth_hess_coord, jth_hprod, ghjvprod])
  end

  @testset "counters and work" begin
    nlp = make_nlp()
    x = nlp.meta.x0
    obj(nlp, x); grad(nlp, x); objgrad(nlp, x); hprod(nlp, x, x)
    @test (neval_obj(nlp), neval_grad(nlp), neval_hprod(nlp)) == (2, 2, 1)
    w = work(nlp)
    @test (w.forward, w.backward, w.hvp) == (3, 2, 1)
    @test w.samples == 4 * 120
    @test w.epochs ≈ 4.0
    check_derivatives(nlp)  # must not count
    @test neval_obj(nlp) == 2 && work(nlp).forward == 3
    NLPModels.reset!(nlp)
    @test neval_obj(nlp) == 0 && work(nlp).forward == 0
  end

  @testset "minibatching" begin
    s = MinibatchSampler(10, 3; rng = Xoshiro(0))
    first_epoch = [next_indices!(s) for _ in 1:3]
    @test all(length.(first_epoch) .== 3)
    @test allunique(reduce(vcat, first_epoch))
    @test s.epoch == 1
    next_indices!(s)
    @test s.epoch == 2   # last incomplete batch dropped
    sp = MinibatchSampler(10, 3; partial = true, shuffle = false)
    @test [next_indices!(sp) for _ in 1:4] == [1:3, 4:6, 7:9, [10]]
    @test numobs((a = rand(2, 5), b = rand(5))) == 5
    @test_throws DimensionMismatch numobs((rand(2, 5), rand(4)))
    @test getobs((rand(2, 5), collect(1:5)), [2, 4])[2] == [2, 4]

    nlp = make_nlp(; batchsize = 30, rng = Xoshiro(4))
    @test size(current_batch(nlp)[1]) == (2, 30)
    b1 = current_batch(nlp)
    x = nlp.meta.x0
    f1 = obj(nlp, x)
    next_batch!(nlp)
    @test current_batch(nlp) !== b1
    @test obj(nlp, x) != f1
    for _ in 1:3; next_batch!(nlp); end
    @test epoch(nlp) == 2
    @test full_objective(nlp, x) ≈ obj(make_nlp(), x)
    # changing batch size invalidates AD preparations and still gives correct gradients
    grad(nlp, x)
    set_batch!(nlp, (X64[:, 1:7], Y64[:, 1:7]))
    @test nlp.grad_prep === nothing
    @test grad(nlp, x) ≈ grad(LuxNLPModel(model, MSELoss(), (X64[:, 1:7], Y64[:, 1:7]); ps = unflatten(nlp, x), st = nlp.st), x)
    use_full_batch!(nlp)
    @test obj(nlp, x) ≈ full_objective(nlp, x)
    reset_batches!(nlp)  # no-op in full-batch mode
    nlp = make_nlp(; batchsize = 30)
    for _ in 1:5; next_batch!(nlp); end
    reset_batches!(nlp)
    @test epoch(nlp) == 1
    @test_throws ArgumentError make_nlp(; batchsize = 1000)
  end

  @testset "JSO solvers (full batch)" begin
    for (name, solve) in [("R2", R2), ("fomo", fomo), ("lbfgs", lbfgs), ("trunk", trunk), ("R2N", R2N)]
      nlp = make_nlp()
      f0 = obj(nlp, nlp.meta.x0)
      stats = solve(nlp; max_iter = 100, max_time = 60.0)
      @test stats.status in (:first_order, :max_iter)
      @test stats.objective < f0 / 2
      @test stats.objective ≈ obj(nlp, stats.solution)
    end
    # convex case (linear least squares): Newton-CG with exact hprod converges
    lin = Dense(2 => 1)
    ps, st = Lux.setup(Xoshiro(1), lin)
    nlp = LuxNLPModel(lin, MSELoss(), (X64, Y64); ps = f64(ps), st)
    stats = trunk(nlp; atol = 1e-10, rtol = 0.0)
    @test stats.status == :first_order
    @test stats.iter ≤ 10
    A = [X64' ones(120)]
    @test stats.solution ≈ A \ vec(Y64) rtol = 1e-8   # x = [w; b] matches least squares
  end

  @testset "Optimisers.jl baseline" begin
    nlp = make_nlp()
    stats = train_optimisers(nlp, Adam(1e-2); max_iter = 200)
    @test stats.status == :max_iter
    @test stats.iter == 201                   # same `iter > max_iter` convention as JSOSolvers
    @test neval_grad(nlp) == stats.iter + 1   # one objgrad per iteration plus the initial one
    @test work(nlp).backward == neval_grad(nlp)
    @test stats.objective < obj(nlp, nlp.meta.x0) / 2
    @test stats.solution != nlp.meta.x0
    # stops on the same rules as JSO solvers
    nlp = make_nlp()
    @test train_optimisers(nlp, Descent(0.1); max_eval = 10).status == :max_eval
    nlp = make_nlp(; batchsize = 40)
    stats = train_optimisers(nlp, Adam(1e-2); max_epochs = 2)
    @test stats.status == :user
    @test stats.iter == 6                     # 3 batches per epoch, 2 epochs: 6 updates
    @test neval_grad(nlp) == 6                # each batch used exactly once
    @test work(nlp).epochs ≈ 2.0
  end

  @testset "callbacks and logging" begin
    nlp = make_nlp()
    log = TrainingLog()
    stats = R2(nlp; max_iter = 30, callback = logging_callback(log; every = 10))
    @test log.iter == [0, 10, 20, 30]
    @test all(isnan, log.full_objective)
    nlp = make_nlp()
    log = TrainingLog()
    stats = R2(nlp; max_iter = 30, callback = logging_callback(log; full = true, metric = (nlp, x) -> norm(x)))
    @test log.iter == 0:stats.iter
    @test log.full_objective ≈ log.objective          # full batch: identical
    @test issorted(log.time) && issorted(log.neval_obj)
    @test log.metric[end] ≈ norm(stats.solution)
    # budget stopper
    nlp = make_nlp()
    stats = lbfgs(nlp; callback = budget_callback(max_passes = 20))
    @test stats.status == :user
    @test 20 < work(nlp).epochs ≤ 25
    # minibatch switching through the callback
    nlp = make_nlp(; batchsize = 40)
    log = TrainingLog()
    stats = R2(nlp; callback = combine_callbacks(logging_callback(log), minibatch_callback(max_epochs = 2)))
    @test stats.status in (:user, :small_step)
    @test epoch(nlp) ≤ 3
  end

  @testset "train_by_batches" begin
    nlp = make_nlp(; batchsize = 40)
    log = TrainingLog()
    stats = train_by_batches(nlp, (nlp, x) -> R2(nlp; x, max_iter = 5);
                             max_epochs = 3, callback = logging_callback(log; full = true))
    @test stats.status == :user
    @test stats.iter == 9                              # 3 batches × 3 epochs
    @test length(log) == 10
    @test log.full_objective[end] < log.full_objective[1]
    @test stats.solver_specific[:inner_statuses][:max_iter] == 9
    @test_throws ArgumentError train_by_batches(make_nlp(), (nlp, x) -> R2(nlp; x))
  end

  @testset "custom batch type (PINN-style loss)" begin
    # u' = cos(x), u(0) = 0 on [0, 1]; the batch is a NamedTuple of collocation points
    pinn = Chain(Dense(1 => 8, tanh), Dense(8 => 1))
    ps, st = Lux.setup(Xoshiro(0), pinn)
    function ode_loss(model, ps, st, b)
      sm = StatefulLuxLayer{true}(model, ps, st)
      ux = reshape(Lux.batched_jacobian(sm, AutoForwardDiff(), b.x), 1, :)
      return sum(abs2, ux .- cos.(b.x)) / size(b.x, 2) + abs2(first(sm(zeros(1, 1))))
    end
    data = (x = reshape(collect(range(0, 1; length = 32)), 1, :),)
    nlp = LuxNLPModel(pinn, ode_loss, data; ps = f64(ps), st)
    @test check_derivatives(nlp; hprod = false).grad < 1e-8
    stats = lbfgs(nlp; max_iter = 300)
    xt = reshape(collect(range(0, 1; length = 11)), 1, :)
    @test maximum(abs.(predict(nlp, stats.solution, xt) .- sin.(xt))) < 1e-2

    # interior and boundary sets of different sizes: no common observation count, so one
    # data pass = one evaluation on all the data, and the work budget still applies
    data2 = (interior = data.x, boundary = zeros(1, 1))
    loss2(model, ps, st, b) = ode_loss(model, ps, st, (x = b.interior,))
    nlp2 = LuxNLPModel(pinn, loss2, data2; ps = f64(ps), st)
    obj(nlp2, nlp2.meta.x0); grad(nlp2, nlp2.meta.x0)
    @test work(nlp2).epochs == 2
    @test_throws ArgumentError LuxNLPModel(pinn, loss2, data2; ps = f64(ps), st, batchsize = 4)
    stats = R2(nlp2; callback = budget_callback(max_passes = 50))
    @test stats.status == :user
    @test 50 < work(nlp2).epochs ≤ 52
  end
end

@testset "LuxNLSModel" begin
  # ½‖F‖² equals the MSE when F = (ŷ - y)·√(2/m)
  res(m, ps, st, (x, y)) = (first(m(x, ps, st)) .- y) .* sqrt(2 / size(y, 2))
  function make_nls(; kwargs...)
    ps, st = Lux.setup(Xoshiro(1), model)
    return LuxNLSModel(model, res, (X64, Y64); ps = f64(ps), st, kwargs...)
  end

  @testset "objective, derivatives, counters" begin
    nls = make_nls()
    x = nls.meta.x0
    @test nls.nls_meta.nequ == 120
    @test obj(nls, x) ≈ obj(make_nlp(), x)
    @test grad(nls, x) ≈ grad(make_nlp(), x)
    c = check_derivatives(nls; ndirs = 5, hprod = true)
    @test c.jprod < 1e-8 && c.adjoint < 1e-12 && c.grad < 1e-12 && c.hprod < 1e-7
    NLPModels.reset!(nls)
    residual(nls, x); jprod_residual(nls, x, x); jtprod_residual(nls, x, ones(120)); objgrad(nls, x)
    @test (neval_residual(nls), neval_jprod_residual(nls), neval_jtprod_residual(nls)) == (1, 1, 1)
    @test (neval_obj(nls), neval_grad(nls)) == (1, 1)
    w = work(nls)
    @test (w.forward, w.backward, w.jvp) == (3, 2, 1)
    @test w.epochs ≈ 4.0
    # jac_op from NLPModels is built on jprod/jtprod
    J = jac_op_residual(nls, x)
    v = randn(Xoshiro(5), nls.meta.nvar)
    @test J * v ≈ jprod_residual(nls, x, v)
    # explicit Jacobian (not recommended): consistent with the products, guarded
    Jd = Matrix(jac_residual(nls, x))
    @test size(Jd) == (120, nls.meta.nvar)
    @test Jd * v ≈ jprod_residual(nls, x, v) rtol = 1e-10
    @test_throws ArgumentError jac_residual(make_nls(; max_jac_entries = 100), x)
    @test full_objective(nls, x) ≈ obj(nls, x)
    @test full_residual(nls, x) ≈ residual(nls, x)
  end

  @testset "least-squares solvers" begin
    for (name, solve) in [("R2NLS+LSMR", n -> R2NLS(n; subsolver = LSMRSubsolver, max_iter = 50)),
                          ("R2NLS+CGLS", n -> R2NLS(n; subsolver = CGLSSubsolver, max_iter = 50)),
                          ("R2NLS QRMumps", n -> R2NLS(n; max_iter = 50)),
                          ("trunk NLS", n -> trunk(n; max_iter = 50)),
                          ("tron NLS", n -> tron(n; max_iter = 50)),
                          ("Adam", n -> train_optimisers(n, Adam(1e-2); max_iter = 50))]
      nls = make_nls()
      f0 = obj(nls, nls.meta.x0)
      stats = solve(nls)
      @test stats.objective < f0 / 10
    end
    nls = make_nls()
    R2NLS(nls; subsolver = LSMRSubsolver, max_iter = 20)
    @test neval_jac_residual(nls) == 0       # operator subsolver: no explicit Jacobian
    @test neval_jprod_residual(nls) > 0 && neval_jtprod_residual(nls) > 0
  end

  @testset "PINN residuals: derivative order" begin
    include(joinpath(@__DIR__, "..", "examples", "poisson1d_problem.jl"))
    data = poisson_data(; n_interior = 32)
    for (R, nout, ok) in [(residual_taylor, 1, true), (residual_firstorder, 2, true), (residual_nested, 1, false)]
      m = mlp(8, 2; nout)
      ps, st = Lux.setup(Xoshiro(1), m)
      nls = LuxNLSModel(m, R, data; ps = f64(ps), st)
      c = check_derivatives(nls; ndirs = 3)
      @test c.jprod < 1e-7                   # forward-over-forward nesting is fine
      if ok
        @test c.adjoint < 1e-12
      else
        @test c.adjoint > 1e-3               # reverse over doubly nested ForwardDiff: wrong Jᵀw
      end
    end
  end

  @testset "minibatches" begin
    # fixed number of residuals: equal-size batches
    nls = make_nls(; batchsize = 40, rng = Xoshiro(3))
    @test nls.nls_meta.nequ == 40
    x = nls.meta.x0
    r1 = residual(nls, x)
    @test minibatch_next_train!(nls) == true
    @test residual(nls, x) != r1
    @test minibatch_next_train!(nls) == true
    @test minibatch_next_train!(nls) == false     # wrapped: first batch of epoch 2
    @test epoch(nls) == 2
    reset_minibatch_train!(nls)
    @test epoch(nls) == 1
    @test_throws DimensionMismatch use_full_batch!(nls)          # 120 ≠ 40 residuals
    @test_throws DimensionMismatch set_batch!(nls, (X64[:, 1:7], Y64[:, 1:7]))
    @test_throws ArgumentError make_nls(; batchsize = 50, partial = true)
    # FluxNLPModels-style switching inside a solver
    nls = make_nls(; batchsize = 40)
    stats = R2NLS(nls; subsolver = LSMRSubsolver, callback = minibatch_callback(max_epochs = 2))
    @test stats.status in (:user, :small_step, :first_order)
    # per-field minibatches for PINN collocation sets of different sizes
    include(joinpath(@__DIR__, "..", "examples", "poisson1d_problem.jl"))
    data = poisson_data(; n_interior = 32)
    m = mlp(8, 2)
    ps, st = Lux.setup(Xoshiro(1), m)
    nls = LuxNLSModel(m, residual_taylor, data; ps = f64(ps), st,
                      batchsize = (interior = 8, boundary = nothing), rng = Xoshiro(2))
    @test nls.nls_meta.nequ == 8 + 2
    @test size(current_batch(nls).interior) == (1, 8)
    @test current_batch(nls).boundary == data.boundary
    @test nls.nobs == 32 && nls.batch_nobs == 8
    seen = Float64[]
    for _ in 1:4
      append!(seen, vec(current_batch(nls).interior)); next_batch!(nls)
    end
    @test sort(seen) == vec(data.interior)        # one epoch covers every interior point once
    @test epoch(nls) == 2
    residual(nls, nls.meta.x0)
    @test work(nls).epochs ≈ 8 / 32
    @test_throws ArgumentError LuxNLSModel(m, residual_taylor, data; ps = f64(ps), st,
                                           batchsize = (boundary = 1, interior = 8))
  end
end

# GPU-array proxy: JLArrays forbids scalar indexing like CUDA does.
@testset "JLArray backend (GPU proxy)" begin
  dev = x -> adapt(JLArray, x)
  ps, st = Lux.setup(Xoshiro(1), model)
  nlp = LuxNLPModel(model, MSELoss(), (X32, Y32); ps, st, device = dev, batchsize = 40)
  nlp_cpu = LuxNLPModel(model, MSELoss(), (X32, Y32); ps, st, batchsize = 40, rng = Xoshiro(9))
  @test nlp.meta.x0 isa JLArray
  x = nlp.meta.x0
  f, g = objgrad(nlp, x)
  @test g isa JLArray
  use_full_batch!(nlp); use_full_batch!(nlp_cpu)
  @test obj(nlp, x) ≈ obj(nlp_cpu, Array(x))
  @test Array(grad(nlp, x)) ≈ grad(nlp_cpu, Array(x)) rtol = 1e-5
  @test Array(hprod(nlp, x, x)) ≈ hprod(nlp_cpu, Array(x), Array(x)) rtol = 1e-4
  hv = hess_coord(nlp, x)
  @test hv isa JLArray
  @test Array(hv) ≈ hess_coord(nlp_cpu, Array(x)) rtol = 1e-4
  @test train_optimisers(nlp, Adam(1f-2); max_iter = 10).solution isa JLArray
  @test trunk(nlp; max_iter = 5).solution isa JLArray
  # JSOSolvers' R2/fomo index a length-M buffer stored with the iterate's array type.
  s_gpu = JLArrays.@allowscalar R2(nlp; max_iter = 20)
  s_cpu = R2(nlp_cpu; max_iter = 20)
  @test Array(s_gpu.solution) ≈ s_cpu.solution rtol = 1e-4
  nlp = LuxNLPModel(model, MSELoss(), (X32, Y32); ps, st, device = dev, batchsize = 40)
  @test next_batch!(nlp).batch[1] isa JLArray
end

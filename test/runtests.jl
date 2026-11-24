using LinearAlgebra, Random, Test
using Adapt, JLArrays, JSOSolvers, Lux, NLPModels
using LuxNLPModels

model = Chain(Dense(2 => 8, tanh), Dense(8 => 1))
X = randn(Xoshiro(0), 2, 120)
Y = sin.(X[1:1, :]) .+ X[2:2, :] .^ 2
ps = f64(Lux.initialparameters(Xoshiro(1), model))
batches(b) = [(X[:, i:(i + b - 1)], Y[:, i:(i + b - 1)]) for i = 1:b:size(X, 2)]

function fd_check(f, ∇f, w; h = 1e-6)
  v = randn(Xoshiro(2), length(w))
  fd = (f(w + h * v) - f(w - h * v)) / (2h)
  return norm(fd - ∇f(w) * v) / norm(fd)
end

@testset "LuxNLPModel" begin
  nlp = LuxNLPModel(model, [(X, Y)]; ps = ps)
  w = nlp.meta.x0
  @test unconstrained(nlp)
  @test nlp.meta.nvar == Lux.parameterlength(model)
  @test obj(nlp, w) ≈ MSELoss()(first(model(X, ps, Lux.initialstates(Xoshiro(1), model))), Y)
  f, g = objgrad(nlp, w)
  @test f ≈ obj(nlp, w)
  @test g ≈ grad(nlp, w)
  @test fd_check(w -> obj(nlp, w), w -> grad(nlp, w)', w) < 1e-7
  @test fd_check(w -> grad(nlp, w), w -> Matrix(hess(nlp, w)), w) < 1e-7
  v = randn(Xoshiro(3), nlp.meta.nvar)
  @test hprod(nlp, w, v) ≈ Matrix(hess(nlp, w)) * v
  @test hprod(nlp, w, v; obj_weight = 2.0) ≈ 2 * hprod(nlp, w, v)
  @test neval_obj(nlp) > 0 && neval_grad(nlp) > 0 && neval_hprod(nlp) > 0 && neval_hess(nlp) > 0
end

@testset "LuxNLSModel" begin
  nls = LuxNLSModel(model, [(X, Y)]; ps = ps)
  w = nls.meta.x0
  @test nls.nls_meta.nequ == 120
  @test obj(nls, w) ≈ norm(residual(nls, w))^2 / 2
  @test fd_check(w -> residual(nls, w), w -> Matrix(jac_residual(nls, w)), w) < 1e-7
  v = randn(Xoshiro(3), nls.meta.nvar)
  u = randn(Xoshiro(4), nls.nls_meta.nequ)
  @test jprod_residual(nls, w, v) ≈ Matrix(jac_residual(nls, w)) * v
  @test dot(u, jprod_residual(nls, w, v)) ≈ dot(jtprod_residual(nls, w, u), v)
  @test grad(nls, w) ≈ jtprod_residual(nls, w, residual(nls, w))
end

struct Points{A}
  x::A
end

function LuxNLPModels.local_residual(nlp::LuxNLSModel, ps, b::Points)
  ŷ, _ = nlp.model(b.x, ps, nlp.st)
  return vec(ŷ .- sin.(b.x[1:1, :]))
end

@testset "custom batch type" begin
  nls = LuxNLSModel(model, [Points(X)]; ps = ps)
  w = nls.meta.x0
  @test nls.nls_meta.nequ == 120
  @test residual(nls, w) ≈ vec(first(model(X, ps, nls.st)) .- sin.(X[1:1, :]))
  v = randn(Xoshiro(3), nls.meta.nvar)
  u = randn(Xoshiro(4), nls.nls_meta.nequ)
  @test dot(u, jprod_residual(nls, w, v)) ≈ dot(jtprod_residual(nls, w, u), v)
end

@testset "minibatches" begin
  train, test = batches(40), batches(60)
  nlp = LuxNLPModel(model, train, test; ps = ps)
  @test nlp.current_training_minibatch == train[1]
  @test nlp.current_test_minibatch == test[1]
  @test minibatch_next_train!(nlp)
  @test nlp.current_training_minibatch == train[2]
  @test minibatch_next_train!(nlp)
  @test !minibatch_next_train!(nlp)
  @test nlp.current_training_minibatch == train[1]
  @test minibatch_next_train!(nlp)
  reset_minibatch_train!(nlp)
  @test nlp.current_training_minibatch == train[1]
  @test minibatch_next_test!(nlp)
  @test !minibatch_next_test!(nlp)
  @test nlp.current_test_minibatch == test[1]
  reset_minibatch_test!(nlp)
  @test nlp.current_test_minibatch == test[1]
  nls = LuxNLSModel(model, train; ps = ps)
  @test nls.nls_meta.nequ == 40
  minibatch_next_train!(nls)
  @test residual(nls, nls.meta.x0) ≈ vec(first(model(train[2][1], ps, nls.st)) .- train[2][2])
end

@testset "accuracy" begin
  classifier = Chain(Dense(2 => 3))
  labels = Float64.((1:3) .== permutedims(rand(Xoshiro(5), 1:3, 120)))
  ps3 = f64(Lux.initialparameters(Xoshiro(1), classifier))
  nlp = LuxNLPModel(
    classifier,
    [(X, labels)];
    ps = ps3,
    loss_f = CrossEntropyLoss(; logits = Val(true)),
  )
  acc = accuracy(nlp, nlp.meta.x0)
  @test 0 ≤ acc ≤ 1
  @test accuracy(nlp, nlp.meta.x0; data_loader = [(X, labels)]) == acc
end

@testset "solvers" begin
  for solver in (R2, lbfgs, trunk, R2N)
    nlp = LuxNLPModel(model, [(X, Y)]; ps = ps)
    stats = solver(nlp; max_iter = 50)
    @test stats.objective < obj(nlp, nlp.meta.x0) / 2
  end
  for subsolver in (LSMRSubsolver, QRMumpsSubsolver)
    nls = LuxNLSModel(model, [(X, Y)]; ps = ps)
    stats = R2NLS(nls; subsolver = subsolver, max_iter = 50)
    @test stats.objective < obj(nls, nls.meta.x0) / 2
  end
  nls = LuxNLSModel(model, [(X, Y)]; ps = ps)
  @test trunk(nls; max_iter = 50).objective < obj(nls, nls.meta.x0) / 2
  nlp = LuxNLPModel(model, batches(40); ps = ps)
  stats = R2(nlp; max_iter = 30, callback = (nlp, solver, stats) -> minibatch_next_train!(nlp))
  @test stats.status == :max_iter
end

@testset "GPU arrays (JLArray)" begin
  JLArrays.allowscalar(false)
  device = x -> adapt(JLArray, x)
  nlp = LuxNLPModel(model, batches(40); ps = ps, device = device)
  nlp_cpu = LuxNLPModel(model, batches(40); ps = ps)
  w, v = nlp.meta.x0, copy(nlp.meta.x0)
  @test w isa JLArray
  @test obj(nlp, w) ≈ obj(nlp_cpu, Array(w))
  @test Array(grad(nlp, w)) ≈ grad(nlp_cpu, Array(w))
  @test Array(hprod(nlp, w, v)) ≈ hprod(nlp_cpu, Array(w), Array(v))
  @test minibatch_next_train!(nlp) && nlp.current_training_minibatch[1] isa JLArray
  nls = LuxNLSModel(model, batches(40); ps = ps, device = device)
  nls_cpu = LuxNLSModel(model, batches(40); ps = ps)
  u = adapt(JLArray, ones(40))
  @test Array(residual(nls, w)) ≈ residual(nls_cpu, Array(w))
  @test Array(jprod_residual(nls, w, v)) ≈ jprod_residual(nls_cpu, Array(w), Array(v))
  @test Array(jtprod_residual(nls, w, u)) ≈ jtprod_residual(nls_cpu, Array(w), ones(40))
  @test trunk(nls; max_iter = 5).solution isa JLArray
end

# Fit u(x) to the solution of u'(x) = cos(x) on [0, 2π] with u(0) = 0, i.e. u(x) = sin(x),
# by minimizing the residuals u'(xᵢ) - cos(xᵢ) at collocation points xᵢ and u(0).

using LuxNLPModels, Lux, NLPModels, JSOSolvers, ComponentArrays, Optimisers
using LinearAlgebra, Printf, Random

struct Collocation{A}
  x::A
end

function LuxNLPModels.local_residual(nlp::LuxNLSModel, ps, b::Collocation)
  u = StatefulLuxLayer{true}(nlp.model, ps, nlp.st)
  du = reshape(Lux.batched_jacobian(u, AutoForwardDiff(), b.x), 1, :)
  r = (du .- cos.(b.x)) ./ sqrt(size(b.x, 2))
  return vcat(vec(r), vec(u(zero(b.x[:, 1:1]))))
end

rng = Xoshiro(0)
model = Chain(Dense(1 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
ps = f64(Lux.initialparameters(rng, model))
x = reshape(collect(range(0, 2π; length = 64)), 1, :)
xtest = reshape(collect(range(0, 2π; length = 1000)), 1, :)

function relative_error(nls, w)
  u, _ = nls.model(xtest, ComponentArray(w, nls.ps_axes), nls.st)
  return norm(vec(u) - sin.(vec(xtest))) / norm(sin.(xtest))
end

# Full batch: Gauss-Newton type method on all the collocation points.
nls = LuxNLSModel(model, [Collocation(x)]; ps = ps)
stats = R2NLS(nls; subsolver = LSMRSubsolver, max_iter = 200, max_time = 300.0)
err = relative_error(nls, stats.solution)
@printf "R2NLS, full batch: relative error %.2e (%s, %d iterations)\n" err stats.status stats.iter

# Minibatches of 16 points: an iterator over `Collocation` batches,
# traversed with `minibatch_next_train!`, which returns `false` at the end of an epoch.
batches = [Collocation(x[:, i]) for i in Iterators.partition(randperm(rng, 64), 16)]
nls = LuxNLSModel(model, batches; ps = ps)
w = copy(nls.meta.x0)
g = similar(w)
state = Optimisers.setup(Adam(1e-2), w)
for epoch = 1:500
  while true
    grad!(nls, w, g)
    Optimisers.update!(state, w, g)
    minibatch_next_train!(nls) || break
  end
end
err = relative_error(nls, w)
@printf "Adam, minibatches: relative error %.2e (%d gradients)\n" err neval_grad(nls)

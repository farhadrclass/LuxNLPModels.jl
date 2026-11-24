# Train a multilayer perceptron on MNIST with R2, one minibatch per iteration, and with Adam.

using LuxNLPModels, Lux, NLPModels, JSOSolvers, Optimisers
using MLDatasets, MLUtils, OneHotArrays, Printf, Random

function getdata(T = Float32)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:]
  data_train = (MLUtils.flatten(xtrain), onehotbatch(ytrain, 0:9))
  data_test = (MLUtils.flatten(xtest), onehotbatch(ytest, 0:9))
  return data_train, data_test
end

data_train, data_test = getdata()
train_loader = DataLoader(data_train; batchsize = 128, shuffle = true)
test_loader = DataLoader(data_test; batchsize = 1000)
model = Chain(Dense(784 => 64, relu), Dense(64 => 10))
loss = CrossEntropyLoss(; logits = Val(true))
epochs = 2

# R2 with the next minibatch at each iteration, stopped after `epochs` passes over the data.
nlp = LuxNLPModel(model, train_loader, test_loader; loss_f = loss, rng = Xoshiro(0))
epoch = Ref(1)
callback = (nlp, solver, stats) -> begin
  minibatch_next_train!(nlp) || (epoch[] += 1)
  epoch[] > epochs && (stats.status = :user)
end
stats = R2(nlp; callback = callback, max_time = 600.0)
train_acc = accuracy(nlp, stats.solution; data_loader = train_loader)
test_acc = accuracy(nlp, stats.solution)
@printf "R2:   train %.3f, test %.3f accuracy (%d gradients)\n" train_acc test_acc neval_grad(nlp)
@printf "      %s after %d iterations\n" stats.status stats.iter

# Adam from the same initial point.
nlp = LuxNLPModel(model, train_loader, test_loader; loss_f = loss, rng = Xoshiro(0))
w = copy(nlp.meta.x0)
g = similar(w)
state = Optimisers.setup(Adam(1.0f-3), w)
for epoch = 1:epochs
  while true
    grad!(nlp, w, g)
    Optimisers.update!(state, w, g)
    minibatch_next_train!(nlp) || break
  end
end
train_acc = accuracy(nlp, w; data_loader = train_loader)
test_acc = accuracy(nlp, w)
@printf "Adam: train %.3f, test %.3f accuracy (%d gradients)\n" train_acc test_acc neval_grad(nlp)

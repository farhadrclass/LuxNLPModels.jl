# Objectives use Lux's training signature:
#     objective(model, ps, st, batch) -> (loss, st_new, stats::NamedTuple)
# Lux's own losses (e.g. MSELoss(), CrossEntropyLoss(; logits = Val(true))) follow the same
# signature and can be passed directly (CHECK keyword names in your Lux version).
# `mean_objective` assumes the loss is a mean over observations, as for the objectives below.

"Mean cross-entropy from logits; `y` one-hot, observations along the last dimension."
function classification_objective(model, ps, st, (x, y))
  ŷ, st_new = model(x, ps, st)
  loss = -sum(y .* logsoftmax(ŷ; dims = 1)) / size(y, ndims(y))
  return loss, st_new, NamedTuple()
end

"Mean squared error per observation. For autoencoders pass `(x, x)` as the data."
function mse_objective(model, ps, st, (x, y))
  ŷ, st_new = model(x, ps, st)
  loss = sum(abs2, ŷ .- y) / size(y, ndims(y))
  return loss, st_new, NamedTuple()
end

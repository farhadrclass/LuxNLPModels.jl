const AbstractLuxModel = Union{AbstractLuxNLPModel, AbstractLuxNLSModel}

"""
    reset_minibatch_train!(nlp)

Select the first training minibatch of `nlp`.
"""
function reset_minibatch_train!(nlp::AbstractLuxModel)
  batch, nlp.current_training_minibatch_status = iterate(nlp.training_minibatch_iterator)
  nlp.current_training_minibatch = nlp.device(batch)
  return nlp
end

"""
    minibatch_next_train!(nlp)

Select the next training minibatch of `nlp`.
Return `false` at the end of the iterator, after selecting its first minibatch again,
and `true` otherwise.
"""
function minibatch_next_train!(nlp::AbstractLuxModel)
  next = iterate(nlp.training_minibatch_iterator, nlp.current_training_minibatch_status)
  if next === nothing
    reset_minibatch_train!(nlp)
    return false
  end
  batch, nlp.current_training_minibatch_status = next
  nlp.current_training_minibatch = nlp.device(batch)
  return true
end

"""
    reset_minibatch_test!(nlp)

Select the first test minibatch of `nlp`.
"""
function reset_minibatch_test!(nlp::AbstractLuxModel)
  batch, nlp.current_test_minibatch_status = iterate(nlp.test_minibatch_iterator)
  nlp.current_test_minibatch = nlp.device(batch)
  return nlp
end

"""
    minibatch_next_test!(nlp)

Select the next test minibatch of `nlp`.
Return `false` at the end of the iterator, after selecting its first minibatch again,
and `true` otherwise.
"""
function minibatch_next_test!(nlp::AbstractLuxModel)
  next = iterate(nlp.test_minibatch_iterator, nlp.current_test_minibatch_status)
  if next === nothing
    reset_minibatch_test!(nlp)
    return false
  end
  batch, nlp.current_test_minibatch_status = next
  nlp.current_test_minibatch = nlp.device(batch)
  return true
end

"""
    accuracy(nlp, w; data_loader = nlp.test_minibatch_iterator)

Fraction of the samples of `data_loader` whose largest output matches the largest entry of
the (one-hot) label.
"""
function accuracy(
  nlp::AbstractLuxModel,
  w::AbstractVector;
  data_loader = nlp.test_minibatch_iterator,
)
  ps = ComponentArray(w, nlp.ps_axes)
  st = Lux.testmode(nlp.st)
  correct, total = 0, 0
  for batch in data_loader
    x, y = nlp.device(batch)
    ŷ, _ = nlp.model(x, ps, st)
    correct += sum(argmax(ŷ, dims = 1) .== argmax(y, dims = 1))
    total += size(y, 2)
  end
  return correct / total
end

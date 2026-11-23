# Minimal, dependency-free minibatching.
#
# Data are any combination of arrays (observations along the last dimension), tuples and
# named tuples of arrays. Other containers can opt in by extending `numobs` and `getobs`.

"""
    numobs(data)

Number of observations in `data`. Arrays store observations along their last dimension.
Tuples and named tuples must contain the same number of observations in every entry.
Extend this function (together with [`getobs`](@ref)) for custom data containers.
"""
numobs(x::AbstractArray) = size(x, ndims(x))
function numobs(x::Union{Tuple, NamedTuple})
  isempty(x) && throw(ArgumentError("cannot count observations of an empty container"))
  ns = map(numobs, values(x))
  n = first(ns)
  all(==(n), ns) || throw(DimensionMismatch("entries of the data container have different numbers of observations: $ns"))
  return n
end
numobs(x) = throw(ArgumentError(
  "LuxNLPModels does not know how to count observations of $(typeof(x)); " *
  "extend `LuxNLPModels.numobs` and `LuxNLPModels.getobs`, or use full-batch mode."))

"""
    getobs(data, idx)

Materialize the observations `idx` of `data` (a copy, not a view, so that the result has the
same array type as `data` and works with BLAS / GPU kernels).
"""
getobs(x::AbstractArray{T, N}, idx) where {T, N} = x[ntuple(_ -> Colon(), Val(N - 1))..., idx]
getobs(x::Tuple, idx) = map(d -> getobs(d, idx), x)
getobs(x::NamedTuple, idx) = map(d -> getobs(d, idx), x)

"""
    AbstractBatchSampler

Draws minibatches from the data. Interface: `_draw!(s, data)` (next batch), `_epoch(s)`,
`_reset_sampler!(s)`, `_sampler_nobs(s, data)` (observations per epoch) and
`_batch_nobs(s, batch)` (observations in a batch), used for work accounting.
"""
abstract type AbstractBatchSampler end

"""
    MinibatchSampler

Sampling without replacement within an epoch; the permutation is redrawn at every epoch
when `shuffle = true`. With `partial = false` (default) the last incomplete batch of an
epoch is dropped so that every minibatch has exactly `batchsize` observations.
`epoch` is the 1-based index of the epoch to which the current batch belongs.
"""
mutable struct MinibatchSampler{R <: AbstractRNG} <: AbstractBatchSampler
  nobs::Int
  batchsize::Int
  shuffle::Bool
  partial::Bool
  rng::R
  perm::Vector{Int}
  pos::Int
  epoch::Int
end

function MinibatchSampler(nobs::Integer, batchsize::Integer; shuffle::Bool = true,
                          partial::Bool = false, rng::AbstractRNG = Random.default_rng())
  1 ≤ batchsize ≤ nobs || throw(ArgumentError("batchsize must be in 1:$nobs, got $batchsize"))
  perm = collect(1:nobs)
  shuffle && Random.shuffle!(rng, perm)
  return MinibatchSampler(Int(nobs), Int(batchsize), shuffle, partial, rng, perm, 0, 1)
end

nbatches(s::MinibatchSampler) = s.partial ? cld(s.nobs, s.batchsize) : fld(s.nobs, s.batchsize)

function next_indices!(s::MinibatchSampler)
  remaining = s.nobs - s.pos
  if remaining == 0 || (!s.partial && remaining < s.batchsize)
    s.epoch += 1
    s.pos = 0
    s.shuffle && Random.shuffle!(s.rng, s.perm)
  end
  stop = min(s.pos + s.batchsize, s.nobs)
  idx = s.perm[(s.pos + 1):stop]
  s.pos = stop
  return idx
end

function _reset_sampler!(s::MinibatchSampler)
  s.pos = 0
  s.epoch = 1
  s.shuffle && Random.shuffle!(s.rng, s.perm)
  return s
end

_draw!(s::MinibatchSampler, data) = getobs(data, next_indices!(s))
_epoch(s::MinibatchSampler) = s.epoch
_sampler_nobs(s::MinibatchSampler, data) = s.nobs
_batch_nobs(::MinibatchSampler, batch) = max(_count(batch), 1)

"""
    FieldSampler

Per-field minibatching of a `NamedTuple` of data sets with different numbers of observations,
e.g. PINN interior and boundary collocation points. Built from
`batchsize = (interior = 64, boundary = nothing)`: an integer draws minibatches of that size
from the field (each field has its own permutation and epochs); `nothing` passes the whole
field in every batch. The epoch counter and the work accounting follow the *driving* field,
the sampled field with the most minibatches per epoch.
"""
mutable struct FieldSampler{NT <: NamedTuple} <: AbstractBatchSampler
  samplers::NT        # field => MinibatchSampler or nothing
  driving::Symbol
end

function FieldSampler(data::NamedTuple, batchsize::NamedTuple; shuffle::Bool = true,
                      partial::Bool = false, rng::AbstractRNG = Random.default_rng())
  keys(batchsize) == keys(data) || throw(ArgumentError(
    "`batchsize` must have the same fields as the data, in the same order: $(keys(data)); got $(keys(batchsize))"))
  samplers = map(keys(data)) do k
    b = batchsize[k]
    b === nothing ? nothing : MinibatchSampler(numobs(data[k]), b; shuffle, partial, rng)
  end
  samplers = NamedTuple{keys(data)}(samplers)
  sampled = [k for k in keys(data) if samplers[k] !== nothing]
  isempty(sampled) && throw(ArgumentError("at least one field must have an integer batch size"))
  driving = sampled[argmax([nbatches(samplers[k]) for k in sampled])]
  return FieldSampler(samplers, driving)
end

function _draw!(s::FieldSampler, data::NamedTuple)
  vals = map(keys(data)) do k
    sk = s.samplers[k]
    sk === nothing ? data[k] : getobs(data[k], next_indices!(sk))
  end
  return NamedTuple{keys(data)}(vals)
end
_epoch(s::FieldSampler) = s.samplers[s.driving].epoch
function _reset_sampler!(s::FieldSampler)
  foreach(sk -> sk === nothing || _reset_sampler!(sk), values(s.samplers))
  return s
end
_sampler_nobs(s::FieldSampler, data) = numobs(data[s.driving])
_batch_nobs(s::FieldSampler, batch) = numobs(batch[s.driving])

# full-batch mode
_batch_nobs(::Nothing, batch) = max(_count(batch), 1)

_count(x) = try
  numobs(x)
catch
  0
end

# Shape signature of a batch; used to decide whether cached AD preparations are still valid.
_shape(x::AbstractArray) = size(x)
_shape(x::Union{Tuple, NamedTuple}) = map(_shape, values(x))
_shape(x) = nothing

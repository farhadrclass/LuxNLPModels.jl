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
    MinibatchSampler

Sampling without replacement within an epoch; the permutation is redrawn at every epoch
when `shuffle = true`. With `partial = false` (default) the last incomplete batch of an
epoch is dropped so that every minibatch has exactly `batchsize` observations.
`epoch` is the 1-based index of the epoch to which the current batch belongs.
"""
mutable struct MinibatchSampler{R <: AbstractRNG}
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

# Shape signature of a batch; used to decide whether cached AD preparations are still valid.
_shape(x::AbstractArray) = size(x)
_shape(x::Union{Tuple, NamedTuple}) = map(_shape, values(x))
_shape(x) = nothing

# Batch sources. Data are arrays, or Tuples / NamedTuples of arrays, with observations along the
# last dimension. A source owns a preallocated host buffer of one batch; `advance!` gathers the
# next batch into it in place.
#
# Interface for `S <: AbstractBatchSource`:
#   advance!(src)              next batch at the nominal size; epoch counter updated per pass
#   set_indices!(src, idx)     explicit observations (any size); reallocates if the size changes
#   replace_data!(src, data)   new data with the same types and observation shape; restarts passes
#   getbatch(src)              host buffer holding the current batch
#   alldata(src)               the full data, for evaluation
#   epoch(src)                 completed passes
#   nominal_batchsize(src)     training batch size (unchanged by set_indices!)

abstract type AbstractBatchSource end

epoch(src::AbstractBatchSource) = src.epoch

# ---------- helpers on arrays and Tuples / NamedTuples of arrays ----------
_nobs(A::AbstractArray) = size(A, ndims(A))
function _nobs(t::Union{Tuple, NamedTuple})
  n = _nobs(first(t))
  all(a -> _nobs(a) == n, t) ||
    throw(DimensionMismatch("all arrays need the same number of observations"))
  return n
end
_obsshape(A::AbstractArray) = Base.front(size(A))
_obsshape(t::Union{Tuple, NamedTuple}) = map(_obsshape, Tuple(t))
_alloc(A::AbstractArray, k::Int) = similar(A, Base.front(size(A))..., k)
_alloc(t::Union{Tuple, NamedTuple}, k::Int) = map(a -> _alloc(a, k), t)
_gather!(buf::AbstractArray, A::AbstractArray, idx) = copyto!(buf, selectdim(A, ndims(A), idx))
_gather!(buf::Union{Tuple, NamedTuple}, t::Union{Tuple, NamedTuple}, idx) =
  map((b, a) -> _gather!(b, a, idx), buf, t)
_obs(A::AbstractArray, idx) = copy(selectdim(A, ndims(A), idx))
_obs(t::Union{Tuple, NamedTuple}, idx) = map(a -> _obs(a, idx), t)

"""
    ArrayBatchSource(data; batchsize = 128, shuffle = true, seed = 0)

`data`: an array, or a Tuple / NamedTuple of arrays (e.g. `(X, Y)` or an RL rollout
`(obs = O, act = A1h, logp_old = L, adv = Adv, ret = R)`), observations along the last dimension.

The last incomplete batch of each pass is dropped, so every batch has `batchsize` observations:
ratio tests compare functions of the same size and buffers are reused.
"""
mutable struct ArrayBatchSource{D, H} <: AbstractBatchSource
  data::D
  batchsize::Int
  shuffle::Bool
  rng::Xoshiro
  perm::Vector{Int}
  pos::Int
  epoch::Int
  idx::Vector{Int}
  host::H
end

function ArrayBatchSource(data; batchsize::Int = 128, shuffle::Bool = true, seed::Int = 0)
  n = _nobs(data)
  1 <= batchsize <= n || throw(ArgumentError("batchsize must be in 1:$n"))
  src = ArrayBatchSource(data, batchsize, shuffle, Xoshiro(seed), Int[], 0, 0,
                         zeros(Int, batchsize), _alloc(data, batchsize))
  return _restart!(src)
end

nominal_batchsize(src::ArrayBatchSource) = src.batchsize
getbatch(src::ArrayBatchSource) = src.host
alldata(src::ArrayBatchSource) = src.data

function _restart!(src::ArrayBatchSource)
  n = _nobs(src.data)
  src.perm = src.shuffle ? randperm(src.rng, n) : collect(1:n)
  src.pos = 0
  src.epoch = 0
  return advance!(src)
end

function advance!(src::ArrayBatchSource)
  if length(src.idx) != src.batchsize          # back to the nominal size after set_indices!
    src.idx = zeros(Int, src.batchsize)
    src.host = _alloc(src.data, src.batchsize)
  end
  n = length(src.perm)
  if src.pos + src.batchsize > n
    src.shuffle && randperm!(src.rng, src.perm)
    src.pos = 0
    src.epoch += 1
  end
  copyto!(src.idx, 1, src.perm, src.pos + 1, src.batchsize)
  src.pos += src.batchsize
  _gather!(src.host, src.data, src.idx)
  return src
end

function set_indices!(src::ArrayBatchSource, idx)
  if length(idx) != length(src.idx)
    src.idx = collect(Int, idx)
    src.host = _alloc(src.data, length(idx))
  else
    copyto!(src.idx, idx)
  end
  _gather!(src.host, src.data, src.idx)
  return src
end

"""
    replace_data!(src, data)

Replace the data (same container and array types, same shape per observation; any number of
observations ≥ the batch size), then restart the passes at epoch 0. Used for RL rollouts.
"""
function replace_data!(src::ArrayBatchSource{D}, data::D) where {D}
  _obsshape(data) == _obsshape(src.data) ||
    throw(DimensionMismatch("new data must have the same shape per observation"))
  n = _nobs(data)
  src.batchsize <= n || throw(ArgumentError("new data has $n observations < batchsize $(src.batchsize)"))
  src.data = data
  return _restart!(src)
end

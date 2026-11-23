# 1D Poisson pilot: -u''(x) = π² sin(πx) on (0, 1), u(0) = u(1) = 0, exact u(x) = sin(πx).
# Residual formulations for LuxNLSModel; ½‖F‖² = mean(interior residual²) + λ·mean(boundary²).
# Include this file after `using LuxNLPModels, Lux, ADTypes`.

using LinearAlgebra

rhs(x) = oftype(x, π)^2 * sinpi(x)

# Data: NamedTuple of collocation points (fields of different sizes).
function poisson_data(T = Float64; n_interior = 128)
  xi = collect(range(T(0), T(1); length = n_interior + 2))[2:(end - 1)]
  return (interior = reshape(xi, 1, :), boundary = T[0 1])
end

# MLPs with tanh hidden layers and identity output; `nout = 2` for the first-order system.
mlp(width, depth; nout = 1) =
  Chain(Dense(1 => width, tanh), [Dense(width => width, tanh) for _ in 2:depth]..., Dense(width => nout))

# (a) Second-order residual, u, u_x, u_xx propagated through the MLP in closed form
#     (forward Taylor mode written with array operations). No nested AD: J v and Jᵀw are
#     first-order derivatives of explicit array code. Specific to tanh MLPs.
function taylor2(ps, x::AbstractMatrix)
  h, h1, h2 = x, one.(x), zero.(x)
  names = propertynames(ps)
  for (i, name) in enumerate(names)
    p = getproperty(ps, name)
    z = p.weight * h .+ p.bias
    z1 = p.weight * h1
    z2 = p.weight * h2
    if i < length(names)
      a = tanh.(z)
      s = 1 .- a .^ 2                       # tanh' = s, tanh'' = -2 a s
      h, h1, h2 = a, s .* z1, s .* z2 .- 2 .* a .* s .* z1 .^ 2
    else
      h, h1, h2 = z, z1, z2
    end
  end
  return h, h1, h2
end

function residual_taylor(model, ps, st, b; λ = 1)
  _, _, uxx = taylor2(ps, b.interior)
  ub, _, _ = taylor2(ps, b.boundary)
  Ni, Nb = size(b.interior, 2), size(b.boundary, 2)
  return vcat(vec(uxx .+ rhs.(b.interior)) .* sqrt(2 / Ni), vec(ub) .* sqrt(2λ / Nb))
end

# (b) First-order system: the network outputs (u, q), residuals q - u_x and -q_x - f.
#     Input derivatives with Lux's batched_jacobian (nested AD of order 2 overall for J v / Jᵀw).
function residual_firstorder(model, ps, st, b; λ = 1)
  sm = StatefulLuxLayer{true}(model, ps, st)
  x = b.interior
  out = sm(x)                                             # 2 × N
  J = Lux.batched_jacobian(sm, AutoForwardDiff(), x)      # 2 × 1 × N
  u_x = reshape(J[1:1, :, :], 1, :)
  q_x = reshape(J[2:2, :, :], 1, :)
  q = out[2:2, :]
  ub = sm(b.boundary)[1:1, :]
  Ni, Nb = size(x, 2), size(b.boundary, 2)
  return vcat(vec(q .- u_x) .* sqrt(2 / Ni), vec(-q_x .- rhs.(x)) .* sqrt(2 / Ni),
              vec(ub) .* sqrt(2λ / Nb))
end

# (c) Second-order residual by nesting batched_jacobian twice (3rd order overall for J v):
#     the combination the Lux documentation advises against.
function residual_nested(model, ps, st, b; λ = 1)
  sm = StatefulLuxLayer{true}(model, ps, st)
  ux(x) = reshape(Lux.batched_jacobian(sm, AutoForwardDiff(), x), 1, :)
  uxx = reshape(Lux.batched_jacobian(ux, AutoForwardDiff(), b.interior), 1, :)
  ub = sm(b.boundary)
  Ni, Nb = size(b.interior, 2), size(b.boundary, 2)
  return vcat(vec(uxx .+ rhs.(b.interior)) .* sqrt(2 / Ni), vec(ub) .* sqrt(2λ / Nb))
end

# (d) Second-order residual with nested Enzyme input derivatives, as in Lux's PINN example.
#     For backend = ReactantBackend() (requires `using Reactant, Enzyme` before including this
#     file): Reactant supports the 3rd-order derivatives that J v and Jᵀw then need.
function residual_enzyme(model, ps, st, b; λ = 1)
  sm = StatefulLuxLayer{true}(model, ps, st)
  ux(sm, x) = Enzyme.gradient(Enzyme.Reverse, sum ∘ sm, x)[1]     # pointwise model: ∂(Σu)/∂xᵢ = u'(xᵢ)
  uxx = Enzyme.gradient(Enzyme.Reverse, sum ∘ ux, Enzyme.Const(sm), b.interior)[2]
  ub = sm(b.boundary)
  Ni, Nb = size(b.interior, 2), size(b.boundary, 2)
  return vcat(vec(uxx .+ rhs.(b.interior)) .* sqrt(2 / Ni), vec(ub) .* sqrt(2λ / Nb))
end

# Relative L2 error of u against sin(πx) on a uniform grid.
function rel_l2(nls, x; ngrid = 1001)
  T = eltype(x)
  xs = reshape(collect(range(T(0), T(1); length = ngrid)), 1, :)
  u = predict(nls, x, xs)[1:1, :]
  return norm(u .- sinpi.(xs)) / norm(sinpi.(xs))
end

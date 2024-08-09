########################################################################################################
########################################################################################################
# Simulation of different multiscale SDE systems and their respective SDE limit obtained through
# homogenization theory. Considered classes of processes involve fast Ornstein-Uhlenbeck, overdamped
# Langevin with a large-scale potential and fast oscillating part in 1D and 2D, Truncated Burger's
# equation.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 15.04.2024

# required packages
using CairoMakie
using DifferentialEquations
using HCubature
using LaTeXStrings

########################################################################################################
## functions for the realization of multiscale and effective processes, different examples
########################################################################################################

## Fast Ornstein-Uhlenbeck ##

# general mutliscale system for fast Ornstein-Uhlenbeck process
# @doc raw""" """ is a combo to avoid escaping latex commands as \\epsilon
@doc raw"""
    Fast_OU_ϵ(x0, y0; <keyword arguments>)

Return a two-dimensional fast-slow Ornstein-Uhlenbeck process starting at `(x0, y0)` as a discretized time series.

The corresponding stochastic differential equation is for ``t \in [0,T]``
```math
\begin{aligned}
  dX_ϵ(t) &= \left( \frac{1}{ϵ} σ(X_ϵ(t)) Y_ϵ(t) + h(X_ϵ(t), Y_ϵ(t)) - σ'(X_ϵ(t))σ(X_ϵ(t)) \right) dt, \quad &X_ϵ(0) = x_0, \\
  dY_ϵ(t) &= -\frac{1}{ϵ^2} Y_ϵ(t) + \frac{\sqrt{2}}{ϵ} dV(t), \quad &Y_ϵ(0) = y_0.
\end{aligned}
```
Here, ``σ'`` is the first derivative of ``σ``. A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `x0::Real`:         initial point ``x_0`` of slow process ``X_ϵ``.
- `y0::Real`:         initial point ``y_0`` of fast process ``Y_ϵ``.
- `func_config`:      collection of the functions ``h, σ`` and ``σ'`` as a tuple.
- `ϵ::Real=0.1`:      small scale parameter ``ϵ``.
- `T::Real=100`:      time horizon of time series.
- `dt:Real=1e-3`:     time discretization step used in the Euler-Maruyama scheme.

---
# Examples
```julia-repl
julia> h = x -> -x
julia> σ = x -> sqrt(2)
julia> σ_prime = x -> 0
julia> Fast_OU_ϵ(1.0, 1.0, func_config=(h, σ, σ_prime))
```

---
See also [`Fast_OU_∞`](@ref), [`LDA`](@ref), [`NLDAM`](@ref), [`NSDP`](@ref).
"""
function Fast_OU_ϵ(x0, y0; func_config, ϵ=0.1, T=100, dt=1e-3)
    h = func_config[1]
    σ = func_config[2]
    σ_prime = func_config[3]

    N = convert(Int64, T/dt) - 1
  
    X = Array{Float64}(undef, 1, N+1)
    Y = Array{Float64}(undef, 1, N+1)
    X[1] = x0
    Y[1] = y0
    
    for k in 1:N
      dW = sqrt(dt)*randn(1)[1]
      X[k+1] = X[k] + (1/ϵ*σ(X[k])Y[k] + h(X[k]) - σ_prime(X[k])σ(X[k]))dt
      Y[k+1] = Y[k] + (-1/ϵ^2*Y[k])dt + sqrt(2)/ϵ*dW
    end
    
    (X, Y)
end

# general limit process for fast Ornstein-Uhlenbeck process
@doc raw"""
    Fast_OU_∞(X0; <keyword arguments>)

Return a one-dimensional limit process, homogenized from the fast-slow Ornstein-Uhlenbeck process, starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is for ``t \in [0,T]``
```math
\begin{aligned}
  dX(t) = \bar{h}(X(t)) dt - \sqrt{2 σ(X(t))^2} dW(t), \quad X(0) = X_0.
\end{aligned}
```
Here, ``\bar{h}`` is the average of ``h`` with respect to the invariant measure of the fast process ``Y_ϵ`` coming from [`Fast_OU_ϵ`](@ref). 
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Real`:         initial point ``X_0`` of limit process ``X``.
- `func_config`:      collection of the functions ``\bar{h}`` and ``σ`` as a tuple.
- `T::Real=100`:      time horizon of time series.
- `dt:Real=1e-3`:     time discretization step used in the Euler-Maruyama scheme.

---
# Examples
```julia-repl
julia> h_aver = x -> -x   # corresponds to an ordinary Ornstein-Uhlenbeck process
julia> σ = x -> sqrt(2)
julia> Fast_OU_∞(1.0, func_config=(h_aver, σ))
```

---
See also [`Fast_OU_ϵ`](@ref), [`LDA`](@ref), [`NLDAM`](@ref), [`NSDP`](@ref).
"""
function Fast_OU_∞(X0; func_config, T=100, dt=1e-3)
  h_aver = func_config[1]
  σ = func_config[2]

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  X[1] = X0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + h_aver(X[k])dt + sqrt(σ(X[k])^2)dW
  end
  
  X
end

# linear drift with additive noise
@doc raw"""
    LDA(A, σ)

Return a tuple of functions used for the definition of [`Fast_OU_ϵ`](@ref).

The returned functions are
```math
\begin{aligned}
  h(x) &= -Ax, \quad &A > 0, \\
  σ(x) &= \sqrt{σ}, \quad &σ>0, \\
  σ'(x) &= 0.
\end{aligned}
```
They yield a linear drift with additive noise.

---
# Arguments
- `A::Real=1`:        non-negative real number.
- `σ::Real=1`:        positive real number.

---
See also [`Fast_OU_ϵ`](@ref).
"""
function LDA(A=1, σ=1)
  h = x -> -A*x
  σ = x -> sqrt(σ)
  σ_prime = x -> 0

  (h, σ, σ_prime)
end

# nonlinear drift with additive and multiplicative noise
@doc raw"""
    NLDAM(A, B, σ_a, σ_b)

Return a tuple of functions used for the definition of [`Fast_OU_ϵ`](@ref).

The returned functions are
```math
\begin{aligned}
  h(x) &= Ax - Bx^3, \quad &A, B > 0, \\
  σ(x) &= \sqrt{σ_a + σ_b x^2}, \quad &σ_a, σ_b > 0, \\
  σ'(x) &= \frac{σ_b x}{\sqrt{σ_a + σ_b x^2}}.
\end{aligned}
```
They yield a nonlinear drift with additive and multiplicative noise.

---
# Arguments
- `A::Real=1`:          non-negative real number.
- `B::Real=1`:          non-negative real number.
- `σ_a::Real=1`:        positive real number.
- `σ_b::Real=1`:        positive real number.

---
See also [`Fast_OU_ϵ`](@ref).
"""
function NLDAM(A=1, B=1, σ_a=1, σ_b=1)
  h = x -> A*x - B*x^3
  σ = x -> sqrt(σ_a + σ_b*x^2)
  σ_prime = x -> σ_b*x/sqrt(σ_a + σ_b*x^2)

  (h, σ, σ_prime)
end
  
# nonlinear drift with additive and multiplicative noise, non-symmetric double-well potential
@doc raw"""
    NSDP(A, B, C, σ_a, σ_b)

Return a tuple of functions used for the definition of [`Fast_OU_ϵ`](@ref).

The returned functions are
```math
\begin{aligned}
  h(x) &= Ax + Bx^2 - Cx^3, \quad &A, B, C > 0, \\
  σ(x) &= \sqrt{σ_a + σ_b x^2}, \quad &σ_a, σ_b > 0, \\
  σ'(x) &= \frac{σ_b x}{\sqrt{σ_a + σ_b x^2}}.
\end{aligned}
```
They yield a nonlinear, non-symmetric double-well potential drift with additive and multiplicative noise.

---
# Arguments
- `A::Real=1`:          non-negative real number.
- `B::Real=1`:          non-negative real number.
- `C::Real=1`:          non-negative real number.
- `σ_a::Real=1`:        positive real number.
- `σ_b::Real=1`:        positive real number.

---
See also [`Fast_OU_ϵ`](@ref).
"""
function NSDP(A=1, B=1, C=1, σ_a=1, σ_b=1)
  h = x -> A*x + B*x^2 - C*x^3
  sigma = x -> sqrt(σ_a + σ_b*x^2)
  sigma_prime = x -> σ_b*x/sqrt(σ_a + σ_b*x^2)

  (h, sigma, sigma_prime)
end

## Overdamped Langevin process with large-scale potential and fast oscillating part ##

# general multiscale system for overdamped Langevin process with large-scale potential V and fast oscillating part p
@doc raw"""
    Overdamped_LO_ϵ(x0; <keyword arguments>)

Return a two-dimensional overdamped Langevin process with a large-scale potential and a fast oscillating part starting at `(x0, y0/ϵ)` as a discretized time series.

The corresponding stochastic differential equation is for ``t \in [0,T]``
```math
\begin{aligned}
  dX_ϵ(t) = -α V'(X_ϵ(t)) - \frac{1}{ϵ} p'\left( \frac{X_ϵ(t)}{ϵ} \right) dt  + \sqrt{2 σ} dW(t), \quad &X_ϵ(0) = x_0, \\
  dY_ϵ(t) = -\frac{α}{ϵ} V'(X_ϵ(t)) - \frac{1}{ϵ^2} p'\left( Y_ϵ(t) \right) dt  + \sqrt{\frac{2 σ}{ϵ^2}} dW(t), \quad &Y_ϵ(0) = y_0.
\end{aligned}
```
Here, ``V`` is a large-scale potential and ``p`` a ``2π``-periodic function, see [`LDO`](@ref) or [`NLDO`](@ref). Note that ``Y_ϵ = X_ϵ/ϵ``. 
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `x0::Real`:         initial point ``x_0`` of slow process ``X_ϵ``.
- `func_config`:      collection of the functions ``V, V', p`` and ``p'`` as a tuple.
- `α::Real`:          non-negative drift parameter ``α``.
- `σ::Real`:          positive diffusion parameter ``σ``.
- `ϵ::Real=0.1`:      small scale parameter ``ϵ``.
- `T::Real=100`:      time horizon of time series.
- `dt:Real=1e-3`:     time discretization step used in the Euler-Maruyama scheme.

---
See also [`Overdamped_LO_∞`](@ref), [`LDO`](@ref), [`NLDO`](@ref).
"""
function Overdamped_LO_ϵ(x0; func_config, α, σ, ϵ=0.1, T=100, dt=1e-3)
  
  V_prime = func_config[2]
  p_prime = func_config[4]

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  Y = Array{Float64}(undef, 1, N+1)
  X[1] = x0
  Y[1] = x0/ϵ
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (-α*V_prime(X[k]) - 1/ϵ*p_prime(Y[k]))dt + sqrt(2σ)dW
    Y[k+1] = Y[k] + (-α/ϵ*V_prime(X[k]) - 1/ϵ^2*p_prime(Y[k]))dt + sqrt(2σ)/ϵ*dW
  end
  
  (X, Y)
end

# general limit process for overdamped Langevin process with large-scale potential V and fast oscillating part p

# corrective constant in effective limit equation where period L=2pi
@doc raw"""
    K(p, σ)

Return corrective constant of the cell problem of the homogenization in the overdamped Langevin case.

---
# Arguments
- `p`:                ``2π``-periodic function.
- `σ`:                positive diffusion parameter of slow process ``X_ϵ``.

---
See also [`Overdamped_LO_∞`](@ref).
"""
function K(p, sigma)
  Z1 = HCubature.hquadrature(x -> exp(p(x)/sigma), 0, 2pi)[1]
  Z2 = HCubature.hquadrature(x -> exp(-p(x)/sigma), 0, 2pi)[1]
  
  (2pi)^2/(Z1*Z2)
end

@doc raw"""
    Overdamped_LO_∞(X0; <keyword arguments>)

Return a one-dimensional overdamped limit Langevin process, homogenized from the multiscale overdamped Langevin process,  starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is for ``t \in [0,T]``
```math
\begin{aligned}
  dX(t) = -α K V'(X(t)) dt  + \sqrt{2 σ K} dW(t), \quad &X(0) = X_0.
\end{aligned}
```
Here, ``K`` is a corrective constant that comes from the cell problem of the homogenization, see also [`K`](@ref), and is computed inside the function.
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Real`:         initial point ``X_0`` of slow process ``X_ϵ``.
- `func_config`:      collection of the functions ``V, V', p`` and ``p'`` as a tuple.
- `α::Real`:          non-negative drift parameter ``α``.
- `σ::Real`:          positive diffusion parameter ``σ``.
- `ϵ::Real=0.1`:      small scale parameter ``ϵ``.
- `T::Real=100`:      time horizon of time series.
- `dt:Real=1e-3`:     time discretization step used in the Euler-Maruyama scheme.

---
See also [`Overdamped_LO_ϵ`](@ref), [`LDO`](@ref), [`NLDO`](@ref).
"""
function Overdamped_LO_∞(X0; func_config, α, σ, T=100, dt=0.001)
  
  V_prime = func_config[2]
  p = func_config[3]

  A = α*K(p, σ)
  Σ = σ*K(p, σ)

  N = convert(Int64, T/dt) - 1

  X = zeros(1, N+1)
  X[1] = X0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] - A*V_prime(X[k])dt + sqrt(2Σ)dW
  end
  
  X
end

# quadratic potential V with sine oscillation p
@doc raw"""
    LDO()

Return a tuple of functions used for the definition of [`Overdamped_LO_ϵ`](@ref) and [`Overdamped_LO_∞`](@ref).

The returned functions are
```math
\begin{aligned}
  V(x) &= \frac12 x^2, \\
  V'(x) &= x, \\
  p(x) &= sin(x), \\
  p'(x) &= cos(x).
\end{aligned}
```
They yield a quadratic potential drift with a sine oscillation.

---
See also [`Overdamped_LO_ϵ`](@ref), [`Overdamped_LO_∞`](@ref).
"""
function LDO()
  V = x -> x^2/2
  V_prime = x -> x
  p = x -> sin(x)
  p_prime = x -> cos(x)

  (V, V_prime, p, p_prime)
end

# bistable potential V with sine oscillation p
@doc raw"""
    NLDO()

Return a tuple of functions used for the definition of [`Overdamped_LO_ϵ`](@ref) and [`Overdamped_LO_∞`](@ref).

The returned functions are
```math
\begin{aligned}
  V(x) &= \frac14 x^4 - \frac12 x^2, \\
  V'(x) &= x^3 - x, \\
  p(x) &= sin(x), \\
  p'(x) &= cos(x).
\end{aligned}
```
They yield a bistable potential drift with a sine oscillation.

---
See also [`Overdamped_LO_ϵ`](@ref), [`Overdamped_LO_∞`](@ref).
"""
function NLDO()
  V = x -> -x^2/2 + x^4/4
  V_prime = x -> -x + x^3
  p = x -> sin(x)
  p_prime = x -> cos(x)

  (V, V_prime, p, p_prime)
end

## Overdamped Langevin process with quadratic potential and fast separable oscillating part in 2D ##

# general multiscale system for overdamped Langevin process with quadratic potential and fast separable oscillating part
# x0 2-dimensional initial vector
# Alpha symmetric positive definite matrix
# p_prime holding two oscillations p1_prime and p2_prime
function Overdamped_LO_ϵ_2D(;x0, y0, Alpha, p_prime, sigma, ϵ=0.1, T=100, dt=0.001)
  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 2, N+1)
  Y = Array{Float64}(undef, 2, N+1)
  X[:,1] = x0
  Y[:,1] = y0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[1,k+1] = X[1,k] - (Alpha[1,1]X[1,k] + Alpha[1,2]X[2,k] + 1/ϵ*p_prime[1](Y[1,k]))dt + sqrt(2sigma)dW1
    X[2,k+1] = X[2,k] - (Alpha[2,1]X[1,k] + Alpha[2,2]X[2,k] + 1/ϵ*p_prime[2](Y[2,k]))dt + sqrt(2sigma)dW2
    Y[1,k+1] = Y[1,k] - (Alpha[1,1]/ϵ*X[1,k] + Alpha[1,2]/ϵ*X[2,k] + 1/ϵ^2*p_prime[1](Y[1,k]))dt + sqrt(2sigma)/ϵ*dW1
    Y[2,k+1] = Y[2,k] - (Alpha[2,1]/ϵ*X[1,k] + Alpha[2,2]/ϵ*X[2,k] + 1/ϵ^2*p_prime[2](Y[2,k]))dt + sqrt(2sigma)/ϵ*dW2
  end
  
  (X, Y)
end

# general limit process for overdamped Langevin process with quadratic potential and fast separable oscillating part
# X0 2-dimensional initial vector
# Alpha symmetric positive definite matrix
# p holding two oscillations p1 and p2
function Overdamped_LO_limit_2D(;X0, Alpha, p, sigma, T=100, dt=0.001)

  CorrK = [K(p[1], sigma) 0 ; 0 K(p[2], sigma)]
  A = CorrK*Alpha
  Sigma = sigma*CorrK

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 2, N+1)
  X[:,1] = X0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[1,k+1] = X[1,k] - (A[1,1]X[1,k] + A[1,2]X[2,k])dt + sqrt(2Sigma)[1,1]dW1 
    X[2,k+1] = X[2,k] - (A[2,1]X[1,k] + A[2,2]X[2,k])dt + sqrt(2Sigma)[2,2]dW2 
  end
  
  X
end

## Truncated Burger's equation ##

# general multiscale system for truncated Burger's equation
function Truncated_B_ϵ(;x0, y0, z0, nu, q1, q2, ϵ=0.1, T=100, dt=0.001)
  
  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  Y = Array{Float64}(undef, 1, N+1)
  Z = Array{Float64}(undef, 1, N+1)
  X[1] = x0
  Y[1] = y0
  Z[1] = z0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[k+1] = X[k] + (nu*X[k] - 1/2ϵ*(X[k]Y[k] + Y[k]Z[k]))dt
    Y[k+1] = Y[k] + (nu*Y[k] - 3/ϵ^2*Y[k] - 1/2ϵ*(2X[k]Z[k] - X[k]^2))dt + q1/ϵ*dW1
    Z[k+1] = Z[k] + (nu*Z[k] - 8/ϵ^2*Z[k]  + 3/2ϵ*X[k]Y[k])dt + q2/ϵ*dW2
  end
  
  (X, Y, Z)
end

# general limit process for truncated Burger's equation
function Truncated_B_limit(;X0, nu, q1, q2, T=100, dt=0.001)

  A = nu + q1^2/396 + q2^2/352
  B = 1/12
  sigma_a = q1^2*q2^2/2112
  sigma_b = q1^2/36

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  X[1] = X0

  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (A*X[k] - B*X[k]^3)dt + sqrt(sigma_a + sigma_b*X[k]^2)dW
  end

  X
end

## Fast chaotic noise ##

# defining ode; specifically written for DifferentialEquations.jl
function Fast_chaotic_ode!(du, u, p, t)
  du[1] = p[1]*u[1] - p[2]*u[1]^3 + p[3]/p[4]*u[3]
  du[2] = 10/p[4]^2*(u[3] - u[2])
  du[3] = 1/p[4]^2*(28u[2] - u[3] - u[2]*u[4])
  du[4] = 1/p[4]^2*(u[2]*u[3] - 8/3*u[4])
end

# general multiscale system for fast chaotic noise solved with DifferentialEquations.jl using a fourth order Runge-Kutta scheme
function Fast_chaotic_ϵ(;xy0, A, B, λ, ϵ=0.1, T=100, dt=0.001)
  p = [A, B, λ, ϵ]
  tspan = (0.0, T)

  prob = ODEProblem(Fast_chaotic_ode!, xy0, tspan, p)
  sol = solve(prob, Tsit5(), saveat = dt, maxiters = 1e8)

  X = sol[1,:]
  pop!(X)
  X
end

# general limit process for fast chaotic noise
function Fast_chaotic_limit(;X0, A, B, σ, T=100, dt=0.001)

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  X[1] = X0

  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (A*X[k] - B*X[k]^3)dt + sqrt(σ)dW
  end

  X
end

#=
## various plots and figures ##

function produce_fig_1D(trajectory, T)

  N = length(trajectory[1])
  T_range = range(0, T, N)

  # create and adjust figure components
  process_fig = Figure(size=(3840,2160), fontsize = 50)
  process_ax = Axis(process_fig[1, 1],
      # x-axis
      xlabel = L"T",
      xticks = LinearTicks(5),
      # y-axis
      yticks = LinearTicks(10),
  )
  Makie.xlims!(process_ax, 0.0, T)
  colsize!(process_fig.layout, 1, Aspect(1, 1.8))
  

  X_ϵ_line = lines!(process_ax, T_range, trajectory[1][:], linewidth = 3.0, color = (:purple, 1.0))
  Y_ϵ_line = lines!(process_ax, T_range, trajectory[2][:], linewidth = 3.0, color = (:grey, 0.4))
  
  axislegend(process_ax,
      [X_ϵ_line, Y_ϵ_line],
      [L"Slow process $X_ϵ$", L"Fast process $Y_ϵ$"],
      labelsize = 80
  )

  process_fig
end

function produce_fig_2D(trajectory)

  # create and adjust figure components
  process_fig = Figure(size=(3840,2160), fontsize = 50)
  process_ax = Axis(process_fig[1, 1],
      # x-axis
      xticks = LinearTicks(10),
      # y-axis
      yticks = LinearTicks(10),
  )
  colsize!(process_fig.layout, 1, Aspect(1, 1.8))
  

  X_ϵ_line = lines!(process_ax, trajectory[1][1,:], trajectory[1][2,:], linewidth = 3.0, color = (:purple, 1.0))
  Y_ϵ_line = lines!(process_ax, trajectory[2][1,:], trajectory[2][2,:], linewidth = 3.0, color = (:grey, 0.4))
  
  axislegend(process_ax,
      [X_ϵ_line, Y_ϵ_line],
      [L"Slow process $X_ϵ$", L"Fast process $Y_ϵ$"],
      labelsize = 80
  )

  process_fig
end

## Fast Ornstein-Uhlenbeck ##

# linear drift with additive noise
T = 25.0
A = 0.5
sig = 1.0
diff_func = LDA(A, sig)
trajectory = Fast_OU_ϵ(x0=1.0, y0=1.0, func_config=diff_func, ϵ=0.1, T=T)

fig = produce_fig_1D(trajectory, T)
#save("trajectory_fast_OU_process.pdf", fig)

# nonlinear drift with additive and multiplicative noise
T = 25.0
A = 2.0
B = 10.0
σ_a = 1.0
σ_b = 1.0
diff_func = NLDAM(A, B, σ_a, σ_b)
trajectory = Fast_OU_ϵ(x0=1.0, y0=1.0, func_config=diff_func, ϵ=0.1, T=T)

fig = produce_fig_1D(trajectory, T)

# nonlinear drift with additive and multiplicative noise, non-symmetric double-well potential
T = 25.0
A = 1.0
B = 2.0
C = 5.0
σ_a = 1.0
σ_b = 1.0
diff_func = NSDP(A, B, C, σ_a, σ_b)
trajectory = Fast_OU_ϵ(x0=1.0, y0=1.0, func_config=diff_func, ϵ=0.1, T=T)

fig = produce_fig_1D(trajectory, T)

## Overdamped Langevin process with large-scale potential and fast oscillating part in 1D ##

# quadratic potential V with sine oscillation p
T = 25.0
alpha = 0.5
sigma = 2.0
diff_func = LDO()
trajectory = Overdamped_LO_ϵ_1D(x0=1.0, func_config=diff_func, alpha=alpha, sigma=sigma, ϵ=0.1, T=T)

fig = produce_fig_1D(trajectory, T)
#save("trajectory_Langevin_process.pdf", fig)


## Overdamped Langevin process with quadratic potential and fast separable oscillating part in 2D ##
T = 10.0
M = [4 2;2 3]
p1_prime, p2_prime = (x-> cos(x), x -> 1/2*cos(x))
σ = 5  
ϵ = 0.05
dt = 1e-3
trajectory = Overdamped_LO_ϵ_2D(x0=[-5.0, -5.0], y0=[-5.0/ϵ, -5.0/ϵ], Alpha=M, p_prime = (p1_prime, p2_prime), sigma=σ, ϵ=ϵ, T=T, dt=dt)

fig = produce_fig_2D(trajectory)
#save("trajectory_Langevin_process_2D.pdf", fig)

V(x) = -x^2 + x^4/12
p(x) = sin(x/ϵ)
x_range = range(-4,4,2000)

# create and adjust figure components
drift_fig = Figure(size=(3840,2160), fontsize = 50)
drift_ax = Axis(drift_fig[1, 1],
  # x-axis
  xlabel = L"x",
  xticks = LinearTicks(5),
  # y-axis
  yticks = LinearTicks(5),
)
Makie.xlims!(drift_ax, x_range[1], x_range[end])
colsize!(drift_fig.layout, 1, Aspect(1, 1.8))
  

V_line = lines!(drift_ax, x_range, map(V, x_range), linewidth = 10.0, color = (:darkgrey, 1.0), linestyle = :dash)
Vp_line = lines!(drift_ax, x_range, map(x->V(x)+p(x), x_range), linewidth = 3.0, color = (:black, 1.0))

axislegend(drift_ax,
[V_line, Vp_line],
[L"$x^4/12-x^2$", L"$x^4/12 - x^2 + \sin(x/0.05)$"],
labelsize = 80
)

drift_fig
save("Langevin_potentials.pdf", drift_fig)
=#
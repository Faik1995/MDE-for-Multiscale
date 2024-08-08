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
using HCubature, DifferentialEquations, LaTeXStrings, CairoMakie

########################################################################################################
## functions for the realization of multiscale and effective processes, different examples
########################################################################################################

## Fast Ornstein-Uhlenbeck ##

# general multiscale system for fast Ornstein-Uhlenbeck process
"""
    Fast_OU_eps(x, y, z...)

Multiplication operator. `x * y * z *...` calls this function with multiple
arguments, i.e. `*(x, y, z...)`.
"""
function Fast_OU_eps(;x0, y0, func_config, eps=0.1, T=100, dt=0.001)
  
    h = func_config[1]
    sigma = func_config[2]
    sigma_prime = func_config[3]

    N = convert(Int64, T/dt) - 1
  
    X = Array{Float64}(undef, 1, N+1)
    Y = Array{Float64}(undef, 1, N+1)
    X[1] = x0
    Y[1] = y0
    
    for k in 1:N
      dW = sqrt(dt)*randn(1)[1]
      X[k+1] = X[k] + (1/eps*sigma(X[k])Y[k] + h(X[k]) - sigma_prime(X[k])sigma(X[k]))dt
      Y[k+1] = Y[k] + (-1/eps^2*Y[k])dt + sqrt(2)/eps*dW
    end
    
    (X, Y)
end

# general limit process for fast Ornstein-Uhlenbeck process
function Fast_OU_limit(;X0, h, sigma, T=100, dt=0.001)
  
  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  X[1] = X0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + h(X[k])dt + sqrt(2sigma(X[k])^2)dW
  end
  
  X
end

# linear drift with additive noise
function LDA(A, sig)
  h = x -> -A*x
  sigma = x -> sqrt(sig)
  sigma_prime = x -> 0

  (h, sigma, sigma_prime)
end

# nonlinear drift with additive and multiplicative noise
function NLDAM(A, B, sig_a, sig_b)
  h = x -> A*x - B*x^3
  sigma = x -> sqrt(sig_a + sig_b*x^2)
  sigma_prime = x -> sig_b*x/sqrt(sig_a + sig_b*x^2)

  (h, sigma, sigma_prime)
end
  
# nonlinear drift with additive and multiplicative noise, non-symmetric double-well potential
function NSDP(A, B, C, sig_a, sig_b)
  h = x -> A*x + B*x^2 - C*x^3
  sigma = x -> sqrt(sig_a + sig_b*x^2)
  sigma_prime = x -> sig_b*x/sqrt(sig_a + sig_b*x^2)

  (h, sigma, sigma_prime)
end

## Overdamped Langevin process with large-scale potential and fast oscillating part in 1D ##

# general multiscale system for overdamped Langevin process with large-scale potential V and fast oscillating part p
function Overdamped_LO_eps_1D(;x0, func_config, alpha, sigma, eps=0.1, T=100, dt=0.001)
  
  V_prime = func_config[2]
  p_prime = func_config[4]

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 1, N+1)
  Y = Array{Float64}(undef, 1, N+1)
  X[1] = x0
  Y[1] = x0/eps
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (-alpha*V_prime(X[k]) - 1/eps*p_prime(Y[k]))dt + sqrt(2sigma)dW
    Y[k+1] = Y[k] + (-alpha/eps*V_prime(X[k]) - 1/eps^2*p_prime(Y[k]))dt + sqrt(2sigma)/eps*dW
  end
  
  (X, Y)
end

# general limit process for overdamped Langevin process with large-scale potential V and fast oscillating part p

# corrective constant in effective limit equation where period L=2pi
function K(p, sigma)
  Z1 = HCubature.hquadrature(x -> exp(p(x)/sigma), 0, 2pi)[1]
  Z2 = HCubature.hquadrature(x -> exp(-p(x)/sigma), 0, 2pi)[1]
  
  (2pi)^2/(Z1*Z2)
end

function Overdamped_LO_limit_1D(;X0, V_prime, p, alpha, sigma, T=100, dt=0.001)

  A = alpha*K(p, sigma)
  Sigma = sigma*K(p, sigma)

  N = convert(Int64, T/dt) - 1

  X = zeros(1, N+1)
  X[1] = X0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] - A*V_prime(X[k])dt + sqrt(2Sigma)dW
  end
  
  X
end

# quadratic potential V with sine oscillation p
function LDO()
  V = x -> x^2/2
  V_prime = x -> x
  p = x -> sin(x)
  p_prime = x -> cos(x)

  (V, V_prime, p, p_prime)
end

# bistable potential V with sine oscillation p
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
function Overdamped_LO_eps_2D(;x0, y0, Alpha, p_prime, sigma, eps=0.1, T=100, dt=0.001)
  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 2, N+1)
  Y = Array{Float64}(undef, 2, N+1)
  X[:,1] = x0
  Y[:,1] = y0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[1,k+1] = X[1,k] - (Alpha[1,1]X[1,k] + Alpha[1,2]X[2,k] + 1/eps*p_prime[1](Y[1,k]))dt + sqrt(2sigma)dW1
    X[2,k+1] = X[2,k] - (Alpha[2,1]X[1,k] + Alpha[2,2]X[2,k] + 1/eps*p_prime[2](Y[2,k]))dt + sqrt(2sigma)dW2
    Y[1,k+1] = Y[1,k] - (Alpha[1,1]/eps*X[1,k] + Alpha[1,2]/eps*X[2,k] + 1/eps^2*p_prime[1](Y[1,k]))dt + sqrt(2sigma)/eps*dW1
    Y[2,k+1] = Y[2,k] - (Alpha[2,1]/eps*X[1,k] + Alpha[2,2]/eps*X[2,k] + 1/eps^2*p_prime[2](Y[2,k]))dt + sqrt(2sigma)/eps*dW2
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
function Truncated_B_eps(;x0, y0, z0, nu, q1, q2, eps=0.1, T=100, dt=0.001)
  
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

    X[k+1] = X[k] + (nu*X[k] - 1/2eps*(X[k]Y[k] + Y[k]Z[k]))dt
    Y[k+1] = Y[k] + (nu*Y[k] - 3/eps^2*Y[k] - 1/2eps*(2X[k]Z[k] - X[k]^2))dt + q1/eps*dW1
    Z[k+1] = Z[k] + (nu*Z[k] - 8/eps^2*Z[k]  + 3/2eps*X[k]Y[k])dt + q2/eps*dW2
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
function Fast_chaotic_eps(;xy0, A, B, λ, ϵ=0.1, T=100, dt=0.001)
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
  

  X_eps_line = lines!(process_ax, T_range, trajectory[1][:], linewidth = 3.0, color = (:purple, 1.0))
  Y_eps_line = lines!(process_ax, T_range, trajectory[2][:], linewidth = 3.0, color = (:grey, 0.4))
  
  axislegend(process_ax,
      [X_eps_line, Y_eps_line],
      [L"Slow process $X_\epsilon$", L"Fast process $Y_\epsilon$"],
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
  

  X_eps_line = lines!(process_ax, trajectory[1][1,:], trajectory[1][2,:], linewidth = 3.0, color = (:purple, 1.0))
  Y_eps_line = lines!(process_ax, trajectory[2][1,:], trajectory[2][2,:], linewidth = 3.0, color = (:grey, 0.4))
  
  axislegend(process_ax,
      [X_eps_line, Y_eps_line],
      [L"Slow process $X_\epsilon$", L"Fast process $Y_\epsilon$"],
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
trajectory = Fast_OU_eps(x0=1.0, y0=1.0, func_config=diff_func, eps=0.1, T=T)

fig = produce_fig_1D(trajectory, T)
#save("trajectory_fast_OU_process.pdf", fig)

# nonlinear drift with additive and multiplicative noise
T = 25.0
A = 2.0
B = 10.0
sig_a = 1.0
sig_b = 1.0
diff_func = NLDAM(A, B, sig_a, sig_b)
trajectory = Fast_OU_eps(x0=1.0, y0=1.0, func_config=diff_func, eps=0.1, T=T)

fig = produce_fig_1D(trajectory, T)

# nonlinear drift with additive and multiplicative noise, non-symmetric double-well potential
T = 25.0
A = 1.0
B = 2.0
C = 5.0
sig_a = 1.0
sig_b = 1.0
diff_func = NSDP(A, B, C, sig_a, sig_b)
trajectory = Fast_OU_eps(x0=1.0, y0=1.0, func_config=diff_func, eps=0.1, T=T)

fig = produce_fig_1D(trajectory, T)

## Overdamped Langevin process with large-scale potential and fast oscillating part in 1D ##

# quadratic potential V with sine oscillation p
T = 25.0
alpha = 0.5
sigma = 2.0
diff_func = LDO()
trajectory = Overdamped_LO_eps_1D(x0=1.0, func_config=diff_func, alpha=alpha, sigma=sigma, eps=0.1, T=T)

fig = produce_fig_1D(trajectory, T)
#save("trajectory_Langevin_process.pdf", fig)


## Overdamped Langevin process with quadratic potential and fast separable oscillating part in 2D ##
T = 10.0
M = [4 2;2 3]
p1_prime, p2_prime = (x-> cos(x), x -> 1/2*cos(x))
σ = 5  
ϵ = 0.05
dt = 1e-3
trajectory = Overdamped_LO_eps_2D(x0=[-5.0, -5.0], y0=[-5.0/ϵ, -5.0/ϵ], Alpha=M, p_prime = (p1_prime, p2_prime), sigma=σ, eps=ϵ, T=T, dt=dt)

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
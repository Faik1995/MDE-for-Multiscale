########################################################################################################
########################################################################################################
# Minimization of weighted L² distance as per main manuscript (cost functional J) with respect to 
# drift parameter. Data comes as a trajectory of a multiscale process. Optimization is performed 
# with the Julia package Optim.jl.
########################################################################################################
# SDE: Overdamped Langevin Diffusion with a fast oscillating part in 1D
# Effective drift function: derived from a quadratic potential
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 28.04.2024

# required packages
using Optim, JLD2
using LaTeXStrings  # for latex symbols in titles, legends, etc.; a latexstring is denoted by L""

# required other files in the folder
include("MDE_gradients.jl")
include("MDE_Gaussian.jl")

# parameters
α = 2.0;                println("⎔ drift coefficient of multiscale process: α = $(α)")
σ = 1.0;                println("⎔ diffusion coefficient of multiscale process: σ = $(σ)")      
drift = LDO()           # linear drift function 
A = α*K(drift[3], σ);   println("⎔ drift coefficient of effective process: A = $(A)")          
Σ = σ*K(drift[3], σ);   println("⎔ diffusion coefficient of effective process: Σ = $(Σ)")
T = 1000;               println("⎔ time horizon: T = $(T)")
ϵ = 0.1;                println("⎔ small scale parameter: ϵ = $(ϵ)")
dt = 1e-3;              println("⎔ time discretization: dt = $(dt)")

# include here since there is a constant factor in that script which is computed on basis of A and Σ
include("asymptotic_variance.jl")

# data
data_eps = Overdamped_LO_eps_1D(x0=10.0, func_config=drift, alpha=α, sigma=σ, eps=ϵ, T=T, dt=dt)[1]
#data_limit = Overdamped_LO_limit_1D(X0=10.0, V_prime=drift[2], p=drift[3], alpha=α, sigma=σ, T=T, dt=dt)

# cost functional and gradient
J(θ) = Δ_Gaussian1D(data_eps, θ, Σ)
∇J(θ) = Δ_Gaussian1D_grad(data_eps, θ, Σ)

# gradient of cost functional, specifically written for Optim.optimize
function gradient!(storage, θ)
    storage[1] = ∇J(θ[1])
end

# specifying boundary constraints, otherwise we might get an error for negative values
initial_θ = 10.0
lower = 0.0
upper = Inf
inner_optimizer = LBFGS()

# small Monte Carlo study for robustness
T_range = 1000 #range(50, 2000, 40)
T_length = length(T_range)
optimizer_aver_values = Array{Float64}(undef, T_length)
optimizer_stdev_values = Array{Float64}(undef, T_length)
optimizer_values = Array{Float64}(undef, reps)
reps = 1000

for i in 1:T_length
    optimizer_values = Array{Float64}(undef, reps)
    for j in 1:reps
        data_eps = Overdamped_LO_eps_1D(x0=10.0, func_config=drift, alpha=α, sigma=σ, eps=ϵ, T=T_range[i], dt=dt)[1]
        optim_res = optimize(θ -> J(first(θ)), gradient!, [lower], [upper], [initial_θ], Fminbox(inner_optimizer), Optim.Options(show_trace = true))   
        optimizer_values[j] = Optim.minimizer(optim_res)[1]
    end
    
    optimizer_aver_values[i] = mean(optimizer_values)
    optimizer_stdev_values[i] = std(optimizer_values)
end

@show optimizer_aver_values
@show optimizer_stdev_values

# saving output data
#jldsave("optimizer_values_eps01.jld2"; optimizer_aver_values_eps01 = optimizer_aver_values, optimizer_stdev_values_eps01 = optimizer_stdev_values)
#jldsave("MDE_Gaussian_optimizer_values_T1000.jld2"; optimizer_values = optimizer_values)

# visualization of robustness

# loading required output data
#optimizer_aver_values = load("MDE_Gaussian_optimizer_values_eps01.jld2")["optimizer_aver_values_eps01"];     optimizer_stdev_values = load("MDE_Gaussian_optimizer_values_eps01.jld2")["optimizer_stdev_values_eps01"]
#optimizer_aver_values = load("MDE_Gaussian_optimizer_values_eps025.jld2")["optimizer_aver_values_eps025"];     optimizer_stdev_values = load("MDE_Gaussian_optimizer_values_eps025.jld2")["optimizer_stdev_values_eps025"]

val_l = minimum(optimizer_aver_values-1.2optimizer_stdev_values)
val_u = maximum(optimizer_aver_values+1.2optimizer_stdev_values)

# create and adjust figure components
robustness_fig = Figure(size=(2560,1440), fontsize = 40)
robustness_ax = Axis(robustness_fig[1, 1],
    # title
    title = L"MDE estimates $\hat{\vartheta}_T \; (X_\epsilon)$ for $\vartheta_0$ when $\epsilon = 0.25$",
    titlegap = 25,
    titlesize = 50,
    # x-axis
    xlabel = L"T",
    xticks = LinearTicks(5),
    # y-axis
    yticks = LinearTicks(10),
)
Makie.xlims!(robustness_ax, T_range[begin], T_range[end]), Makie.ylims!(robustness_ax, val_l, val_u)
colsize!(robustness_fig.layout, 1, Aspect(1, 1.8))

STD_band = band!(robustness_ax, T_range, optimizer_aver_values-optimizer_stdev_values, optimizer_aver_values+optimizer_stdev_values,
                color = (:lightblue, 0.5)
)
MDE_line = lines!(robustness_ax, T_range, optimizer_aver_values, linewidth = 3.0)
A_line = hlines!(robustness_ax, A, color = (:red, 0.8), linewidth = 5.0, linestyle = :dash)

axislegend(robustness_ax,
    [A_line, MDE_line, STD_band],
    [L"True drift parameter $\vartheta_0$", L"MDE estimates $\hat{\vartheta}_T \; (X_\epsilon)$", L"$1$ standard deviation band"]
)
robustness_fig

#save("MDE_Gaussian_1D_robustness_eps025.pdf",robustness_fig)

# visualization of asymptotic normality

# loading required output data
#optimizer_values = load("MDE_Gaussian_optimizer_values_T1000.jld2")["optimizer_values"]

standardized_optim_values = sqrt(T_range)*(optimizer_values.-A)

asy_var = Σ_∞_l(A, Σ)
MDE_∞(x) = pdf(Normal(0, sqrt(asy_var)), x)
x_range = range(-6,6,1000)
MDE_∞_val = map(MDE_∞, x_range)

# create and adjust figure components
asy_fig = Figure(size=(2560,1440), fontsize = 40)
asy_ax = Axis(asy_fig[1, 1],
    # title
    title = L"MDE estimates $\sqrt{T}\left(\hat{\vartheta}_{T} \; (X_\epsilon) - \vartheta_0\right)$ for $T=1000$",
    titlegap = 25,
    titlesize = 50,
    # x-axis
    xlabel = L"\vartheta",
    xticks = LinearTicks(5),
    # y-axis
    yticks = LinearTicks(10),
)
colsize!(asy_fig.layout, 1, Aspect(1, 1.8))

#MDE_density = density!(asy_ax, standardized_optim_values, color = (:blue, 0.3), strokecolor = :blue, strokewidth = 2.0, strokearound = true)

# bin number for histogram set according to Freedman-Diaconis rule where IQR is the interquartile range    
n = length(standardized_optim_values)
q75 = quantile(standardized_optim_values, 0.75)
q25 = quantile(standardized_optim_values, 0.25)
IQR = q75 - q25
FD = 2IQR/n^(1/3)
bin_num = round((maximum(standardized_optim_values)-minimum(standardized_optim_values))/FD)

MDE_hist = hist!(asy_ax, standardized_optim_values, 
                bins =  convert(Int64, bin_num), color = (:lightblue, 0.7),
                strokewidth = 1.0, strokecolor = :black, 
                normalization = :pdf
)
asy_line = lines!(asy_ax, x_range, MDE_∞_val, 
                linewidth = 3.0, linestyle = :dash, 
                color = (:red, 0.8)
)

axislegend(asy_ax,
    [asy_line],
    [L"Asymptotic Distribution $J(\vartheta_0)^{-1} \mathcal{N}(0,\; \tau^2(\vartheta_0))$"]
)
asy_fig

save("MDE_Gaussian_1D_normality_T1000.pdf", asy_fig)


#=
@time J(A)
@time ∇J(A)

# lineplot of cost functional for reality check
θ_space = range(0.5, 1.5, 10)
f, ax, l = lines(θ_space, map(J, θ_space))
vlines!(ax, A, color = :red)
f
lines(θ_space, map(∇J, θ_space))
=#

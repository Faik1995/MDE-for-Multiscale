########################################################################################################
########################################################################################################
# A new approach for the drift estimation in multiscale settings based on a minimum distance estimation 
# method utilizing characteristic functions of the invariant distribution of the effective limit model.
# The weight function ϕ is chosen as a centered normal distribution with variance β², so that
# a lot of formulas simplify the implementation. This script evaluates the MDE in the case of a 
# Gaussian invariant  measure in one and two dimensions.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 03.05.2024

# required packages
using LinearAlgebra, Dates, Distributions, NaNMath

########################################################################################################
## estimator relevant functions
########################################################################################################

## Gaussian case in 1D via exact distance formula ##

# gaussian convolution kernel for estimation, see calculations from main manuscript
function k(x; β=1)
    exp(-β^2*x^2/2), β
end

# weighted L² distance minus weighted L² norm of C_T in 1D
function Δ_Gaussian1D(data, A, Σ)
    #time_stamp = Dates.format(now(), "HH:MM:SS")
    #@info "⊙ $(time_stamp) - Function call with parameter value $(A)."
    β = k(0)[2]
    δ1 = 1/sqrt(1 + β^2*Σ/A)
    δ2 = 1/sqrt(1 + 2β^2*Σ/A)
    
    N = length(data)
    single_integral = 0.0

    for i in 1:N
      single_integral = single_integral + k(data[i]δ1)[1]
    end
    -2δ1/N*single_integral+δ2
end

## Gaussian case in 2D via exact distance formula ##

# transforming 2D data into 1D data for the exponential in the distance Δ and 
# calculating a determinant relevant for the distance formula; here A is passed 
# as a 4-dimensional vector containing the coefficients of the effective drift matrix
# and Σ is the effective 2x2-dimensional diffusion matrix, i.e. σK
function transf_data_2D(data, A_vec, Σ)
  d = length(data[:,1])   # d=2 in our considered case
  N = length(data[1,:])
  I_d = LinearAlgebra.I[1:d,1:d]  # identity matrix
  β = k(0)[2]
  A = [A_vec[1] A_vec[2]; A_vec[3] A_vec[4]]
  A_inv = LinearAlgebra.inv(A)
  inverse_mat = LinearAlgebra.inv(I_d + β^2*A_inv*Σ)
  det_A_Σ = LinearAlgebra.det(I_d + β^2*A_inv*Σ)

  #time_stamp = Dates.format(now(), "HH:MM:SS")
  #@info "⊙ $(time_stamp) - Covariance matrix equals $(A_inv*Σ)."

  transformed_data = [NaNMath.sqrt(data[:,i]' * inverse_mat * data[:,i]) for i ∈ 1:N]
  transformed_data, det_A_Σ, A_inv
end

# weighted L² distance minus weighted L² norm of C_T in 2D
function Δ_Gaussian2D(data, A_vec, Σ)
  #time_stamp = Dates.format(now(), "HH:MM:SS")
  #@info "⊙ $(time_stamp) - Function call with parameter value $(A_vec)."
  d = length(data[:,1])
  N = length(data[1,:])
  β = k(0)[2]
  I_d = LinearAlgebra.I[1:d,1:d]
  transformed_data, det_A_Σ, A_inv = transf_data_2D(data, A_vec, Σ)

  δ1 = 1/NaNMath.sqrt(det_A_Σ)
  δ2 = 1/NaNMath.sqrt(det(I_d + 2β^2*A_inv*Σ))
    
  single_integral = 0.0
  
  for i in 1:N
    single_integral += k(transformed_data[i])[1]
  end
    
  -2δ1/N*single_integral+δ2
end
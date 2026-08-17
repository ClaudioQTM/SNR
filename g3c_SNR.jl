using LinearAlgebra
using DifferentialEquations
using SteadyStateDiffEq
using SparseArrays
using Distributions


# global variables always change, which slows the code
println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const steps = 10000
const Δt = tot_t / steps
const β = 0.05
const Γtot = 1.0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0

const α = sqrt(0.5)  # Actually it is α/√(L) in the paper
const N = 2   
const P_in  = abs(α)^2
const P_sat = Γtot/β
const resol = 101               # ODE solver saves the values at 101 time points including the initial time
const Δt = tot_t/(resol-1)
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)

# Beam splitter parameters
detector_label = [1,2,3]
BS_prob = [1/3,1/3,1/3]
BS = DiscreteNonParametric(detector_label, BS_prob) # random variables to determine which branch the photon goes into

# detector parameters
detector_state = [0,0,0] # 0 represents Ready state and 1 represents Dead state
τdd = 2 / Γtot # this value is taken from the 2nd wiseman paper

# operators
const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const σp = 1/2*(σx+1im*σy)
const σm = 1/2*(σx-1im*σy)
const θ  = 0.0


global σp_full = Vector{Matrix}(undef,N)
global σm_full = Vector{Matrix}(undef,N)


for k in 1:N
    set_p = [Matrix{ComplexF64}(I, 2, 2) for _ in 1:N]
    set_m = [Matrix{ComplexF64}(I, 2, 2) for _ in 1:N]
    set_p[k] = σp
    set_m[k] = σm
    σp_k = 1
    σm_k = 1
    for j in 1:N
        σp_k = kron(σp_k,set_p[j])
        σm_k = kron(σm_k,set_m[j])
    end
    σp_full[k] = σp_k
    σm_full[k] = σm_k
end

σp_full = sparse(σp_full)
σm_full = sparse(σm_full)

global ρ0 = 1
for k in 1:N
    ρ0_atom = [0 0; 0 1]      # each atom is initialized in the ground state
    global ρ0 = kron(ρ0_atom,ρ0)
end
ρ0 = convert(Matrix{ComplexF64},ρ0)


# define commutators
com(A,B) = A*B-B*A
D(x,ρ)= x*ρ*x'-1/2*(x'*x*ρ+ρ*x'*x)
sum1 = sum(σm_full+σp_full)
sum1 = sparse(sum1)

if N == 1 
    sum2 = 0
else
    sum2 = sum(σp_full[l]*σm_full[j]-σp_full[j]*σm_full[l] for j in 1:N for l in 1:j-1)
    sum2 = sparse(sum2)
end


sum3 = sum(σm_full)
sum3 = sparse(sum3)



function EOM!(dρ,ρ,p,t)
    Γtot,β,sum1,sum2,sum3,P_in,P_sat = p # Localizing variables helps to ensure type stability
    dρ.= Γtot*(-1im*sqrt(P_in/P_sat)*com(sum1,ρ)+(1-β)*sum(x -> D(x,ρ),σm_full)+β/2*com(sum2,ρ)+β*D(sum3,ρ))
    nothing
end



# Steady State Problem
prob_ss = SteadyStateProblem{true}(EOM!,ρ0,(Γtot,β,sum1,sum2,sum3,P_in,P_sat))
sol_ss = solve(prob_ss, DynamicSS(nothing),abstol=1e-10,reltol=1e-8) # The default alg of nothing works only if DifferentialEquations.jl is installed and loaded.

println("Steady state is obtained")
 
a_out = α*I - 1im*sqrt(γ)*sum(σm_full)
n_out = a_out' * a_out

out_power = tr(a_out'*a_out*sol_ss.u)


function curlJ(B,ρ)
    val = B * ρ * B'
    return val
end

function curlH(A,B)
    C = A * B + B * A'
    val = C - tr(C) * B
    return val
end

function curlg(A,B)
    J = curlJ(A,B)
    val = J / tr(J) - B
    return val
end

H_prime =0.5 * Γtot * sqrt(P_in/P_sat) * sum1 + β/2 * Γtot * sum2

function SME(H_prime,ρ,dN_val)
    individual_decay = (1-β)*Γtot*sum(x -> D(x,ρ),σm_full)
    RHS = -curlH(1im * H_prime+ 0.5 * n_out,ρ) + dN_val * curlg(a_out,ρ) + individual_decay
    return RHS
    
"""generate a Bernoulli random variable with the parameters determined by the real time photon flux and length of time interval dt"""
function dN(ρ)
    λ = tr(n_out * ρ)
    p = Bernoulli(λ*Δt)
    val = rand(p)
    return val















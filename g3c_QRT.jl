using LinearAlgebra
using DifferentialEquations
using SteadyStateDiffEq
using SparseArrays
using Base.Threads
using Plots
using Measures
using JLD2

# global variables always change, which slows the code
println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const β = 0.05
const Γtot = 1.0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0

const α = sqrt(0.5)  # Actually it is α/√(L) in the paper
const N = 8   
const P_in  = abs(α)^2
const P_sat = Γtot/β
const resol = 101               # ODE solver saves the values at 101 time points including the initial time
const Δt = tot_t/(resol-1)
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)


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
#=
    for j in N:-1:1                # the ordering of the tensor producted space is H_1 \otimes H_2 .... \otimes H_n
        σp_k = kron(set_p[j],σp_k)
        σm_k = kron(set_m[j],σm_k)
    end
=#
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

out_power = tr(a_out'*a_out*sol_ss.u)

function DynMap(ρ_in::Matrix,O_L,O_R,t_i,t_f)
    ρ_new = O_L*ρ_in*O_R
    prob = ODEProblem{true}(EOM!,ρ_new,(0.0,round((t_f-t_i)*100)/100),(Γtot,β,sum1,sum2,sum3,P_in,P_sat)) # in-place form is true
    sol  = solve(prob,saveat = 0.0:Δt:round((t_f-t_i)*100)/100,abstol = 1e-12,reltol = 1e-10)
    return sol
end


Step1 = DynMap(sol_ss.u,a_out,a_out',0.0,tot_t)

#=
# compute twp-point functions 

aτa0 = DynMap(sol_ss.u,a_out,I,0.0,tot_t)
ad0aτ = DynMap(sol_ss.u,I,a_out',0.0,tot_t)

XX = 0.25*2*real([tr(a_out*(aτa0[k]+ad0aτ[k])) for k in 1:resol])

ΔXΔX = XX.- avg_X^2

println("two-point functions are obtained")



# compute three-point functions <X(t)X(τ)X(0)>

solset_aaρτ = [DynMap(aτa0.u[k],a_out,I,aτa0.t[k],tot_t) for k in 1:resol]

solset_aρτadtmτ = [DynMap(aτa0.u[k],I,a_out',aτa0.t[k],tot_t) for k in 1:resol]

solset_aρadτtmτ = [DynMap(ad0aτ.u[k],a_out,I,ad0aτ.t[k],tot_t) for k in 1:resol]


ataτa0 = zeros(ComplexF64, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        ataτa0[i,i+j-1] = tr(a_out*solset_aaρτ[i].u[j])
    end
end

adτata0 = zeros(ComplexF64, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        adτata0[i,i+j-1] = tr(a_out*solset_aρτadtmτ[i].u[j])
    end
end

adtaτa0 = zeros(ComplexF64, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        adtaτa0[i,i+j-1] = tr(a_out'*solset_aaρτ[i].u[j])
    end
end

ad0ataτ = zeros(ComplexF64, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        ad0ataτ[i,i+j-1] = tr(a_out*solset_aρadτtmτ[i].u[j])
    end
end

XXX = 1/8*2*real(exp(3im*θ)*ataτa0+exp(1im*θ)*(adτata0+adtaτa0+ad0ataτ))




ΔXΔXΔX = zeros(ComplexF64, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        ΔXΔXΔX[i,i+j-1] = XXX[i,i+j-1] - avg_X*(XX[i]+XX[j]+XX[i+j-1])+2*avg_X^3
    end
end

ΔXΔXΔX = real(ΔXΔXΔX + transpose(ΔXΔXΔX) - diagm(diag(ΔXΔXΔX)))
ΔXΔXΔX = ΔXΔXΔX/abs(α)^3
heatmap(ΔXΔXΔX,aspect_ratio=1,xlims=(0,100))


# Specify the file path
#file_path = joinpath("/Users/wangyangming/Documents/Fiber&Atoms/Julia/alpha0.05", "8DXDXDX.jld2")


=#

#=
tot_t = 10
resol = 100
θ = 0.0
Γtot = 1
Δt = tot_t/resol

ϕ(t1,t2) =  -16*exp(-Γtot/2*t1)
ΔXΔXΔX2(t1,t2) = real(exp(3*1im*θ)*ϕ(t1,t2))

z = zeros(resol,resol)

for c in 1:resol
    for r in 1:c
        z[r,c] = ΔXΔXΔX2(c*Δt,r*Δt)
    end
end

z = z + transpose(z) - diagm(diag(z))

heatmap(0:tot_t/resol:tot_t, 0:tot_t/resol:tot_t, z)


=#











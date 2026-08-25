using LinearAlgebra
using DifferentialEquations, SteadyStateDiffEq
using SparseArrays
using Base.Threads


# global variables always change, which slows the code
println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const β = 0.05f0
const Γtot = 1.0f0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0f0

const α = sqrt(0.3)  # Actually it is α/√(L) in the paper
const N = 8
const P_in  = abs(α)^2
const P_sat = Γtot/β
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)


# operators
const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const σp = sparse([0 1; 0 0])    # raising operator, sparse
const σm = sparse([0 0; 1 0])    # lowering operator, sparse
const id2 = sparse(I, 2, 2)                # 2x2 sparse identity
const idN = spdiagm(0 => ones(ComplexF64,2^N))  # N-qubit identity sparse

global L = spzeros(ComplexF64, 2^(2*N), 2^(2*N))

σp_full = Vector{SparseMatrixCSC{Int64,Int64}}(undef, N)
σm_full = Vector{SparseMatrixCSC{Int64,Int64}}(undef, N)

for k in 1:N
    set_p = [id2 for _ in 1:N]
    set_m = [id2 for _ in 1:N]
    set_p[k] = σp
    set_m[k] = σm
    σp_full[k] = reduce(kron, set_p)
    σm_full[k] = reduce(kron, set_m)
end

sum1 = sum(σm_full+σp_full)
sum1 = sparse(sum1)

# define commutators
com(A) = kron(A, idN) - kron(idN, transpose(A))  # sparse commutator operator

term1 = -1im*sqrt(P_in/P_sat)*com(sum1)

L += term1

term1 = nothing

if N == 1 
    sum2 = 0
else
    sum2 = sum(σp_full[l]*σm_full[j]-σp_full[j]*σm_full[l] for j in 1:N for l in 1:j-1)
end

term3 = β/2 * com(sum2)
L += term3
term3 = nothing


D(x)= kron(x,idN)*kron(idN,transpose(x'))-1f0/2f0*(kron(x'*x,idN)+kron(idN,transpose(x'*x)))

term2 = (1-β) * sum(D(σm_full[k]) for k in 1:N)
L += term2
term2 = nothing


term4 = β * D(sum(σm_full))
L += term4
term4 = nothing



L = Γtot*L



global ρ0 = 1
for k in 1:N
    ρ0_atom = [0 0; 0 1]      # each atom is initialized in the ground state
    global ρ0 = kron(ρ0_atom,ρ0)
end
ρ0 = convert(Matrix{ComplexF64},ρ0)
ρ0_v = vec(copy(transpose(ρ0)))
ρ0 = nothing


EOM_v!(dρ,ρ,p,t) = mul!(dρ,L,ρ)


# Steady State Problem
prob_ss = SteadyStateProblem{true}(EOM_v!,ρ0_v)
sol_ss = solve(prob_ss, DynamicSS(Tsit5()),abstol=1e-8,reltol=1e-6)

println("Steady state is obtained")

ρ_ss = Matrix(transpose(reshape(sol_ss.u,2^(N),2^(N))))
sol_ss = nothing

a_out = α*idN - 1im*sqrt(γ)*sum(σm_full)


out_power = tr(a_out'*a_out*ρ_ss)
EV_a = tr(a_out*ρ_ss)


G3 = tr(a_out'*a_out'*a_out'* a_out * a_out * a_out * ρ_ss)

G2 = tr(a_out'*a_out'* a_out * a_out * ρ_ss)
 

G3c =  2 * out_power^3 + G3 - 3 * G2 * out_power


# calculate <δa(t1)δa(t2)>  and <δa†(t1)δa*(t2)>
global aa  = tr(a_out*a_out*ρ_ss)
global ada = tr(a_out'*a_out*ρ_ss)


# <δa(0)δa(0)>
δaδa = aa - EV_a^2
# <δa†(t1)δa(t2)>
δadδa= ada - abs(EV_a)^2


noαterm = δadδa*δadδa*conj(δadδa) + conj(δaδa)*δaδa*conj(δadδa) + conj(δaδa)*δaδa*conj(δadδa)+conj(δaδa)*δaδa*conj(δadδa)

αsq_term = conj(δaδa)*(conj(δadδa)+conj(δadδa)) + conj(δaδa)*(conj(δadδa)+δadδa) + conj(δaδa)*(δadδa+δadδa)

αabssq_term = conj(δadδa)*δadδa + δadδa*δadδa + δadδa*conj(δadδa) +conj(δaδa)*δaδa + conj(δaδa)*δaδa + conj(δaδa)*δaδa

gaussian_G3c = 2*real(noαterm + EV_a^2 * αsq_term + abs(EV_a)^2 * αabssq_term)

gaussian_G3 = gaussian_G3c + 3 * G2 * out_power - 2*out_power^3

G3_diff = G3 - gaussian_G3


println("G3:$G3")
println()
println("G3 Isserlis:$gaussian_G3")
println()
println("G3_diff:$G3_diff")



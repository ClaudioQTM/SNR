using LinearAlgebra
using DifferentialEquations, SteadyStateDiffEq
using SparseArrays
using Base.Threads
using CUDA
using Plots
using Measures
using DiffEqCallbacks
using JLD2

# global variables always change, which slows the code
println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const β = 0.05f0
const Γtot = 1.0f0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0f0

const α = sqrt(0.8f0)  # Actually it is α/√(L) in the paper
const N = 2
const P_in  = abs(α)^2
const P_sat = Γtot/β
const resol = 101               # ODE solver saves the values at 101 time points including the initial time
const Δt = tot_t/(resol-1)
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)


# operators
const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const σp = sparse([0 1; 0 0])    # raising operator, sparse
const σm = sparse([0 0; 1 0])    # lowering operator, sparse
const id2 = sparse(I, 2, 2)                # 2x2 sparse identity
const idN = spdiagm(0 => ones(ComplexF32,2^N))  # N-qubit identity sparse

global L = spzeros(ComplexF32, 2^(2*N), 2^(2*N))

σp_full = Vector{SparseMatrixCSC{Int32,Int32}}(undef, N)
σm_full = Vector{SparseMatrixCSC{Int32,Int32}}(undef, N)

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
L = CUSPARSE.CuSparseMatrixCSC(L)



global ρ0 = 1
for k in 1:N
    ρ0_atom = [0 0; 0 1]      # each atom is initialized in the ground state
    global ρ0 = kron(ρ0_atom,ρ0)
end
ρ0 = convert(Matrix{ComplexF32},ρ0)
ρ0_v = reshape(transpose(ρ0) ,2^(2*N))
ρ0 = nothing
ρ0_v = CuArray(ρ0_v)  # move initial state vector to GPU

#=
# define commutators
com_m(A,B) = A*B-B*A
D_m(x,ρ)= x*ρ*x'-1/2*(x'*x*ρ+ρ*x'*x)




sum3 = sum(σm_full)
sum3 = sparse(sum3)

ρ1 = Γtot*(-1im*sqrt(P_in/P_sat)*com_m(sum1,ρ0)+(1-β)*sum(x -> D_m(x,ρ0),σm_full)+β/2*com_m(sum2,ρ0)+β*D_m(sum3,ρ0))


ρ1 = reshape(transpose(ρ1),2^(2*N))    

L*ρ0_v ≈ ρ1
=#


#=
function EOM_v!(dρ,ρ,p,t)
    # Unpack the Liouvillian from the parameter tuple to avoid elementwise comparisons
    L = p
    dρ .= L * ρ
    nothing
end

=#
EOM_v!(dρ,ρ,p,t) = mul!(dρ,L,ρ)


# Steady State Problem (pass L inside a tuple to skip DiffEqBase's isequal on GPU sparse)
prob_ss = SteadyStateProblem{true}(EOM_v!,ρ0_v)   # Do not pass L to p inside a tuple to avoid DiffEqBase's isequal on GPU sparse
sol_ss = solve(prob_ss, DynamicSS(Tsit5()),abstol=1e-8,reltol=1e-6)

println("Steady state is obtained")

ρ_ss = transpose(reshape(Array(sol_ss.u),2^(N),2^(N)))
sol_ss = nothing

a_out = α*idN - 1im*sqrt(γ)*sum(σm_full)


out_power = tr(a_out'*a_out*ρ_ss)
EV_a = tr(a_out*ρ_ss)

#=
function DynMap(ρ_in,O_L,O_R,t_i,t_f)
    ρ_new = Array(ρ_in)
    ρ_new = transpose(reshape(ρ_new,2^(N),2^(N)))
    ρ_new = O_L*ρ_new*O_R
    ρ_new = reshape(transpose(ρ_new),2^(2*N))
    ρ_new = CuArray(ρ_new)
    prob = ODEProblem{true}(EOM_v!,ρ_new,(0.0,round((t_f-t_i)*100)/100)) # in-place form is true
    sol  = solve(prob, Tsit5();
    saveat = 0.0:Δt:round((t_f-t_i)*100)/100,
                abstol=1e-6, reltol=1e-4)
    return sol
end
=#

ch1 = Channel{Vector{ComplexF32}}(resol)


function DynMap(ρ_in,O_L,O_R,t_i,t_f,c::Channel)
    time_span = round((t_f - t_i) * 100) / 100
    
    ρ_new = O_L*ρ_in*O_R
    ρ_new = reshape(transpose(ρ_new),2^(2*N))
    ρ_new = CuArray(ρ_new)

    saved_values = SavedValues(Float64,Array{ComplexF32,1})
    cb = SavingCallback((u,t,integrator)-> put!(c,Array(u)),saved_values,
    saveat=0.0:Δt:time_span,save_everystep=false,
    save_start=true)

    prob = ODEProblem{true}(EOM_v!,ρ_new,(0.0,round((t_f-t_i)*100)/100)) # in-place form is true
    sol = solve(prob, Tsit5();
            callback=cb,
            abstol=1e-9, reltol=1e-7
            )
end

global ρ_Step1 = Vector{Array{ComplexF32,2}}(undef, resol)

global G2 = Vector{Float32}(undef, resol)



@sync begin
  # Producer task: generate vectors and put! into ch1
  @async begin
    DynMap(ρ_ss, a_out, a_out', 0.0, tot_t, ch1)
    close(ch1)   # signal "no more data"
  end

  # Consumer task: pull from ch1 as soon as items arrive
  @async begin
    k = 1
    for vec in ch1            # iterates until ch1 is closed
      ρ_Step1[k] = transpose(reshape(vec, 2^N, 2^N))
      G2[k]       = tr(a_out'*a_out*ρ_Step1[k])
      k += 1
    end
  end
end



G3 = zeros(ComplexF32, resol, resol)

# Create time points array
t_points = [k * Δt for k in 0:(resol-1)]


for i in 1:resol
    ch_tmp = Channel{Vector{ComplexF32}}(resol)
    @sync begin
        # Producer: solve dynamics from t[i] to tot_t
        @async begin
            DynMap(ρ_Step1[i], a_out, a_out', t_points[i], tot_t, ch_tmp)
            close(ch_tmp)
        end

        # Consumer: compute G3 values
        @async begin
            j = 1
            for vec in ch_tmp
                if i+j-1 <= resol  # bounds check
                    ρ_tmp = transpose(reshape(vec,2^(N),2^(N)))
                    G3[i,i+j-1] = tr(a_out'*a_out*ρ_tmp)
                end
                j += 1
            end
        end
    end
end


g3c = zeros(ComplexF32, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        g3c[i,i+j-1] = 2 + G3[i,i+j-1]/out_power^3 - (G2[i]+G2[j]+G2[i+j-1])/out_power^2
    end
end

g3c = real(g3c + transpose(g3c) - diagm(diag(g3c)))



# calculate <δa(t1)δa(t2)>  and <δa†(t1)δa*(t2)>
global aa  = Vector{Float32}(undef, resol)
global ada = Vector{Float32}(undef, resol)
ch_aρ = Channel{Vector{ComplexF32}}(resol)
global aρ = Vector{Array{ComplexF32,2}}(undef, resol)

@sync begin

    @async begin
      DynMap(ρ_ss, a_out, I, 0.0, tot_t, ch_aρ)
      close(ch_aρ)  
    end
  
    @async begin
      k = 1
      for vec in ch_aρ            
        aρ[k] = transpose(reshape(vec, 2^N, 2^N))
        aa[k]       = tr(a_out*aρ[k])
        ada[k]      = tr(a_out'*aρ[k])
        k += 1
      end
    end
end

# <δa(t1)δa(t2)>
δaδa = [aa[i]-EV_a^2 for i in 1:resol]
# <δa†(t1)δa(t2)>
δadδa = [ada[i]-conj(EV_a)*EV_a for i in 1:resol]

gaussian_G3c = zeros(Float32,resol,resol)

for i in 1:resol
    for j in 1:(resol-i+1)
        noαterm = δadδa[i]*δadδa[j]*conj(δadδa[i+j-1]) + conj(δaδa[i])*δaδa[j]*conj(δadδa[i+j-1]) + conj(δaδa[i])*δaδa[i+j-1]*conj(δadδa[j])+conj(δaδa[i+j-1])*δaδa[j]*conj(δadδa[i])

        αsq_term = conj(δaδa[i])*(conj(δadδa[i+j-1])+conj(δadδa[j])) + conj(δaδa[i+j-1])*(conj(δadδa[i])+δadδa[j]) + conj(δaδa[j])*(δadδa[i]+δadδa[i+j-1])

        αabssq_term = conj(δadδa[i])*δadδa[i+j-1] + δadδa[i]*δadδa[j] + δadδa[i+j-1]*conj(δadδa[j]) +conj(δaδa[i])*δaδa[i+j-1] + conj(δaδa[i])*δaδa[j] + conj(δaδa[j])*δaδa[i+j-1]

        gaussian_G3c[i,i+j-1] = 2*real(noαterm + EV_a^2 * αsq_term + abs(EV_a)^2 * αabssq_term)
    end
end


gaussian_G3c = gaussian_G3c + transpose(gaussian_G3c) -diagm(diag(gaussian_G3c))

gaussian_g3c = real(1/out_power^3  * gaussian_G3c) 

g3c_diff = g3c-gaussian_g3c
#file_path = joinpath("/Users/ywan8652/Documents/Fiber&Atoms/Julia/identity violation/g3cGausPin"*string(replace(string(round(P_in, digits=2)), "." => ""))*"/", "$N"*"g3c.jld2")
#jldsave(file_path;g3c)

file_path = joinpath("D:/iCloudDrive/Documents/Fiber&Atoms/Julia/identity violation/g3cDiffPin"*string(replace(string(round(P_in,digits=2)),"."=>""))*"/","$N"*"g3cDiff.jld2")
jldsave(file_path;g3c_diff)
# Plot G3 using heatmap
#heatmap(gaussian_g3c,right_margin=10mm)

heatmap(g3c_diff,right_margin=10mm)



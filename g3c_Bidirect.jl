using LinearAlgebra, Random, Distributions
using DifferentialEquations, SteadyStateDiffEq
using SparseArrays
using Base.Threads
using CUDA
using Plots
using Measures
using DiffEqCallbacks
#using JLD2

# global variables always change, which slows the code
println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const β = 0.05f0
const ϵ = 0.05f0
const Γtot = 1.0f0
const γR = β * Γtot
const γL = ϵ * Γtot
const Γ = (1-β-ϵ)*Γtot                      # make sure that sqrt(β) << 1
const d = 448  # the average distance between atoms, unit is nm
const λ_0 = 852 # the wavelength of probe laser, unit is nm
const dtoλ_0 = d/λ_0
const η = 0.25  # parameter for disorder
const α = sqrt(0.1f0)  # Actually it is α/√(L) in the paper
const N = 4
const P_in = abs(α)^2
const P_sat = Γtot/β
const resol = 101               # ODE solver saves the values at 101 time points including the initial time
const Δt = tot_t/(resol-1)


# operators
const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const σp = sparse([0 1; 0 0])    # raising operator, sparse
const σm = sparse([0 0; 1 0])    # lowering operator, sparse
const id2 = sparse(I, 2, 2)                # 2x2 sparse identity
const idN = spdiagm(0 => ones(ComplexF32, 2^N))  # N-atom identity sparse

global L = spzeros(ComplexF32, 2^(2*N), 2^(2*N))

# generate a list of i.i.d. Gaussian random variables
seed = 123
φ = zeros(ComplexF32, N)

R = zeros(Float32, N)
for j in 1:N
    Random.seed!(seed+j-1)
    R[j] = rand(Normal(0, 1))
end

z = [j + η*R[j] for j in 1:N]

if !all(diff(z) .> 0)
    error("atoms are not ordered")
end

for j in 1:N
    φ[j] = 2im * 2 * pi * dtoλ_0 * z[j]
end



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
    sum2 = spzeros(ComplexF32, 2, 2)
else
    sum2 = β/2 * sum(σp_full[l]*σm_full[j]-σp_full[j]*σm_full[l] for j in 1:N for l in 1:(j-1))
    sum2 += ϵ/2 * sum(σp_full[l]*σm_full[j]*exp(φ[j]-φ[l])-σp_full[j]*σm_full[l]*exp(-(φ[j]-φ[l])) for l in 1:N for j in 1:(l-1))
end

term3 = com(sum2)
L += term3
term3 = nothing



D(x) = kron(x, idN)*kron(idN, conj(x))-1f0/2f0*(kron(x'*x, idN)+kron(idN, transpose(x'*x)))

term2 = (1-β-ϵ) * sum(D(σm_full[k]) for k in 1:N)
L += term2
term2 = nothing


term4 = β * D(sum(σm_full))
L += term4
term4 = nothing

term5 = ϵ * D(sum(σm_full[i]*exp(φ[i]) for i in 1:N))
L += term5
term5 = nothing

L = Γtot*L
# Float64 positions promote the assembled L to ComplexF64; match the ComplexF32 states.
L = CUSPARSE.CuSparseMatrixCSC{ComplexF32}(L)



global ρ0 = 1
for k in 1:N
    ρ0_atom = [0 0; 0 1]      # each atom is initialized in the ground state
    global ρ0 = kron(ρ0_atom, ρ0)
end
ρ0 = convert(Matrix{ComplexF32}, ρ0)
ρ0_v = reshape(transpose(ρ0), 2^(2*N))
ρ0 = nothing
ρ0_v = CuArray(ρ0_v)  # move initial state vector to GPU


EOM_v!(dρ, ρ, p, t) = mul!(dρ, L, ρ)


# Steady State Problem (pass L inside a tuple to skip DiffEqBase's isequal on GPU sparse)
prob_ss = SteadyStateProblem{true}(EOM_v!, ρ0_v)   # Do not pass L to p inside a tuple to avoid DiffEqBase's isequal on GPU sparse
sol_ss = solve(prob_ss, DynamicSS(Tsit5()), abstol=1e-8, reltol=1e-6)

println("Steady state is obtained")

ρ_ss = transpose(reshape(Array(sol_ss.u), 2^(N), 2^(N)))
sol_ss = nothing

a_out = α*idN - 1im*sqrt(γR)*sum(σm_full)


out_power = tr(a_out'*a_out*ρ_ss)
EV_a = tr(a_out*ρ_ss)



ch1 = Channel{Vector{ComplexF32}}(resol)


function DynMap(ρ_in, O_L, O_R, t_i, t_f, c::Channel)
    time_span = round((t_f - t_i) * 100) / 100

    ρ_new = O_L*ρ_in*O_R
    ρ_new = reshape(transpose(ρ_new), 2^(2*N))

    if iszero(time_span) # avoid t_i==t_f 
        put!(c, Array(ρ_new))
        return nothing
    end

    ρ_new = CuArray(ρ_new)

    saved_values = SavedValues(Float64, Array{ComplexF32,1})
    cb = SavingCallback((u, t, integrator) -> put!(c, Array(u)), saved_values,
        saveat=0.0:Δt:time_span, save_everystep=false,
        save_start=true)

    prob = ODEProblem{true}(EOM_v!, ρ_new, (0.0, round((t_f-t_i)*100)/100)) # in-place form is true
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
            G2[k] = real(tr(a_out'*a_out*ρ_Step1[k]))
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
                    ρ_tmp = transpose(reshape(vec, 2^(N), 2^(N)))
                    G3[i, i+j-1] = tr(a_out'*a_out*ρ_tmp)
                end
                j += 1
            end
        end
    end
end


g3c = zeros(ComplexF32, resol, resol)
Threads.@threads for i in 1:resol
    for j in 1:(resol-i+1)
        g3c[i, i+j-1] = 2 + G3[i, i+j-1]/out_power^3 - (G2[i]+G2[j]+G2[i+j-1])/out_power^2
    end
end

g3c = real(g3c + transpose(g3c) - diagm(diag(g3c)))


heatmap(g3c, right_margin=10mm)
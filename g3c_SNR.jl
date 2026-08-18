using LinearAlgebra
using DifferentialEquations
using SteadyStateDiffEq
using SparseArrays
using Distributions
using Plots
using Random



println(Threads.nthreads()) # check the number of threads

const tot_t = 50.0               # total time, data type should be float.
const steps = Int(1e3)
const Δt = tot_t / steps
const n_traj = 50               # the number of quantum trajectories
const β = 0.3
const Γtot = 1.0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0             # detuning of the input photons

const α = sqrt(0.5)  # Actually it is α/√(L) in the paper
const N = 2
const P_in  = abs(α)^2
const P_sat = Γtot/β
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)


# Beam splitter parameters
detector_label = [1,2,3]
BS_prob = [1/3,1/3,1/3]
BS = DiscreteNonParametric(detector_label, BS_prob) # random variables to determine which branch the photon goes into

# detector parameters
detector_state = [0,0,0] # 0 represents Ready state and 1 represents Dead state
τdd = 2 / Γtot # this value is taken from wiseman's 2nd paper
t_bin = 3 / Γtot # the length of time bin for defining the three-photon coincidence event. The value is taken from Section D from our long paper.

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
anticom(A,B) = A*B+B*A
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


function curlJ(B,ρ) #jump operator
    val = B * ρ * B'
    return val
end


H_prime = 0.5 * Γtot * sqrt(P_in/P_sat) * sum1 + 1im* β/2 * Γtot * sum2

function L0(H_prime,ρ)
    individual_decay = (1-β)*Γtot*sum(x -> D(x,ρ),σm_full)
    L0ρ = -1im * com(H_prime,ρ) - 0.5 * anticom(n_out,ρ) + individual_decay
    return L0ρ
end

function L1(ρ)
    ρ_unnorm = curlJ(a_out,ρ)
    return ρ_unnorm
end

    
"""generate a Bernoulli random variable with the parameters determined by the real time photon flux and length of time interval dt"""
function dN(ρ, rng::AbstractRNG=Random.default_rng())
    λ = tr(n_out * ρ)
    if abs(imag(λ)) > 1e-10
        error("Photon flux has a significant imaginary part")
    end
    λ = real(λ) # make sure that λ is a real number
    if λ < 0.0
        error("Negative photon flux: λ = $λ")
    end
    p = λ * Δt
    if !(0.0 <= p <= 1.0)
        error("Invalid jump probability λΔt = $p")
    end
    x = Bernoulli(p)
    val = rand(rng, x)
    return val
end


global ρt0 = sol_ss.u

function trajectory(ρt0, rng::AbstractRNG=Random.default_rng())
    ρt = copy(ρt0)
    emission_record = zeros(Int,steps)
    for tt in 1:steps
        dN_val = dN(ρt, rng)

        if dN_val == 1
        ρt = L1(ρt)

        elseif dN_val == 0
            dρt = L0(H_prime,ρt)
            ρt = ρt + dρt*Δt
        else
            error("dN_val is not 0 or 1")
        end

        ρt = ρt / tr(ρt) # re-normalize the density matrix
        emission_record[tt] = dN_val
    end
    
    return ρt, emission_record
end

function trajectories_parallel(ρt0, n::Integer; seed::Integer=1234)
    n > 0 || throw(ArgumentError("number of trajectories must be positive"))
    
    final_states = Vector{typeof(ρt0)}(undef, n)
    emission_record = Vector{Vector{Int}}(undef, n)
    Threads.@threads for i in 1:n
        # Each trajectory owns its RNG, so execution is thread-safe and
        # reproducible even when the thread scheduler changes the run order.
        rng = Xoshiro(seed + i)
        final_states[i],emission_record[i] = trajectory(ρt0, rng)
    end

    return final_states, emission_record
end


function BS_branch_selector(emission_history,rng::AbstractRNG = Random.default_rng())
    detector_rec = zeros(Int, steps)
    detector_state = [(0.0,0),(0.0,0),(0.0,0)] # in each tuple, the second element is the state of the detector (0 for Ready, 1 for Dead). the first element is the time stamp of the last detection if the detector state is dead.
    for tt in 1:steps
        for detector_label in 1:3
            if detector_state[detector_label][2] == 1 && (tt - detector_state[detector_label][1]) * Δt > τdd
                detector_state[detector_label] = (0.0,0) # reset the detector state to Ready
            end
        end
        dN = emission_history[tt]
        if dN == 1
            branch_label = rand(rng, BS)
            if detector_state[branch_label][2] == 0
                detector_rec[tt] = branch_label
                detector_state[branch_label] = (tt,1) # record the time stamp of the detection and change the state to Dead
            end
        end
    end
    return detector_rec
end


function BS_branch_selector_parallel(emission_histories; seed::Integer=1234)
    n = length(emission_histories)
    n > 0 || throw(ArgumentError("emission histories must not be empty"))

    detector_records = Vector{Vector{Int}}(undef, n)
    Threads.@threads for i in eachindex(emission_histories)
        # Give each trajectory its own RNG so branch selection is thread-safe
        # and reproducible independently of the thread scheduling order. The
        # offset avoids reusing trajectories_parallel's RNG streams.
        rng = Xoshiro(seed + n + i)
        detector_records[i] = BS_branch_selector(emission_histories[i], rng)
    end

    return detector_records
end



final_state_list, emission_histories = trajectories_parallel(ρt0, n_traj)
branch_record = BS_branch_selector_parallel(emission_histories)
#println(emission_histories) # test the trajectory function
#println(size(branch_record))


"""We split the entire simuation time tot_t into several time bins. In each time bin, we count the presence of the detected three photon coincidence."""
function three_photon_coincidence_counter(branch_record,t_bin)
    bin_step_size = Int(t_bin/Δt)
    coincidence_count = 0
    bin_number = Int(steps/bin_step_size)

    for b in 1:bin_number

        for tt in 1:bin_step_size
            
        end
    end






    


#=
final_states = trajectories_parallel(ρt0, n_traj)
ρ_mean = reduce(+, final_states) / n_traj
println(ρ_mean)

relative_error = norm(ρ_mean - sol_ss.u) / norm(sol_ss.u)
println("Relative Frobenius error: ", relative_error)
=#


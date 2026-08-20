using LinearAlgebra
using DifferentialEquations
using SteadyStateDiffEq
using SparseArrays
using Distributions
using Plots
using Random



println(Threads.nthreads()) # check the number of threads

const tot_t = 50.0               # total time, data type should be float.
const steps = Int(5e4)
const Δt = tot_t / steps
const n_traj = 1000              # the number of quantum trajectories
const β = 0.05
const Γtot = 1.0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0             # detuning of the input photons

const α = sqrt(0.08)  # Actually it is α/√(L) in the paper
const N = 2
const P_in  = abs(α)^2
const P_sat = Γtot/β
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)


# Beam splitter parameters
detector_label = [1,2,3]
const BS_prob = [1/3,1/3,1/3]
BS = DiscreteNonParametric(detector_label, BS_prob) # random variables to determine which branch the photon goes into

# detector parameters
detector_state = [0,0,0] # 0 represents Ready state and 1 represents Dead state
#τdd = 2 / Γtot # this value is taken from wiseman's 2nd paper
τdd = 0.0
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
third_order_I = a_out'*a_out'*a_out'*a_out*a_out*a_out

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

    
"""generate a Bernoulli random variable with the parameters determined by the real time photon flux and length of time interval Δt"""
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


"""
Estimate a finite-bin third-order correlation from one registered
detector record.

The record convention is:

    record[k] == 0  : no registered click in time step k
    record[k] == 1  : detector 1 clicked
    record[k] == 2  : detector 2 clicked
    record[k] == 3  : detector 3 clicked

For each non-overlapping bin b, the function computes

    n1_b * n2_b * n3_b,

where ni_b is the number of clicks registered by detector i in that
bin.

The returned `G3_apparent` is corrected using the beam-splitter
probabilities in `BS_prob` and assumes ideal intrinsic detector efficiency.
It is not corrected for detector dead time. Therefore, when `record`
contains dead-time-filtered events, it is the apparent measured G^(3), not
an unbiased reconstruction of the ideal source G^(3).
"""
function three_photon_coincidence_counter(
    record::AbstractVector{<:Integer},
    t_bin::Real,
)
    t_bin > 0 ||
        throw(ArgumentError("t_bin must be positive"))

    length(BS_prob) == 3 ||
        throw(ArgumentError("BS_prob must contain three beam-splitter probabilities"))

    BS_prob_values = Float64.(BS_prob)

    all(x -> x > 0, BS_prob_values) ||
        throw(ArgumentError("all beam-splitter probabilities must be positive"))

    isapprox(sum(BS_prob_values), 1.0; atol = 1e-12, rtol = 1e-12) ||
        throw(ArgumentError("beam-splitter probabilities must sum to one"))

    bin_steps = round(Int, t_bin / Δt)

    bin_steps >= 1 ||
        throw(ArgumentError("t_bin must contain at least one time step"))

    Δbin = bin_steps * Δt

    isapprox(Δbin, t_bin; atol = 1e-12, rtol = 1e-10) ||
        throw(
            ArgumentError(
                "t_bin/Δt must be an integer. " *
                "Received t_bin = $t_bin and Δt = $Δt."
            )
        )

    # Use only complete bins. Any incomplete tail is discarded.
    n_bins = fld(length(record), bin_steps)

    n_bins >= 1 ||
        throw(ArgumentError("the record is shorter than one time bin"))

    used_steps = n_bins * bin_steps
    discarded_steps = length(record) - used_steps

    T_eff = used_steps * Δt

    total_n1 = 0
    total_n2 = 0
    total_n3 = 0

    # Sum of n1_b*n2_b*n3_b over all bins.
    triple_weight = 0

    # Diagnostic only: number of bins containing at least one click
    # from each detector.
    coincidence_bins = 0

    for b in 0:(n_bins - 1)
        first_index = b * bin_steps + 1
        last_index = (b + 1) * bin_steps

        n1 = 0
        n2 = 0
        n3 = 0

        @inbounds for k in first_index:last_index
            arm = record[k]

            if arm == 0
                continue
            elseif arm == 1
                n1 += 1
            elseif arm == 2
                n2 += 1
            elseif arm == 3
                n3 += 1
            else
                throw(
                    ArgumentError(
                        "record[$k] = $arm; allowed values are 0, 1, 2, 3"
                    )
                )
            end
        end

        total_n1 += n1
        total_n2 += n2
        total_n3 += n3

        triple_weight += n1 * n2 * n3

        if n1 > 0 && n2 > 0 && n3 > 0
            coincidence_bins += 1
        end
    end

    # Registered triple-coincidence density.
    #
    # Dimension:
    #   triple_weight / (time * time^2) = time^(-3)
    registered_G3_density =
        triple_weight / (T_eff * Δbin^2)

    splitter_probability_factor = prod(BS_prob_values)

    # Source-referred apparent G^(3).
    # Dead-time bias remains present when the input record is
    # dead-time filtered.
    G3_apparent =
        registered_G3_density / splitter_probability_factor

    registered_rate_1 = total_n1 / T_eff
    registered_rate_2 = total_n2 / T_eff
    registered_rate_3 = total_n3 / T_eff

    rate_product =
        registered_rate_1 *
        registered_rate_2 *
        registered_rate_3

    # Normalized correlation of the registered records.
    g3_registered =
        rate_product > 0 ?
        registered_G3_density / rate_product :
        NaN

    return (
        G3_apparent = G3_apparent,
        registered_G3_density = registered_G3_density,
        g3_registered = g3_registered,
        triple_weight = triple_weight,
        coincidence_bins = coincidence_bins,
        coincidence_bin_fraction = coincidence_bins / n_bins,
        singles = (total_n1, total_n2, total_n3),
        registered_rates = (
            registered_rate_1,
            registered_rate_2,
            registered_rate_3,
        ),
        bin_steps = bin_steps,
        bin_width = Δbin,
        number_of_bins = n_bins,
        effective_time = T_eff,
        discarded_steps = discarded_steps,
        discarded_time = discarded_steps * Δt,
    )
end



"""
Estimate finite-bin G^(3) separately for every trajectory.

`std_G3_apparent` is the run-to-run uncertainty for one trajectory
having the simulated acquisition time. It is the relevant quantity
for an experimental SNR forecast.

`sem_G3_apparent` is only the Monte Carlo uncertainty in the estimated
mean.
"""
function estimate_G3_ensemble(
    records::AbstractVector,
    t_bin::Real,
)
    isempty(records) &&
        throw(ArgumentError("records must not be empty"))

    results = [
        three_photon_coincidence_counter(
            record,
            t_bin,
        )
        for record in records
    ]

    G3_values = [
        result.G3_apparent
        for result in results
    ]

    g3_values = [
        result.g3_registered
        for result in results
        if isfinite(result.g3_registered)
    ]

    n = length(G3_values)

    σ_G3 =
        n > 1 ? std(G3_values) : NaN

    sem_G3 =
        n > 1 ? σ_G3 / sqrt(n) : NaN

    # Pool all raw triple counts. This is equivalent to treating all
    # trajectories as one combined acquisition, provided their bin
    # widths are identical.
    total_triple_weight =
        sum(result.triple_weight for result in results)

    total_effective_time =
        sum(result.effective_time for result in results)

    Δbin = results[1].bin_width

    splitter_probability_factor = prod(Float64.(BS_prob))

    pooled_G3_apparent =
        total_triple_weight /
        (
            total_effective_time *
            Δbin^2 *
            splitter_probability_factor
        )

    return (
        pooled_G3_apparent = pooled_G3_apparent,
        mean_G3_apparent = mean(G3_values),
        std_G3_apparent = σ_G3,
        sem_G3_apparent = sem_G3,
        mean_g3_registered =
            isempty(g3_values) ? NaN : mean(g3_values),
        std_g3_registered =
            length(g3_values) > 1 ? std(g3_values) : NaN,
        per_trajectory_G3 = G3_values,
        per_trajectory_results = results,
        total_effective_time = total_effective_time,
    )
end



G3_stats = estimate_G3_ensemble(
    branch_record,
    t_bin,
)

true_G30 = tr(third_order_I*sol_ss.u)
println("True value of G^(3):$true_G30")

println(
    "Pooled apparent G^(3) = ",
    G3_stats.pooled_G3_apparent,
)

println(
    "Mean trajectory G^(3) = ",
    G3_stats.mean_G3_apparent,
)

println(
    "Run-to-run standard deviation = ",
    G3_stats.std_G3_apparent,
)

println(
    "Monte Carlo standard error of the mean = ",
    G3_stats.sem_G3_apparent,
)

println(
    "Mean registered g^(3) = ",
    G3_stats.mean_g3_registered,
)

#=
final_states = trajectories_parallel(ρt0, n_traj)
ρ_mean = reduce(+, final_states) / n_traj
println(ρ_mean)

relative_error = norm(ρ_mean - sol_ss.u) / norm(sol_ss.u)
println("Relative Frobenius error: ", relative_error)
=#


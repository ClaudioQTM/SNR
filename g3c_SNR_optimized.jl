module G3CSNROptimized

using LinearAlgebra
using Random
using SparseArrays
using Statistics

export SimulationConfig,
       SimulationModel,
       TrajectoryStats,
       build_model,
       config_from_environment,
       ensemble_statistics,
       run_ensemble,
       trajectory_statistics,
       trajectory_with_records

Base.@kwdef struct SimulationConfig
    total_time::Float64 = 75.0
    steps::Int = 75_000
    trajectories::Int = 4_000
    beta::Float64 = 0.05
    gamma_total::Float64 = 1.0
    alpha::Float64 = sqrt(0.2)
    atom_count::Int = 5
    detector_dead_time::Float64 = 1
    bin_width::Float64 = 3.0
    splitter_probabilities::NTuple{3, Float64} = (1 / 3, 1 / 3, 1 / 3)
    seed::Int = 1_234
end

struct SimulationModel
    config::SimulationConfig
    dimension::Int
    rho_ss::Matrix{ComplexF64}
    no_jump_map::SparseMatrixCSC{ComplexF64, Int}
    jump_map::SparseMatrixCSC{ComplexF64, Int}
    flux_weights::Vector{ComplexF64}
    third_order_intensity::Matrix{ComplexF64}
    output_power::ComplexF64
end

struct TrajectoryStats
    G3_apparent::Float64
    registered_G3_density::Float64
    g3_registered::Float64
    triple_weight::Int
    coincidence_bins::Int
    coincidence_bin_fraction::Float64
    singles::NTuple{3, Int}
    registered_rates::NTuple{3, Float64}
    bin_steps::Int
    bin_width::Float64
    number_of_bins::Int
    effective_time::Float64
    discarded_steps::Int
    discarded_time::Float64
    emitted_photons::Int
end

time_step(config::SimulationConfig) = config.total_time / config.steps

function validate(config::SimulationConfig)
    config.total_time > 0 || throw(ArgumentError("total_time must be positive"))
    config.steps > 0 || throw(ArgumentError("steps must be positive"))
    config.trajectories > 0 || throw(ArgumentError("trajectories must be positive"))
    config.atom_count > 0 || throw(ArgumentError("atom_count must be positive"))
    0 <= config.beta <= 1 || throw(ArgumentError("beta must lie in [0, 1]"))
    config.gamma_total > 0 || throw(ArgumentError("gamma_total must be positive"))
    config.detector_dead_time >= 0 ||
        throw(ArgumentError("detector_dead_time must be nonnegative"))
    config.bin_width > 0 || throw(ArgumentError("bin_width must be positive"))

    probabilities = config.splitter_probabilities
    all(>(0), probabilities) ||
        throw(ArgumentError("all splitter probabilities must be positive"))
    isapprox(sum(probabilities), 1.0; atol = 1e-12, rtol = 1e-12) ||
        throw(ArgumentError("splitter probabilities must sum to one"))

    bin_steps = round(Int, config.bin_width / time_step(config))
    bin_steps >= 1 || throw(ArgumentError("bin_width must contain at least one step"))
    config.steps >= bin_steps ||
        throw(ArgumentError("the simulation must contain at least one complete bin"))
    isapprox(
        bin_steps * time_step(config),
        config.bin_width;
        atol = 1e-12,
        rtol = 1e-10,
    ) || throw(ArgumentError("bin_width / time_step must be an integer"))

    return config
end

function local_operator(operator::AbstractMatrix, site::Int, atom_count::Int)
    result = Matrix{ComplexF64}(I, 1, 1)
    identity_2 = Matrix{ComplexF64}(I, 2, 2)
    for index in 1:atom_count
        result = kron(result, index == site ? operator : identity_2)
    end
    return result
end

"""Matrix acting on `vec(rho)` for `rho -> left * rho * right`."""
left_right_map(left::AbstractMatrix, right::AbstractMatrix) =
    kron(transpose(right), left)

function commutator_map(operator::AbstractMatrix)
    dimension = size(operator, 1)
    identity_d = Matrix{ComplexF64}(I, dimension, dimension)
    return left_right_map(operator, identity_d) -
           left_right_map(identity_d, operator)
end

function dissipator_map(operator::AbstractMatrix)
    dimension = size(operator, 1)
    identity_d = Matrix{ComplexF64}(I, dimension, dimension)
    number_operator = operator' * operator
    return left_right_map(operator, operator') -
           0.5 * left_right_map(number_operator, identity_d) -
           0.5 * left_right_map(identity_d, number_operator)
end

function trace_weights(dimension::Int)
    weights = zeros(ComplexF64, dimension^2)
    @inbounds for index in 1:dimension
        weights[index + (index - 1) * dimension] = 1
    end
    return weights
end

function solve_steady_state(liouvillian::AbstractMatrix, dimension::Int)
    system = Matrix{ComplexF64}(liouvillian)
    rhs = zeros(ComplexF64, dimension^2)

    # Replace one dependent stationary equation by tr(rho) = 1.
    system[end, :] .= trace_weights(dimension)
    rhs[end] = 1
    rho = reshape(system \ rhs, dimension, dimension)
    rho ./= tr(rho)
    return rho
end

function build_model(config::SimulationConfig = SimulationConfig())
    validate(config)

    sigma_x = ComplexF64[0 1; 1 0]
    sigma_y = ComplexF64[0 -im; im 0]
    sigma_plus = 0.5 * (sigma_x + im * sigma_y)
    sigma_minus = 0.5 * (sigma_x - im * sigma_y)

    raising = [
        local_operator(sigma_plus, site, config.atom_count)
        for site in 1:config.atom_count
    ]
    lowering = [
        local_operator(sigma_minus, site, config.atom_count)
        for site in 1:config.atom_count
    ]

    dimension = 2^config.atom_count
    sum1 = zeros(ComplexF64, dimension, dimension)
    sum2 = zeros(ComplexF64, dimension, dimension)
    sum3 = zeros(ComplexF64, dimension, dimension)

    for site in 1:config.atom_count
        sum1 .+= lowering[site] .+ raising[site]
        sum3 .+= lowering[site]
        for earlier_site in 1:(site - 1)
            sum2 .+= raising[earlier_site] * lowering[site] -
                     raising[site] * lowering[earlier_site]
        end
    end

    beta = config.beta
    gamma_total = config.gamma_total
    input_power = abs2(config.alpha)
    saturation_power = gamma_total / beta
    drive_scale = sqrt(input_power / saturation_power)

    dimension_squared = dimension^2
    liouvillian = zeros(ComplexF64, dimension_squared, dimension_squared)
    liouvillian .+= -im * gamma_total * drive_scale * commutator_map(sum1)
    liouvillian .+= 0.5 * gamma_total * beta * commutator_map(sum2)
    for operator in lowering
        liouvillian .+=
            gamma_total * (1 - beta) * dissipator_map(operator)
    end
    liouvillian .+= gamma_total * beta * dissipator_map(sum3)

    rho_ss = solve_steady_state(liouvillian, dimension)

    identity_d = Matrix{ComplexF64}(I, dimension, dimension)
    gamma = beta * gamma_total
    a_out = config.alpha * identity_d - im * sqrt(gamma) * sum3
    n_out = a_out' * a_out
    third_order_intensity = a_out'^3 * a_out^3

    h_prime =
        0.5 * gamma_total * drive_scale * sum1 +
        0.5im * beta * gamma_total * sum2

    no_jump_generator = -im * commutator_map(h_prime)
    no_jump_generator .-= 0.5 * (
        left_right_map(n_out, identity_d) +
        left_right_map(identity_d, n_out)
    )
    for operator in lowering
        no_jump_generator .+=
            gamma_total * (1 - beta) * dissipator_map(operator)
    end

    no_jump_map_dense =
        Matrix{ComplexF64}(I, dimension_squared, dimension_squared) +
        time_step(config) * no_jump_generator
    jump_map_dense = left_right_map(a_out, a_out')

    # vec(rho) is column-major. These weights reproduce tr(n_out * rho).
    flux_weights = vec(copy(transpose(n_out)))

    return SimulationModel(
        config,
        dimension,
        rho_ss,
        sparse(no_jump_map_dense),
        sparse(jump_map_dense),
        flux_weights,
        third_order_intensity,
        tr(n_out * rho_ss),
    )
end

@inline function vector_trace(state::AbstractVector, dimension::Int)
    value = zero(eltype(state))
    index = 1
    @inbounds for _ in 1:dimension
        value += state[index]
        index += dimension + 1
    end
    return value
end

@inline function photon_flux(model::SimulationModel, state::AbstractVector)
    value = zero(ComplexF64)
    @inbounds for index in eachindex(state, model.flux_weights)
        value += model.flux_weights[index] * state[index]
    end
    abs(imag(value)) <= 1e-10 ||
        error("Photon flux has a significant imaginary part: $value")
    flux = real(value)
    flux >= 0 || error("Negative photon flux: $flux")
    return flux
end

@inline function select_branch(rng::AbstractRNG, probabilities::NTuple{3, Float64})
    draw = rand(rng)
    draw < probabilities[1] && return 1
    draw < probabilities[1] + probabilities[2] && return 2
    return 3
end

function make_trajectory_stats(
    counts::Matrix{Int},
    emitted_photons::Int,
    config::SimulationConfig,
)
    bin_steps = round(Int, config.bin_width / time_step(config))
    number_of_bins = size(counts, 1)
    used_steps = number_of_bins * bin_steps
    discarded_steps = config.steps - used_steps
    effective_time = used_steps * time_step(config)
    actual_bin_width = bin_steps * time_step(config)

    singles = (
        sum(@view counts[:, 1]),
        sum(@view counts[:, 2]),
        sum(@view counts[:, 3]),
    )
    triple_weight = 0
    coincidence_bins = 0
    @inbounds for bin in 1:number_of_bins
        n1 = counts[bin, 1]
        n2 = counts[bin, 2]
        n3 = counts[bin, 3]
        triple_weight += n1 * n2 * n3
        coincidence_bins += n1 > 0 && n2 > 0 && n3 > 0
    end

    registered_G3_density =
        triple_weight / (effective_time * actual_bin_width^2)
    G3_apparent =
        registered_G3_density / prod(config.splitter_probabilities)
    registered_rates = (
        singles[1] / effective_time,
        singles[2] / effective_time,
        singles[3] / effective_time,
    )
    rate_product = prod(registered_rates)
    g3_registered =
        rate_product > 0 ? registered_G3_density / rate_product : NaN

    return TrajectoryStats(
        G3_apparent,
        registered_G3_density,
        g3_registered,
        triple_weight,
        coincidence_bins,
        coincidence_bins / number_of_bins,
        singles,
        registered_rates,
        bin_steps,
        actual_bin_width,
        number_of_bins,
        effective_time,
        discarded_steps,
        discarded_steps * time_step(config),
        emitted_photons,
    )
end

function simulate_trajectory(
    model::SimulationModel,
    trajectory_index::Integer,
    ensemble_size::Integer,
    ::Val{capture_records},
) where {capture_records}
    config = model.config
    dimension = model.dimension
    dt = time_step(config)
    state = vec(copy(model.rho_ss))
    next_state = similar(state)
    jump_rng = Xoshiro(config.seed + trajectory_index)
    splitter_rng = Xoshiro(config.seed + ensemble_size + trajectory_index)

    bin_steps = round(Int, config.bin_width / dt)
    number_of_bins = fld(config.steps, bin_steps)
    used_steps = number_of_bins * bin_steps
    counts = zeros(Int, number_of_bins, 3)
    last_detection = zeros(Int, 3)
    emitted_photons = 0

    emission_record = capture_records ? falses(config.steps) : nothing
    branch_record = capture_records ? zeros(UInt8, config.steps) : nothing

    @inbounds for step in 1:config.steps
        probability = photon_flux(model, state) * dt
        0 <= probability <= 1 ||
            error("Invalid jump probability at step $step: $probability")
        jumped = rand(jump_rng) < probability

        if jumped
            mul!(next_state, model.jump_map, state)
            emitted_photons += 1
            capture_records && (emission_record[step] = true)

            branch = select_branch(splitter_rng, config.splitter_probabilities)
            ready =
                last_detection[branch] == 0 ||
                (step - last_detection[branch]) * dt > config.detector_dead_time
            if ready
                last_detection[branch] = step
                capture_records && (branch_record[step] = UInt8(branch))
                if step <= used_steps
                    bin = fld(step - 1, bin_steps) + 1
                    counts[bin, branch] += 1
                end
            end
        else
            mul!(next_state, model.no_jump_map, state)
        end

        normalization = vector_trace(next_state, dimension)
        @. next_state = next_state / normalization
        state, next_state = next_state, state
    end

    stats = make_trajectory_stats(counts, emitted_photons, config)
    if capture_records
        return (
            final_state = reshape(copy(state), dimension, dimension),
            stats = stats,
            emission_record = emission_record,
            branch_record = branch_record,
        )
    end
    return stats
end

trajectory_statistics(
    model::SimulationModel,
    trajectory_index::Integer;
    ensemble_size::Integer = model.config.trajectories,
) = simulate_trajectory(model, trajectory_index, ensemble_size, Val(false))

trajectory_with_records(
    model::SimulationModel,
    trajectory_index::Integer;
    ensemble_size::Integer = model.config.trajectories,
) = simulate_trajectory(model, trajectory_index, ensemble_size, Val(true))

function run_ensemble(model::SimulationModel)
    trajectory_count = model.config.trajectories
    results = Vector{TrajectoryStats}(undef, trajectory_count)
    Threads.@threads :static for index in 1:trajectory_count
        results[index] = trajectory_statistics(
            model,
            index;
            ensemble_size = trajectory_count,
        )
    end
    return results
end

function ensemble_statistics(
    results::AbstractVector{TrajectoryStats},
    model::SimulationModel,
)
    isempty(results) && throw(ArgumentError("results must not be empty"))

    G3_values = [result.G3_apparent for result in results]
    finite_g3_values = [
        result.g3_registered
        for result in results
        if isfinite(result.g3_registered)
    ]
    trajectory_count = length(results)
    sigma_G3 = trajectory_count > 1 ? std(G3_values) : NaN
    sem_G3 = trajectory_count > 1 ? sigma_G3 / sqrt(trajectory_count) : NaN
    total_triple_weight = sum(result.triple_weight for result in results)
    total_effective_time = sum(result.effective_time for result in results)
    pooled_G3_apparent =
        total_triple_weight /
        (
            total_effective_time *
            results[1].bin_width^2 *
            prod(model.config.splitter_probabilities)
        )

    return (
        pooled_G3_apparent = pooled_G3_apparent,
        mean_G3_apparent = mean(G3_values),
        std_G3_apparent = sigma_G3,
        sem_G3_apparent = sem_G3,
        mean_g3_registered =
            isempty(finite_g3_values) ? NaN : mean(finite_g3_values),
        std_g3_registered =
            length(finite_g3_values) > 1 ? std(finite_g3_values) : NaN,
        per_trajectory_G3 = G3_values,
        per_trajectory_results = results,
        total_effective_time = total_effective_time,
    )
end

function environment_value(name::String, default, converter)
    return haskey(ENV, name) ? converter(ENV[name]) : default
end

function config_from_environment()
    defaults = SimulationConfig()
    return SimulationConfig(
        total_time = environment_value("G3C_TOTAL_TIME", defaults.total_time, x -> parse(Float64, x)),
        steps = environment_value("G3C_STEPS", defaults.steps, x -> parse(Int, x)),
        trajectories = environment_value("G3C_TRAJECTORIES", defaults.trajectories, x -> parse(Int, x)),
        beta = environment_value("G3C_BETA", defaults.beta, x -> parse(Float64, x)),
        gamma_total = environment_value("G3C_GAMMA_TOTAL", defaults.gamma_total, x -> parse(Float64, x)),
        alpha = environment_value("G3C_ALPHA", defaults.alpha, x -> parse(Float64, x)),
        atom_count = environment_value("G3C_ATOM_COUNT", defaults.atom_count, x -> parse(Int, x)),
        detector_dead_time = environment_value("G3C_DEAD_TIME", defaults.detector_dead_time, x -> parse(Float64, x)),
        bin_width = environment_value("G3C_BIN_WIDTH", defaults.bin_width, x -> parse(Float64, x)),
        splitter_probabilities = defaults.splitter_probabilities,
        seed = environment_value("G3C_SEED", defaults.seed, x -> parse(Int, x)),
    )
end

function main()
    config = config_from_environment()
    println(Threads.nthreads())
    model = build_model(config)
    println("Steady state is obtained")

    elapsed = @elapsed results = run_ensemble(model)
    G3_stats = ensemble_statistics(results, model)
    true_G30 = tr(model.third_order_intensity * model.rho_ss)

    println("True value of G^(3):", true_G30)
    println("Pooled apparent G^(3) = ", G3_stats.pooled_G3_apparent)
    println("Mean trajectory G^(3) = ", G3_stats.mean_G3_apparent)
    println("Run-to-run standard deviation = ", G3_stats.std_G3_apparent)
    println("Monte Carlo standard error of the mean = ", G3_stats.sem_G3_apparent)
    println("Mean registered g^(3) = ", G3_stats.mean_g3_registered)
    println("Trajectory simulation time (s) = ", elapsed)
    return nothing
end

end # module G3CSNROptimized

if abspath(PROGRAM_FILE) == @__FILE__
    G3CSNROptimized.main()
end

using LinearAlgebra, Random, Distributions
using DifferentialEquations, SteadyStateDiffEq
using SparseArrays
using Base.Threads
using CUDA
using Plots
using Measures
using DiffEqCallbacks



println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const β = 0.05f0
const Γtot = 1.0f0
const N_max = 11
const filling_factor = 0.1
const base_seed = 124
const γR = β * Γtot
const d = 448  # the average distance between atoms, unit is nm
const λ_0 = 852 # the wavelength of probe laser, unit is nm
const dtoλ_0 = d/λ_0
const α = sqrt(0.1f0)  # Actually it is α/√(L) in the paper
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


function partial_lattice_exact_N(rng::AbstractRNG, N::Integer;
                                filling::Real=0.1,
                                Nsites::Integer=round(Int, N/filling),
                                max_attempts::Integer=100_000)
    0 < filling < 1 || throw(ArgumentError("filling must lie strictly between 0 and 1"))
    1 <= N <= Nsites || throw(ArgumentError("require 1 <= N <= Nsites"))
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))

    for attempt in 1:max_attempts
        occupied = rand(rng, Bernoulli(filling), Nsites)
        if count(occupied) == N
            return findall(occupied), attempt
        end
    end
    error("No pattern with N=$N among $Nsites sites after $max_attempts attempts")
end


function g3c(N::Integer,ϵ;
                filling::Real=filling_factor,
                Nsites::Integer=round(Int, N/filling),
                seed::Integer=base_seed,
                max_attempts::Integer=100_000)

    idN = SparseMatrixCSC{ComplexF32,Int32}(spdiagm(0 => ones(ComplexF32, 2^N)))
    φ = zeros(ComplexF32, N)

    z, attempts = partial_lattice_exact_N(MersenneTwister(seed), N;
        filling=filling, Nsites=Nsites, max_attempts=max_attempts)
#    println("N=$N, Nsites=$Nsites, attempts=$attempts, occupied sites=$z")

    φ = ComplexF32.(4im * pi * dtoλ_0 .* z)
    

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


    ρ0 = 1
    for k in 1:N
        ρ0_atom = [0 0; 0 1]      # each atom is initialized in the ground state
        ρ0 = kron(ρ0_atom, ρ0)
    end
    ρ0 = convert(Matrix{ComplexF32}, ρ0)
    ρ0_v = reshape(transpose(ρ0), 2^(2*N))
    ρ0 = nothing
    ρ0_v = CuArray(ρ0_v)  # move initial state vector to GPU




    # Create time points array
    t_points = [k * Δt for k in 0:(resol-1)]


    ϵ = Float32(ϵ)
    # Initialize directly: adding to an Int64-indexed spzeros would widen the indices.
    L = -1im*sqrt(P_in/P_sat)*com(sum1)

    if N == 1
        sum2 = spzeros(ComplexF32, Int32, 2, 2)
    else
        sum2 = β/2 * sum(σp_full[l]*σm_full[j]-σp_full[j]*σm_full[l] for j in 1:N for l in 1:(j-1))
        sum2 += ϵ/2 * sum(σp_full[l]*σm_full[j]*exp(φ[j]-φ[l])-σp_full[j]*σm_full[l]*exp(-(φ[j]-φ[l])) for l in 1:N for j in 1:(l-1))
    end

    term3 = com(sum2)
    L += term3
    term3 = nothing

    function D(x)
        xdagx = x'*x
        val = kron(x, conj(x))-0.5f0*(kron(xdagx, idN)+kron(idN, transpose(xdagx)))
        return val
    end


    term2 = (1-β-ϵ) * sum(D(σm_full[k]) for k in 1:N)
    L += term2
    term2 = nothing


    term4 = β * D(sum(σm_full))
    L += term4
    term4 = nothing

    term5 = ϵ * D(sum(σm_full[i]*exp(φ[i]) for i in 1:N))
    L += term5
    term5 = nothing

    nonzeros(L) .*= Γtot
    # Avoid full-size CPU conversion buffers in the GPU constructor.
    @assert L isa SparseMatrixCSC{ComplexF32,Int32} "Liouvillian assembly widened its value or index type"
    L = CUSPARSE.CuSparseMatrixCSC{ComplexF32}(L)

    EOM_v!(dρ, ρ, p, t) = mul!(dρ, L, ρ)
    # Steady State Problem (pass L inside a tuple to skip DiffEqBase's isequal on GPU sparse)
    prob_ss = SteadyStateProblem{true}(EOM_v!, ρ0_v)   # Do not pass L to p inside a tuple to avoid DiffEqBase's isequal on GPU sparse
    sol_ss = solve(prob_ss, DynamicSS(Tsit5());
    save_everystep=false,
    save_start=true,
    dense=false,
    abstol=1e-8,
    reltol=1e-6)

#    println("Steady state is obtained")

    ρ_ss = transpose(reshape(Array(sol_ss.u), 2^(N), 2^(N)))
    sol_ss = nothing

    a_out = α*idN - 1im*sqrt(γR)*sum(σm_full)

    out_power = tr(a_out'*a_out*ρ_ss)

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

        saved_values = SavedValues(Float64, Nothing)

        cb = SavingCallback(
            (u, t, integrator) -> begin
                put!(c, Array(u))
                nothing
            end,
            saved_values;
            saveat=0.0:Δt:time_span,
            save_everystep=false,
            save_start=true,
        )

        prob = ODEProblem{true}(EOM_v!, ρ_new, (0.0, round((t_f-t_i)*100)/100)) # in-place form is true
        sol = solve(prob, Tsit5();
            callback=cb,
            save_everystep=false,
            save_start=false,
            save_end=false,
            dense=false,
            abstol=1e-9,
            reltol=1e-7
        )
    end

    ρ_Step1 = Vector{Array{ComplexF32,2}}(undef, resol)

    G2 = Vector{Float32}(undef, resol)


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

    g3c_val = zeros(ComplexF32, resol, resol)
    Threads.@threads for i in 1:resol
        for j in 1:(resol-i+1)
            g3c_val[i, i+j-1] = 2 + G3[i, i+j-1]/out_power^3 - (G2[i]+G2[j]+G2[i+j-1])/out_power^2
        end
    end
    g3c_val = real(g3c_val + transpose(g3c_val) - diagm(diag(g3c_val)))
    return g3c_val
end

error_line = zeros(Float32, N_max)
for n in 1:N_max
    g3c_0 = g3c(n,0.0)

    g3c_005 = g3c(n,0.1*β)

    rel_err = norm(g3c_005 - g3c_0) / norm(g3c_0)
    println(rel_err)
    error_line[n] = rel_err
end

plot(error_line, xlabel="N", ylabel="Relative Error", title="Relative Error of g3c with ϵ=0.005 vs ϵ=0.0", size=(800, 600), dpi=300)
#g3c_h = g3c(h)

#g3c_2h = g3c(2*h)


#g3c_derv = (-3*g3c_0 + 4*g3c_h - g3c_2h)/(2*h)

#heatmap(g3c_derv, xlabel="t1", ylabel="t2", title="g3c derivative with respect to ϵ", colorbar_title="g3c_derv", size=(800, 600), dpi=300)

using LinearAlgebra
using DifferentialEquations
using SteadyStateDiffEq
using SparseArrays
using Base.Threads
using RecursiveArrayTools: ArrayPartition


# global variables always change, which slows the code
println(Threads.nthreads()) # check the number of threads

const tot_t = 5.0               # total time, data type should be float.
const β = 0.08
const Γtot = 1.0
const γ = β*Γtot
const Γ = (1-β)*Γtot                      # make sure that sqrt(β) << 1
const k_0 = 0.0

const α = sqrt(0.8)  # Actually it is α/√(L) in the paper
const N = 5  
const P_in  = abs(α)^2
const P_sat = Γtot/β
const resol = 501               # ODE solver saves the values at 501 time points including the initial time
const Δt = tot_t/(resol-1)
const a = k_0 + 1im*Γ*(1-2*β)/(2*β)


# operators
const σx = [0 1; 1 0]
const σy = [0 -1im; 1im 0]
const σz = [1 0; 0 -1]
const σp = 1/2*(σx+1im*σy)
const σm = 1/2*(σx-1im*σy)
const θ  = 0.0


global σp_full =
    Vector{Matrix{ComplexF64}}(undef, N)

global σm_full =
    Vector{Matrix{ComplexF64}}(undef, N)


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

σp_full = [
    sparse(Matrix{ComplexF64}(A))
    for A in σp_full
]

σm_full = [
    sparse(Matrix{ComplexF64}(A))
    for A in σm_full
]

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

const EOM_P = (
    Γtot = Γtot,
    β = β,
    sum1 = sum1,
    sum2 = sum2,
    sum3 = sum3,
    P_in = P_in,
    P_sat = P_sat,
    σm_full = σm_full,
)

function EOM!(dρ, ρ, p, t)
    drive_part =
        -1im *
        sqrt(p.P_in / p.P_sat) *
        com(p.sum1, ρ)

    cascade_part =
        (p.β / 2) *
        com(p.sum2, ρ)

    guided_decay =
        p.β *
        D(p.sum3, ρ)

    dρ .=
        p.Γtot .* (
            drive_part +
            cascade_part +
            guided_decay
        )

    individual_decay_coefficient =
        p.Γtot * (1 - p.β)

    for sm in p.σm_full
        dρ .+=
            individual_decay_coefficient .* D(sm, ρ)
    end

    return nothing
end



# Steady State Problem
prob_ss =
    SteadyStateProblem{true}(
        EOM!,
        ρ0,
        EOM_P,
    )

sol_ss =
    solve(
        prob_ss,
        DynamicSS(Tsit5());
        abstol = 1e-10,
        reltol = 1e-8,
    )

println("Steady state is obtained")

ρss = Matrix{ComplexF64}(sol_ss.u)

# Remove roundoff-level non-Hermiticity.
ρss =
    0.5 .* (ρss + ρss')

ρss ./=
    real(tr(ρss))

steady_state_residual =
    similar(ρss)

EOM!(
    steady_state_residual,
    ρss,
    EOM_P,
    0.0,
)

println(
    "Steady-state residual norm = ",
    norm(steady_state_residual),
)
 
a_out =
    α * I -
    1im * sqrt(γ) * sum3

a_out_dag =
    copy(adjoint(a_out))

# Dense form is useful for the trace inner products below.
n_out =
    Matrix{ComplexF64}(
        a_out_dag * a_out
    )

out_power =
    real(dot(n_out, ρss))

println(
    "Steady-state output power = ",
    out_power,
)

"""
Augmented quantum-regression equation for

    ∫₀ᵀ dt₁ ∫₀ᵀ dt₂ G⁽³⁾(0,t₁,t₂).

State components:

    X(t) = exp(L t) J[ρss]

    Y(t) = ∫₀ᵗ ds exp[L(t-s)] J[X(s)]

    S(t) = 2 ∫₀ᵗ du Tr[J Y(u)]

where J[ρ] = a_out * ρ * a_out†.
"""
function G3_square_integral_rhs!(du, u, p, t)
    X = u.x[1]
    Y = u.x[2]

    dX = du.x[1]
    dY = du.x[2]
    dS = du.x[3]

    # dX/dt = L X
    EOM!(
        dX,
        X,
        p.eom,
        t,
    )

    # Begin with dY/dt = L Y.
    EOM!(
        dY,
        Y,
        p.eom,
        t,
    )

    # Add J[X] = a_out * X * a_out†.
    #
    # The two scratch matrices prevent allocation of a new
    # d×d matrix on every RHS evaluation.
    mul!(
        p.tmp_left,
        p.a_out,
        X,
    )

    mul!(
        p.tmp_jump,
        p.tmp_left,
        p.a_out_dag,
    )

    dY .+= p.tmp_jump

    # Tr[J Y] = Tr[a_out† a_out Y].
    #
    # Because n_out is Hermitian,
    # dot(n_out,Y) = Tr(n_out * Y).
    dS[1] =
        2 * dot(p.n_out, Y)

    return nothing
end

"""
Compute

    I_G3(T) = ∫₀ᵀ dt₁ ∫₀ᵀ dt₂ G⁽³⁾(0,t₁,t₂)

using one augmented ODE solve.

Also returns the square-averaged value

    G3_average = I_G3(T) / T².
"""
function integrate_G3_square(
    ρss,
    a_out,
    n_out,
    eom_parameters,
    T;
    alg = Vern8(),
    abstol = 1e-11,
    reltol = 1e-9,
)
    T > 0 ||
        throw(
            ArgumentError(
                "The upper integration time T must be positive."
            )
        )

    A =
        copy(a_out)

    Adag =
        copy(adjoint(A))

    # First detection at time zero:
    #
    # X(0) = J[ρss].
    X0 =
        A * ρss * Adag

    Y0 =
        zeros(
            ComplexF64,
            size(ρss),
        )

    S0 =
        ComplexF64[0.0]

    augmented_initial_state =
        ArrayPartition(
            X0,
            Y0,
            S0,
        )

    parameters = (
        eom = eom_parameters,
        a_out = A,
        a_out_dag = Adag,
        n_out = n_out,
        tmp_left = similar(ρss),
        tmp_jump = similar(ρss),
    )

    problem =
        ODEProblem(
            G3_square_integral_rhs!,
            augmented_initial_state,
            (0.0, Float64(T)),
            parameters,
        )

    solution =
        solve(
            problem,
            alg;
            abstol = abstol,
            reltol = reltol,
            save_start = false,
            save_end = true,
            save_everystep = false,
            dense = false,
            maxiters = 10^7,
        )

    integral_complex =
        solution.u[end].x[3][1]

    imaginary_residual =
        abs(imag(integral_complex))

    imaginary_tolerance =
        100 * abstol +
        100 * reltol *
        max(
            1.0,
            abs(real(integral_complex)),
        )

    imaginary_residual <= imaginary_tolerance ||
        @warn(
            "The G^(3) integral has a non-negligible imaginary part",
            integral_complex,
            imaginary_residual,
        )

    integral_G3 =
        real(integral_complex)

    average_G3 =
        integral_G3 / T^2

    # Equal-time point value, useful as a small-T check.
    A3 =
        A * A * A

    point_G30 =
        real(
            tr(
                A3' *
                A3 *
                ρss
            )
        )

    output_flux =
        real(
            dot(
                n_out,
                ρss,
            )
        )

    normalized_integral =
        output_flux > 0 ?
        integral_G3 / output_flux^3 :
        NaN

    normalized_average =
        output_flux > 0 ?
        average_G3 / output_flux^3 :
        NaN

    return (
        integral_G3 = integral_G3,
        average_G3 = average_G3,
        point_G30 = point_G30,
        output_flux = output_flux,
        normalized_integral = normalized_integral,
        normalized_average = normalized_average,
        imaginary_residual = imaginary_residual,
        solution = solution,
    )
end


const T_G3 =
    1.5 / Γtot

G3_result =
    integrate_G3_square(
        ρss,
        a_out,
        n_out,
        EOM_P,
        T_G3;
        alg = Vern8(),
        abstol = 1e-11,
        reltol = 1e-9,
    )

println(
    "G^(3)(0,0,0) = ",
    G3_result.point_G30,
)

println(
    "∫₀ᵀ dt₁ ∫₀ᵀ dt₂ G^(3)(0,t₁,t₂) = ",
    G3_result.integral_G3,
)

println(
    "Square-averaged G^(3) = ",
    G3_result.average_G3,
)

println(
    "∫₀ᵀ dt₁ ∫₀ᵀ dt₂ g^(3)(0,t₁,t₂) = ",
    G3_result.normalized_integral,
)

println(
    "Square-averaged g^(3) = ",
    G3_result.normalized_average,
)








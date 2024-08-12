#=
Author: Cooper Simpson

SFN step solvers.
=#

using Krylov: CraigmrSolver, craigmr!
using Arpack: eigs

########################################################

#=
Find search direction using low-rank eigendecomposition with Arpack
=#
mutable struct ArpackSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    rank::I #
    p::S #search direction
end

function ArpackSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    # rank = Int(ceil(log(dim)))
    rank = Int(ceil(sqrt(dim)))

    return ArpackSolver(rank, type(undef, dim))
end

function step!(solver::ArpackSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0

    D, V = eigs(Hv, nev=solver.rank, which=:LM, ritzvec=true)

    @. D = pinv(sqrt(D^2+λ))
    # mul!(solver.p, V', b)
    # mul!(solver.p, Diagonal(D), solver.p)
    # mul!(solver.p, V, solver.p)
    mul!(solver.p, V*Diagonal(D)*V', b) #not the fastest way to do this

    return
end

########################################################

#=
Find search direction using CRAIGMR
=#
mutable struct CraigSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CraigmrSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function CraigSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if length(nodes) < quad_order
        quad_order = length(nodes)
        println("Quadrature weight precision reached, using $(quad_order) quadrature locations.")
    end

    #=
    NOTE: Performing some extra global operations here.
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #krylov solver
    solver = CraigmrSolver(dim, dim, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return CraigSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::CraigSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0
    
    shifts = sqrt.(solver.quad_nodes .+ λ)

    solved = true

    @inbounds for i in eachindex(shifts)
        craigmr!(solver.krylov_solver, Hv, b, λ=shifts[i], itmax=solver.krylov_order, timemax=time_limit)

        solved = solved && solver.krylov_solver.stats.solved

        solver.p .+= solver.quad_weights[i]*solver.krylov_solver.y
    end

    # if solved == false
    #     println("WARNING: Solver failure")
    # end

    return
end

########################################################

#=
Find search direction using randomized indefinite Nystrom
=#
mutable struct NystromIndefiniteSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    r::I #rank
    s::I #sketch size
    p::S #search direction
end

function NystromIndefiniteSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, r::I=Int(ceil(sqrt(dim))), c::T=1.5) where {I<:Integer, T<:AbstractFloat}
    # return NystromIndefiniteSolver(r, Int(ceil(c*r)), type(undef, dim))
    return NystromIndefiniteSolver(20, 50, type(undef, dim))
end

function step!(solver::NystromIndefiniteSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    Ω = randn(size(Hv,1), solver.s) #Guassian test matrix

    C = Hv*Ω #sketch
    W = Ω'*C #sketch
    Λ, V = eigen(Hermitian(W), sortby=(-)∘(abs)) #eigendecomposition
    Λ, V = Λ[1:solver.r], V[:,1:solver.r]

    Wr = V*Diagonal(pinv.(Λ))*V'

    Q, R = qr(C)
    Σ, U = eigen(Hermitian(R*Wr*R'), sortby=(-)∘(abs))
    E = Eigen(Σ[1:solver.r], Q*U[:,1:solver.r])
    @. E.values = sqrt(E.values^2+λ)

    mul!(solver.p, pinv(E), b)

    # println(pinv.(E.values[1:4]))

    return 
end

########################################################

#=
Find search direction using randomized SVD
=#
mutable struct RSVDSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    k::I #rank
    r::I #sketch size
    p::S #search direction
end

function RSVDSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, k::I=Int(ceil(sqrt(dim))), r::I=Int(ceil(1.5*k))) where {I<:Integer, T<:AbstractFloat}
    return RSVDSolver(k, r, type(undef, dim))
end

function step!(solver::RSVDSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    # Ω = srft(solver.k)
    Ω = randn(size(Hv,1), solver.k)
    
    Y=Hv*Ω; Q = Matrix(qr!(Y).Q)
    B=(Q'Y)\(Q'Ω)
    E=eigen!(B)
    E=Eigen(E.values, Q*real.(E.vectors))

    @. E.values = sqrt(real(E.values)^2+λ)
    mul!(solver.p, pinv(E), b)

    return
end
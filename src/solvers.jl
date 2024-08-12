#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre, gausschebyshevt
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!, CgLanczosSolver, cg_lanczos!, CgLanczosShaleSolver, cg_lanczos_shale!
using KrylovKit: eigsolve, Lanczos, KrylovDefaults
using IterativeSolvers: lobpcg
# using RandomizedPreconditioners: NystromPreconditioner

########################################################

#=
Shifted CG Lanczos with Gauss-Laguerre quadrature.
=#
mutable struct GLKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #Krylov solver
    const krylov_order::I #maximum Krylov subspace size
    const quad_nodes::S #quadrature nodes
    const quad_weights::S #quadrature weights
    p::S #search direction
end

function hvp_power(solver::GLKSolver)
    return 2
end

function GLKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if length(nodes) < quad_order
        quad_order = length(nodes)
        println("Quadrature weight precision reached, using $(quad_order) quadrature locations.")
    end

    #=
    Global operations
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2.0/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #Krylov solver
    solver = CgLanczosShiftSolver(dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return GLKSolver(solver, krylov_order, T.(nodes), T.(weights), type(undef, dim))
end

function step!(solver::GLKSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::Float64=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}
    
    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    # println("λ: ", λ)

    #Reset search direction
    solver.p .= 0.0

    #Quadrature scaling factor
    # β = eigmax(Hv, tol=1e-6)
    β = eigmean(Hv)
    
    # println("β: ", β)

    #Preconditioning
    P = I

    # E = eigen(Matrix(Hv))
    # @. E.values = pinv(E.values)
    # P = Matrix(E)

    # k = Int(ceil(log(size(Hv, 1))))
    # r = Int(ceil(1.5*k))
    # P = NystromPreconditionerInverse(NystromSketch(Hv, k, r), 0)

    #Shifts
    shifts = β*solver.quad_nodes .+ λ
    
    #Tolerance
    # cg_atol = sqrt(eps(T))
    # cg_rtol = sqrt(eps(T))

    ζ = 0.5
    ξ = T(0.01)

    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

    #CG solves
    cg_lanczos_shift!(solver.krylov_solver, Hv, -g, shifts, M=P, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    converged = sum(solver.krylov_solver.converged)
    if converged != length(shifts)
        println("WARNING: Solver failed, only ", converged, " converged")
    end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    #Update search direction
    for i in eachindex(shifts)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    solver.p .*= sqrt(β)

    return
end

########################################################

#=
Shifted and scaled CG Lanczos with Gauss-Chebyshev quadrature.
=#
mutable struct GCKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShaleSolver #Krylov solver
    const krylov_order::I #maximum Krylov subspace size
    const quad_nodes::S #quadrature nodes
    const quad_weights::S #quadrature weights
    p::S #search direction
end

function hvp_power(solver::GCKSolver)
    return 2
end

function GCKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Quadrature
    nodes, weights = gausschebyshevt(quad_order)
    @. weights *= 2.0/pi #global scaling

    #Krylov solver
    solver = CgLanczosShaleSolver(dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return GCKSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::GCKSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    # println("λ: ", λ)

    #Reset search direction
    solver.p .= 0.0

    #Quadrature scaling factor
    # β = eigmax(Hv, tol=1e-6)
    β = eigmean(Hv)

    # println("β: ", β)

    #Preconditioning
    P = I

    # E = eigen(Matrix(Hv))
    # @. E.values = pinv(E.values)
    # P = Matrix(E)

    # k = Int(ceil(log(size(Hv, 1))))
    # r = Int(ceil(1.5*k))
    # P = NystromPreconditionerInverse(NystromSketch(Hv, k, r), 0)
    
    #Shifts and scalings
    shifts = (λ-β) .* solver.quad_nodes .+ (λ+β)
    scales = solver.quad_nodes .+ 1.0

    #Tolerance
    # cg_atol = sqrt(eps(T))
    # cg_rtol = sqrt(eps(T))

    ζ = 0.5
    ξ = T(0.01)

    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

    #CG Solves
    cg_lanczos_shale!(solver.krylov_solver, Hv, -g, shifts, scales, M=P, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    converged = sum(solver.krylov_solver.converged)
    if converged != length(shifts)
        println("WARNING: Solver failed, only ", converged, " converged")
    end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    #Update search direction
    for i in eachindex(shifts)
        @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
    end

    solver.p .*= sqrt(β)

    return
end

########################################################

#=
Krylov based low-rank approximation.
=#
mutable struct KrylovSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    const rank::I #target rank
    krylov_solver::Lanczos #Krylov solver
    # krylovdim::I #maximum Krylov subspace size
    # maxiter::I #maximum restarts
    p::S #search direction
end

function hvp_power(solver::KrylovSolver)
    return 1
end

function KrylovSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = Int(ceil(sqrt(dim)))
    else
        k = Int(ceil(log(dim)))
    end

    r = Int(ceil(1.5*k))

    krylov_solver = Lanczos(krylovdim=r, maxiter=1, tol=1e-1, orth=KrylovDefaults.orth, eager=false, verbosity=0)
    return KrylovSolver(k, krylov_solver, type(undef, dim))

    # return KrylovSolver(rank, k, 1, type(undef, dim))
end

function step!(solver::KrylovSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Low-rank eigendecomposition
    D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, solver.krylov_solver)
    # D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, krylovdim=solver.krylovdim, maxiter=solver.maxiter, tol=1e-1)

    push!(stats.krylov_iterations, info.numops)
    
    #Select, collect, and update eigenstuff
    # idx = min(info.converged, solver.rank)

    # D = D[1:info.converged]
    # V = V[1:info.converged]

    V = stack(V)

    #Temporary memory
    #NOTE: This could be part of the solver struct
    cache = S(undef, length(D))

    mul!(cache, V', -g)
    @. cache *= sqrt(D[end]+λ)*inv(sqrt(D+λ)) - one(T)
    mul!(solver.p, V, cache)
    @. solver.p += -g 

    return
end

########################################################

#=
Full eigendecomposition.
=#
mutable struct EigenSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function hvp_power(solver::EigenSolver)
    return 1
end

function EigenSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return EigenSolver(type(undef, dim))
end

function step!(solver::EigenSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Eigendecomposition
    E = eigen(Matrix(Hv))

    #Temporary memory
    cache = similar(g)

    #Update search direction
    mul!(cache, E.vectors', -g)
    @. cache *= pinv(sqrt(E.values^2+λ))
    mul!(solver.p, E.vectors, cache)

    return
end

########################################################

#=
Randomized Nystrom low-rank eigendecomposition.

NOTE: This uses the positive-definite Nystrom approximation which implicitly uses H^2
=#
mutable struct NystromSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    const k::I #rank
    const r::I #sketch size
    p::S #search direction
end

function hvp_power(solver::NystromSolver)
    return 2
end

function NystromSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = Int(ceil(sqrt(dim)))
    else
        k = Int(ceil(log(dim)))
    end

    r = Int(ceil(1.5*k))
    
    return NystromSolver(k, r, type(undef, dim))
end

#NOTE: https://github.com/tjdiamandis/RandomizedPreconditioners.jl/blob/main/src/sketch.jl
function NystromSketch(A::H, k::Int, r::Int) where {H<:HvpOperator}
    n = size(A, 1)
    Y = zeros(n, r)

    Ω = 1/sqrt(n) * randn(n, r)
    mul!(Y, A, Ω)
    
    ν = sqrt(n)*eps(norm(Y))
    @. Y = Y + ν*Ω

    Z = zeros(r, r)
    mul!(Z, Ω', Y)
    
    B = Y / cholesky(Hermitian(Z), check=false).U
    U, Σ, _ = svd(B)
    D = max.(0, Σ.^2 .- ν)

    return U[:,1:k], D[1:k]
end

function step!(solver::NystromSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Nystrom approximation
    U, D = NystromSketch(Hv, solver.k, solver.r)

    #Temporary memory
    #NOTE: This could be part of the solver struct
    cache = S(undef, solver.k)

    mul!(cache, U', -g)
    @. cache *= sqrt(D[end]+λ)*inv(sqrt(D+λ)) - one(T)
    mul!(solver.p, U, cache)
    @. solver.p += -g

    return
end

########################################################

#=
Direct inverse square-root solver.
=#
mutable struct DirectSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function hvp_power(solver::DirectSolver)
    return 2
end

function DirectSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return DirectSolver(type(undef, dim))
end

function step!(solver::DirectSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Update search direction
    solver.p .= -sqrt(Matrix(Hv)+λ*I)\g

    return
end

########################################################

#=
Locally-Optimal Block Preconditioned Conjugate Gradient (LOBPCG) based low-rank approximation
=#
mutable struct LOBPCGSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    const rank::I
    const maxiter::I
    p::S #search direction
end

function hvp_power(solver::LOBPCGSolver)
    return 2
end

function LOBPCGSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = Int(ceil(sqrt(dim)))
    else
        k = Int(ceil(log(dim)))
    end

    r = Int(ceil(1.5*k))

    return LOBPCGSolver(rank, r, type(undef, dim))
end

function step!(solver::LOBPCGSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Low-rank eigendecomposition
    r = lobpcg(Hv, true, solver.rank, maxiter=solver.maxiter)

    #Temporary memory
    #NOTE: This could be part of the solver struct
    cache = S(undef, length(D))

    mul!(cache, r.X', -g)
    @. cache *= sqrt(r.λ[end]+λ)*inv(sqrt(r.λ+λ)) - one(T)
    mul!(solver.p, r.X, cache)
    @. solver.p += -g 

    return
end

########################################################

#=
Find search direction using shifted CG Lanczos
=#
mutable struct RNSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #krylov inverse mat vec solver
    const krylov_order::I #maximum Krylov subspace size
    p::S #search direction
end

function hvp_power(solver::RNSolver)
    return 1
end

function RNSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #krylov solver
    solver = CgLanczosShiftSolver(dim, dim, 1, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return RNSolver(solver, krylov_order, type(undef, dim))
end

function step!(solver::RNSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::Float64=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, sqrt(M*g_norm)), 1e-15)

    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))
    
    cg_lanczos_shift!(solver.krylov_solver, Hv, -g, [λ], itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    if sum(solver.krylov_solver.converged) != length(shifts)
        println("WARNING: Solver failure")
    end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    solver.p .= solver.krylov_solver.x[1]

    return
end

########################################################

#=
Adaptive Regularization with Cubics (ARC) solver using shifted CG Lanczos
=#
mutable struct ARCSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CgLanczosShiftSolver #Krylov solver
    const krylov_order::I #maximum Krylov subspace size
    const shifts::S #shifts
    p::S #search direction
end

function hvp_power(solver::ARCSolver)
    return 1
end

function ARCSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, num_shifts::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Shifts
    #TODO: Make this variable to num_shifts
    shifts = 10.0 .^ (collect(-10.0:0.5:20.0))

    #Krylov solver
    solver = CgLanczosShiftSolver(dim, dim, num_shifts, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return ARCSolver(solver, krylov_order, shifts, type(undef, dim))
end

function step!(solver::ARCSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::Float64=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}
    
    #Tolerance
    ζ = 0.5
    ξ = T(0.01)

    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

    #Solver callback, exits when at least one solution that will work has been found
    cb = (slv) -> begin
        for i = eachindex(solver.shifts)
            if !slv.not_cv[i] && (norm(slv.x[i]) / solver.shifts[i] - M > 0)
                return true
            end
        end
        return false
    end

    #Solve subproblem
    cg_lanczos_shift!(solver.krylov_solver, Hv, -g, solver.shifts, itmax=solver.krylov_order, timemax=time_limit, check_curvature=true, atol=cg_atol, rtol=cg_rtol, callback=cb)

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    return
end
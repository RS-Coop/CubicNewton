#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre, gausschebyshevt
using Krylov: CgLanczosShiftSolver, cg_lanczos_shift!, CgLanczosSolver, cg_lanczos!, CgLanczosShaleSolver, cg_lanczos_shale!, hermitian_lanczos

########################################################

#=
Lanczos tri-diagonal function approximation
=#
mutable struct LanczosFA{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    rank::I #target rank
    const max_rank::I #maximum rank
    p::S #search direction
end

function hvp_power(solver::LanczosFA)
    return 1
end

function LanczosFA(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = ceil(sqrt(dim))
    else
        k = ceil(log(dim))
    end

    r = Int(ceil(1.5*k))

    return LanczosFA(min(dim, r), min(dim, 100), type(undef, dim))
end

function step!(solver::LanczosFA, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Hermitian Lanczos: Unitary tridiagonalization
    Q, _, B = hermitian_lanczos(Hv, g, solver.rank)

    push!(stats.krylov_iterations, solver.rank) #NOTE: I think, could be OB1

    Q = Q[:,1:solver.rank] #NOTE: do a view instead?
    
    #NOTE: This whole process isn't ideal
    # do a view instead
    # ideally the output of hermitian_lanczos would already be Julia tridiagonal and not sparsecsc
    # ideally the output wouldn't have any Nans, or you could check for this in the conversion, or in Krylov
    B = Matrix(B[1:solver.rank,:])
    # B[isnan.(B)] .= 0.0 #Should we even be doing this
    B = SymTridiagonal(B)
    
    #Tridiagonal eigendecomposition
    E = eigen(B, sortby=x->abs(x))

    r = findfirst(x->abs(x)≥λ, E.values) #check eigenvalues

    #Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = S(undef, solver.rank)
    cache2 = S(undef, solver.rank)

    #Update search direction
    # @. cache1 = pinv(sqrt(E.values^2+λ))*E.vectors[1,:]
    # mul!(cache2, E.vectors, cache1)
    # mul!(solver.p, Q, cache2)

    # solver.p *= -g_norm

    #Update search direction, NOTE: Is this correct
    @. cache1 = (pinv(sqrt(E.values^2+λ)) - pinv(sqrt(λ)))*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)
    mul!(solver.p, Q, cache2)

    solver.p *= -g_norm
    solver.p .-= pinv(sqrt(λ))*g

    #Update rank
    if isnothing(r) #All eigenvalues are strictly less than regularization, seems unlikely
        #
    elseif r == 1 #All eigenvalues greater than or equal to reg.
        solver.rank = min(solver.max_rank, 2*solver.rank) #increase rank
    else #At least one eigenvalue strictly less than reg
        solver.rank -= r-2 #decrease rank
    end

    return
end

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

    #Reset search direction
    solver.p .= 0.0

    #Quadrature scaling factor
    # β = eigmax(Hv, tol=1e-6)
    β = eigmean(Hv)

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
    cg_atol = sqrt(eps(T))
    cg_rtol = sqrt(eps(T))

    # ζ = 0.5
    # ξ = T(0.01)

    # cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    # cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

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

function GCKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=10, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

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

    #Reset search direction
    solver.p .= 0.0

    #Quadrature scaling factor
    # β = eigmax(Hv, tol=1e-6)
    β = eigmean(Hv)+λ 

    #Preconditioning
    P = I

    # E = eigen(Matrix(Hv), sortby=x->-1/abs(x))
    # β = mean(E.values[1:end-2])
    # @. E.values = pinv(sqrt(E.values))
    # P = Matrix(E)

    # k = Int(ceil(log(size(Hv, 1))))
    # r = Int(ceil(1.5*k))
    # P = NystromPreconditionerInverse(NystromSketch(Hv, k, r), 0)
    
    #Shifts and scalings
    shifts = (λ-β) .* solver.quad_nodes .+ (λ+β)
    scales = solver.quad_nodes .+ 1.0

    #Tolerance
    cg_atol = sqrt(eps(T))
    cg_rtol = sqrt(eps(T))

    # ζ = 0.5
    # ξ = T(0.01)

    # cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    # cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

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

########################################################

#=

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

    if sum(solver.krylov_solver.converged) != 1
        println("WARNING: Solver failure")
    end

    push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

    solver.p .= solver.krylov_solver.x[1]

    return
end
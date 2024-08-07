#=
Author: Cooper Simpson

Newton-type optimizers.
=#

abstract type Optimizer end

########################################################

#=
SFN optimizer struct.
=#
mutable struct SFNOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    M::T1 #hessian lipschitz constant
    solver::S #search direction solver
    η::T2 #step-size
    const ϵ::T2 #regularization minimum
    const linesearch::Bool #whether to use linsearch
    const α::T2 #linesearch factor
    const atol::T2 #absolute gradient norm tolerance
    const rtol::T2 #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    ϵ :: regularization minimum
    linsearch :: whether to use linesearch
    η :: step-size in (0,1)
    α :: linsearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function SFNOptimizer(dim::I, solver::Symbol=:KrylovSolver; M::T1=1.0, ϵ::T2=eps(Float64), linesearch::Bool=false, η::T2=1.0, α::T2=0.5, atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}
    
    #Regularization
    @assert 0≤M && 0≤ϵ

    if linesearch
        @assert 0<α && α<1
        @assert 0<η && η≤1
    end

    solver = eval(solver)(dim)

    return SFNOptimizer(M, solver, η, ϵ, linesearch, α, atol, rtol)
end

########################################################

#=
ARC optimizer struct.
=#
mutable struct ARCOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    M::T1 #
    solver::S #search direction solver
    const linesearch::Bool
    const η1::T2 #
    const η2::T2 #
    const γ1::T2 #
    const γ2::T2 #
    const atol::T2 #absolute gradient norm tolerance
    const rtol::T2 #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    ϵ :: regularization minimum
    linsearch :: whether to use linesearch
    η :: step-size in (0,1)
    α :: linsearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function ARCOptimizer(dim::I; M::T1=10.0, η1::T2=0.1, η2::T2=0.75, γ1::T2=0.1, γ2::T2=5.0, atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}

    #
    @assert 0<M
    @assert 0<η1 && η1<η2 && η2<1
    @assert 0<γ1 && γ1<1 && 1<γ2

    solver = ARCSolver(dim)

    return ARCOptimizer(M, solver, true, η1, η2, γ1, γ2, atol, rtol)
end

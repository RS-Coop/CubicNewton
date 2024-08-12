#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module QuasiNewton

#=
Setup
=#
using LinearAlgebra

export SFNOptimizer, minimize!, update!

include("stats.jl")
include("hvp.jl")
include("solvers.jl")
include("optimizers.jl")
include("minimize.jl")
include("linesearch.jl")

#=
High-level interfaces
=#

#R-SFN
function rsfn!(x::S, f::F; mode::Symbol, itmax::I, time_limit::T2, M::T1=NaN, atol::T2=1e-5, rtol::T2=1e-6, linesearch::Bool=false) where {T1<:Real, T2<:AbstractFloat, S<:AbstractVector{T2}, F, I}
	opt = SFNOptimizer(size(x,1), mode, M=M, linesearch=linesearch, atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f, itmax=itmax, time_limit=time_limit)

	return stats
end

function rsfn!(x::S, f::F1, fg!::F2, H::L; mode::Symbol, itmax::I, time_limit::T2, M::T1=NaN, atol::T2=1e-5, rtol::T2=1e-6, linesearch::Bool=false) where {T1<:Real, T2<:AbstractFloat, S<:AbstractVector{T2}, F1, F2, L, I}
	opt = SFNOptimizer(size(x,1), mode, M=M, linesearch=linesearch, atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f, fg!, H, itmax=itmax, time_limit=time_limit)

	return stats
end

#ARC
function arc!(x::S, f::F; itmax::I, time_limit::T, atol::T=1e-5, rtol::T=1e-6) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I}
	opt = ARCOptimizer(size(x,1), atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f, itmax=itmax, time_limit=time_limit)

	return stats
end

function arc!(x::S, f::F1, fg!::F2, H::L; itmax::I, time_limit::T, atol::T=1e-5, rtol::T=1e-6) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
	opt = ARCOptimizer(size(x,1), atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f, fg!, H, itmax=itmax, time_limit=time_limit)

	return stats
end

#=
If optional packages are loaded then export compatible functions.
=#
function __init__()
    
end

end #module

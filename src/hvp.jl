#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

import Base.*

abstract type HvpOperator{T} <: AbstractMatrix{T} end

include("hvp/EnzymeHvpExt.jl")
include("hvp/LinopHvpExt.jl")
include("hvp/RdiffHvpExt.jl")
include("hvp/ZygoteHvpExt.jl")

#=
Base and LinearAlgebra implementations for HvpOperator
=#
Base.eltype(Hv::HvpOperator{T}) where {T} = T
Base.size(Hv::HvpOperator) = (length(Hv.x), length(Hv.x))
Base.size(Hv::HvpOperator, d::Integer) = d ≤ 2 ? length(Hv.x) : 1
Base.adjoint(Hv::HvpOperator) = Hv
LinearAlgebra.ishermitian(Hv::HvpOperator) = true
LinearAlgebra.issymmetric(Hv::HvpOperator) = true

#=
In place update of HvpOperator
Input:
=#
function reset!(Hv::HvpOperator)
	Hv.nprod = 0

	return nothing
end

#=
Form full matrix
=#
function Base.Matrix(Hv::HvpOperator{T}) where {T}
	n = size(Hv, 1)
	H = Matrix{T}(undef, n, n)

	ei = zeros(T, n)

	@inbounds for i = 1:n
		ei[i] = one(T)
		mul!(@view(H[:,i]), Hv, ei)
		ei[i] = zero(T)
	end

	return Hermitian(H)
end

#=
Out of place matrix vector multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	Hv :: HvpOperator
	v :: rhs vector
=#
function *(Hv::H, v::S) where {S<:AbstractVector{<:AbstractFloat}, H<:HvpOperator}
	res = similar(v)
	mul!(res, Hv, v)
	return res
end

#=
Out of place matrix matrix multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	Hv :: HvpOperator
	v :: rhs vector
=#
function *(Hv::H, V::M) where {M<:Matrix{<:AbstractFloat}, H<:HvpOperator}
	res = similar(V)
	mul!(res, Hv, V)
	return res
end

#=
In-place matrix vector multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::AbstractVector, Hv::H, v::S) where {S<:AbstractVector{<:AbstractFloat}, H<:HvpOperator}
	apply!(result, Hv, v)

	@inbounds for i=1:Hv.power-1
		apply!(result, Hv, result) #NOTE: Is this okay reusing result like this?
	end

	return nothing
end

#=
In-place matrix-matrix multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::AbstractMatrix, Hv::H, V::M) where {M<:AbstractMatrix{<:AbstractFloat}, H<:HvpOperator}
	for i=1:size(V,2)
		@views mul!(result[:,i], Hv, V[:,i])
	end

	return nothing
end

#=
In-place approximation of Hessian norm via power method

https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/0b2f1c5d352069df1bc891750087deda2d14cc9d/src/simple.jl#L58-L63

Input:
	Hv :: HvpOperator
=#
function eigmax(Hv::H; tol::T=1e-6, maxiter::I=Int(ceil(sqrt(size(Hv, 1))))) where {H<:HvpOperator, T<:AbstractFloat, I<:Integer}
	x0 = rand(eltype(Hv), size(Hv, 1))
    rmul!(x0, one(eltype(Hv)) / norm(x0))

	r = similar(x0)
	Ax = similar(x0)

	θ = 0.0

	for i=1:maxiter
		apply!(Ax, Hv, x0)

		θ = dot(x0, Ax)

		copyto!(r, Ax)
		axpy!(-θ, x0, r)

		res_norm = norm(r)

		if res_norm ≤ tol
			return abs(θ)
		end

		copyto!(x0, Ax)
		rmul!(x0, one(eltype(x0))/norm(x0))
	end

	return abs(θ)
end

#=
In-place approximation of Hessian^2 spectrum mean, i.e. trace/n

https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/0b2f1c5d352069df1bc891750087deda2d14cc9d/src/simple.jl#L58-L63

Input:
	Hv :: HvpOperator

NOTE: This is assuming the power is 2
NOTE: Don't need the tolerance as this is exact
=#
function eigmean(Hv::HvpOperator{T}) where {T}
	n = size(Hv, 1)

	trace = zero(T)
	ei = zeros(T, n)
	res = similar(ei)

	@inbounds for i = 1:n
		ei[i] = one(T)

		apply!(res, Hv, ei)
		trace += dot(res, res)

		ei[i] = zero(T)
	end

	return trace/n
end
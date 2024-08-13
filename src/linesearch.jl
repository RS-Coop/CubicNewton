#=
Author: Cooper Simpson

Line-search procedures.
=#

########################################################

function search!(opt::SFNOptimizer, stats::Stats, x::S, f::F, fval::T, g::S, g_norm::T, Hv::H) where {F, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    return search_M!(opt, stats, x, f, fval, g, g_norm, Hv)
end

########################################################

#=
In place SFN step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search_η!(opt::SFNOptimizer, stats::Stats, x::S, f::F, fval::T, g::S, g_norm::T, Hv::H) where {F, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    success = true
    λ = max(min(1e15, opt.M*g_norm), 1e-15)

    #Test search direction, select negative gradient if too small
    p_norm = norm(opt.solver.p)

    if p_norm < eps(T)
        stats.status = "Search direction too small"
        return !success
    end
    
    #Increase step-size
    η = 2.0

    #Scale search direction and norm
    p .*= η
    p_norm *= η 
    
    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    #NOTE: Can we just iteratively update x, is that even that much better?
    while true
        stats.f_evals += 1

        if f(x+p)-fval ≤ dec
            # println("Reduction: ", f(x+p)-fval, " Dec: ", dec)
            break
        else
            η *= opt.α #reduce step-size
            p .*= opt.α #scale search direction
            dec *= opt.α^2 #scale decrement
        end

        #Check step-size
        if η < eps(T)
            success = false
            stats.status = "Linesearch failed"
            break
        end
    end

    #Update regularization
    # println("Accepted η: ", η)
    opt.M = max(min(1e8, opt.M/η^2), 1e-8)
    # println("Updated M: ", opt.M)

    return success
end

########################################################

#=
In place SFN regularization line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search_M!(opt::SFNOptimizer, stats::Stats, x::S, f::F, fval::T, g::S, g_norm::T, Hv::H) where {F, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    success = true
    λ = max(min(1e15, opt.M*g_norm), 1e-15)

    #Test search direction, select negative gradient if too small
    p_norm = norm(opt.solver.p)

    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    if p_norm ≥ eps(T) && f(x+p)-fval ≤ dec
        opt.M *= opt.α #decrease regularization
    else
        opt.M /= opt.α #increase regularization
        success = false
    end

    return success
end

########################################################

#=
In place SFN step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search!(opt::ARCOptimizer, stats::Stats, x::S, f::F, fval::T, g::S, g_norm::T, Hv::H) where {F, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    #Cubic sub-problem
    cubic_subprob = (d) -> begin
        res = similar(g)
        mul!(res, Hv, d)
        return fval + dot(g,d) + 0.5*dot(d, res)
    end

    success = false
    shift_failure = false
    M_new = opt.M
    
    i = findfirst(opt.solver.krylov_solver.converged)

    if i === nothing
        return success
    end

    j = argmin(abs.(opt.M*opt.solver.shifts[i:end]-norm.(opt.solver.krylov_solver.x[i:end]))) + i-1

    while !success && !shift_failure
        stats.f_evals += 1

        ρ = (fval - f(x + opt.solver.krylov_solver.x[j]))/(fval - cubic_subprob(opt.solver.krylov_solver.x[j]))

        #unsuccessful
        if ρ < opt.η1
            M_new = opt.M

            while M_new > opt.γ1*opt.M
                if j == length(opt.solver.shifts)
                    stats.status = "No next shift"
                    shift_failure = true
                    break
                end
                M_new = norm(opt.solver.krylov_solver.x[j+1])/opt.solver.shifts[j+1]
                j += 1
            end
            
        #successful
        else
            success = true

            #step
            opt.solver.p .= opt.solver.krylov_solver.x[j]

            #very successful
            if ρ > opt.η2
                M_new = opt.γ2*opt.M
            else
                M_new = opt.M
            end
        end
    end

    opt.M = min(M_new, 1e15)

    return success
end
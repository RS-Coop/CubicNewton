#=
Author: Cooper Simpson

SFN optimizer.
=#

using Zygote: pullback
using Enzyme: make_zero!, ReverseWithPrimal

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::O, x::S, f::F; itmax::I=1000, time_limit::T2=Inf) where {O<:Optimizer, T1<:AbstractFloat, S<:AbstractVector{T1}, T2, F, I}
    #Setup hvp operator

    #NEW: Using Enzyme
    Hv = EHvpOperator(f, x, power=hvp_power(opt.solver))

    function fg!(grads::S, x::S)
        make_zero!(grads)

        _, fval = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, grads))

        return fval
    end

    #OLD: Using Zygote
    # Hv = RHvpOperator(f, x, power=hvp_power(opt.solver))
    
    # function fg!(grads::S, x::S)
        
    #     fval, back = let f=f; pullback(f, x) end
    #     grads .= back(one(fval))[1]

    #     return fval
    # end

    #iterate
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    g! :: inplace gradient function of f
    H :: hvp generator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::O, x::S, f::F1, fg!::F2, H::L; itmax::I=1000, time_limit::T=Inf) where {O<:Optimizer, T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
    #Setup hvp operator
    Hv = LHvpOperator(H, x, power=hvp_power(opt.solver))

    #iterate
    stats = iterate!(opt, x, f, fg!, Hv, itmax, time_limit)

    return stats
end

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    fg! :: compute f and gradient norm after inplace update of gradient
    Hv :: hvp operator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function iterate!(opt::O, x::S, f::F1, fg!::F2, Hv::H, itmax::I, time_limit::T) where {O<:Optimizer, T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, H<:HvpOperator, I}
    #Start time
    tic = time_ns()
    
    #Stats
    stats = Stats(T)
    converged = false
    iterations = 0
    
    #Gradient allocation
    grads = similar(x)

    #Compute function and gradient
    fval = fg!(grads, x)
    g_norm = norm(grads)

    #Estimate regularization
    if isnan(opt.M)
        opt.M = inv(g_norm)
    end

    # if opt.linesearch && isnan(opt.M)
        # ζ = randn(length(x))
        # D = norm(ζ)^2

        # g2 = similar(grads)
        # fg!(g2, x+ζ)

        # if any(isnan.(g2))
        #     opt.M = 1e15
        # else
        #     apply!(ζ, Hv, ζ) 
        #     ζ .= g2-grads-ζ

        #     opt.M = min(1e8, 2*norm(ζ)/(D))
        # end

        # g2 = nothing #mark for collection
    # end

    #Tolerance
    tol = opt.atol + opt.rtol*g_norm

    #Initial stats
    push!(stats.f_seq, fval)
    push!(stats.g_seq, g_norm)

    #Iterate
    while iterations<itmax+1

        #Check gradient norm
        if g_norm <= tol
            converged = true
            break
        end

        #Check other exit conditions
        time = elapsed(tic)

        if time>=time_limit
            stats.status = "Time limit exceeded"
            break
        elseif iterations==itmax
            stats.status = "Maximum iterations exceeded"
            break
        end

        #Step
        ##########

        #Solve for search direction
        step!(opt.solver, stats, Hv, grads, g_norm, opt.M, time_limit-time)

        #Linesearch
        if opt.linesearch && !search!(opt, stats, x, f, fval, grads, g_norm, Hv)
            break
        else
            x .+= opt.solver.p
        end
        ##########

        #Update function and gradient
        fval = fg!(grads, x)
        g_norm = norm(grads)

        #Update stats
        push!(stats.f_seq, fval)
        push!(stats.g_seq, g_norm)

        #Update Hvp operator
        update!(Hv, x)

        #Increment
        iterations += 1
    end

    #Update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.f_evals += iterations+1
    stats.hvp_evals = Hv.nprod
    stats.run_time = elapsed(tic)

    return stats
end
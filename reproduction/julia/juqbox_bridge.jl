using LinearAlgebra, SparseArrays, Random, JSON3, Ipopt
BLAS.set_num_threads(1)
const TIME_BUDGET_NUM_OPTIONS=Dict("tol"=>1e-300,"tiny_step_tol"=>0.0)

# Load pristine author numerical files without optional plotting/document packages.
module Juqbox
using LinearAlgebra, SparseArrays, Random, Printf, Ipopt
const ROOT = normpath(joinpath(@__DIR__, "third_party", "Juqbox.jl", "src"))
include(joinpath(ROOT, "bsplines.jl"))
include(joinpath(ROOT, "linear_solvers.jl"))
include(joinpath(ROOT, "StormerVerlet.jl"))
include(joinpath(ROOT, "ImplicitMidpoint.jl"))
const BsplineParams = Union{splineparams, bcparams}
const MyRealMatrix = Union{Array{Float64,2}, SparseMatrixCSC{Float64,Int64}}
const WeightMatrix = Union{Array{Float64,2}, Diagonal{Float64,Vector{Float64}}}
include(joinpath(ROOT, "evalobjgrad.jl"))
include(joinpath(ROOT, "ipopt_interface.jl"))
end

function build_case(task)
    if task == "juqbox_risk_neutral_swap02"
        ne, ng, T, D1 = [3], [1], 300.0, 12
        a = Matrix(Bidiagonal(zeros(4), sqrt.(1.0:3.0), :U))
        nn = Diagonal(0.0:3.0)
        H = Matrix(-pi*.2198*(nn*nn-nn))
        hs, ha, cf, rf = [a+a'], [a-a'], reshape([0.,-2pi*.2198],1,2), [4.10336]
        U0 = Juqbox.initial_cond(ne,ng)
        target = complex.(U0[:,[3,2,1]])
        bound = 2pi*.012/2
        steps = Juqbox.calculate_timestep(T,H,hs,ha,[2pi*.012])
        # Golub-Welsch normalized Gauss-Legendre weights, identical 9-node rule.
        eg = eigen(SymTridiagonal(zeros(9), [k/sqrt(4k*k-1) for k in 1:8]))
        nodes, weights = (2pi*.01).*eg.values, eg.vectors[1,:].^2
        tik, amp, tol, memory = .01, bound*.05, 1e-5, 5
        origin = "author Risk_Neutral/run_all.jl + swap-02-risk-neutral.jl; author default Tikhonov retained"
    elseif task == "juqbox_guarded_cnot"
        ne, ng, T, D1 = [2,2], [2,2], 250.0, 25
        aa = Matrix(Bidiagonal(zeros(4),sqrt.(1.0:3.0),:U))
        a, b = kron(Matrix{Float64}(I,4,4),aa), kron(aa,Matrix{Float64}(I,4,4))
        n1,n2 = a'*a,b'*b
        delta = (5.17839-5.06323)/2
        H = 2pi*(delta*n1-delta*n2-.3411/2*(n1*n1-n1)-.3413/2*(n2*n2-n2)+.001995*(a*b'+a'*b))
        hs, ha = [a+a',b+b'], [a-a',b-b']
        dressed = 2pi*sqrt(delta^2+.001995^2)
        cf,rf = [dressed -dressed; dressed -dressed], fill((5.17839+5.06323)/2,2)
        U0 = Juqbox.initial_cond(ne,ng)
        target = complex.(U0[:,[1,2,4,3]])
        bound,steps,nodes,weights = .040,20750,[0.],[1.]
        tik,amp,tol,memory = 0.,bound/100,.05,10
        origin = "paper arXiv:2106.14310v2 section 6.1 reconstructed configuration; native author numerical kernels"
    else
        error("Unknown task: $task")
    end
    p = Juqbox.objparams(ne,ng,T,steps,Uinit=U0,Utarget=target,Cfreq=cf,Rfreq=rf,
        Hconst=H,Hsym_ops=hs,Hanti_ops=ha,wmatScale=1.,Integrator=1)
    p.quiet=true; p.tik0=tik; p.objThreshold=-Inf; p.traceInfidelityThreshold=-Inf
    if task == "juqbox_guarded_cnot"
        w=zeros(16)
        for j in 0:3,i in 0:3
            if i>=2 || j>=2
                w[1+i+4j]= ((i,j) in ((0,2),(1,2),(2,0),(2,1))) ? 1/7000 : 1/7
            end
        end
        p.wmat=Diagonal(w); p.wmat_real=Diagonal(w)
    end
    nc = 2*length(hs)*2*D1
    lower,upper=fill(-bound,nc),fill(bound,nc)
    Juqbox.zero_start_end!(p,D1,lower,upper)
    Juqbox.estimate_Neumann!(eps(Float64),p,fill(bound,length(hs)))
    wa=Juqbox.Working_Arrays(p,nc)
    # Required by the native cached callback before the first evaluation.
    p.last_pcof=fill(1e9,nc); p.last_infidelity_grad=zeros(nc);p.last_leak_grad=zeros(0)
    free=findall(lower.<upper)
    meta=Dict("task"=>task,"source_commit"=>"3e6279d550afc214f5eccf99ba837cfc1d46b836",
        "paper"=>"https://arxiv.org/pdf/2106.14310v2","origin"=>origin,
        "T_ns"=>T,"nsteps"=>steps,"D1"=>D1,"ncoeff"=>nc,"nfree"=>length(free),
        "matrix_order"=>"F","physical_dimension"=>size(U0,1),"computational_dimension"=>size(U0,2),
        "ne"=>ne,"ng"=>ng,"H0"=>H,"Hsym"=>hs,"Hanti"=>ha,"target_real"=>real(target),
        "target_imag"=>imag(target),"U0"=>U0,"carrier_angular"=>cf,"W"=>diag(p.wmat),
        "nodes_angular"=>nodes,"weights_normalized"=>weights,"lower"=>lower,"upper"=>upper,
        "free_indices_zero_based"=>free.-1,"initial_uniform_halfwidth"=>amp,
        "tik0"=>tik,"tik_denominator"=>nc,"ipopt_tol"=>tol,"lbfgs_history"=>memory,
        "stopping_policy"=>"user_requested_time_budget_only",
        "effective_ipopt_stopping_numeric_options"=>TIME_BUDGET_NUM_OPTIONS,
        "effective_ipopt_acceptable_iter"=>0,"objective_threshold"=>"disabled",
        "julia_version"=>string(VERSION),"ipopt_julia_version"=>string(Base.pkgversion(Ipopt)),
        "blas_threads"=>BLAS.get_num_threads(),
        "integrator"=>"author Stormer-Verlet order 2; author Neumann implicit solver",
        "neumann_terms"=>p.linear_solver.max_iter,"objective"=>"weighted (trace infidelity + SV path leakage) + tik0*sum(alpha^2)/ncoeff")
    (;p,wa,nc,free,lower,upper,nodes,weights,D1,meta,tol,memory)
end

function author_oracle(c,x; gradient=true)
    p,wa=c.p,c.wa
    if gradient
        Juqbox.eval_f_g_grad!(x,p,wa,c.nodes,c.weights,true)
        g=copy(p.last_infidelity_grad)
        tg=zeros(c.nc)
        Juqbox.tikhonov_grad!(x,p,tg)
        g .+= tg
        f=p.last_infidelity+p.last_leak+Juqbox.tikhonov_pen(x,p)
    else
        Juqbox.eval_f_g_grad!(x,p,wa,c.nodes,c.weights,false)
        g=Float64[]
        f=p.last_infidelity+p.last_leak+Juqbox.tikhonov_pen(x,p)
    end
    Dict("f"=>f,"g"=>g,"gate_error"=>p.last_infidelity,"leakage"=>p.last_leak,
        "tikhonov"=>Juqbox.tikhonov_pen(x,p))
end

function export_trajectory(c,x,path)
    arrs=Array{ComplexF64,3}[]
    Hbase=copy(c.p.Hconst)
    for node in c.nodes
        c.p.Hconst .= Hbase
        for j in 2:size(Hbase,1)
            c.p.Hconst[j,j] += .01*node*10.0^(j-2)
        end
        push!(arrs,Juqbox.eval_forward(c.p.Uinit,x,c.p,saveEndOnly=false))
    end
    c.p.Hconst .= Hbase
    open(path,"w") do io
        for a in arrs
            write(io,a)
        end
    end
    Dict("path"=>path,"dtype"=>"complex128","order"=>"F","shape_per_system"=>size(first(arrs)),
        "systems"=>length(arrs),"max_column_norm_defect"=>maximum(abs(norm(a[:,j,k])^2-1) for a in arrs for j in axes(a,2) for k in axes(a,3)))
end

function native_ipopt(c,x,seconds,maxiter)
    t0=time_ns()
    p=c.p
    prob=Juqbox.setup_ipopt_problem(p,c.wa,c.nc,c.lower,c.upper,maxIter=maxiter,
        lbfgsMax=c.memory,ipTol=c.tol,acceptTol=1e-5,acceptIter=15,nodes=c.nodes,weights=c.weights)
    # Independent full-grid directional validation is saved by juqbox_validate.jl.
    # This removes a version-specific diagnostic, not an optimization update.
    Ipopt.AddIpoptStrOption(prob,"derivative_test","none")
    # User explicitly requested wall-time exploration without precision stopping.
    # Ipopt requires strictly positive tol. Preserve the absolute tolerances:
    # constr_viol_tol also participates in bound relaxation, not just stopping.
    for (key,value) in TIME_BUDGET_NUM_OPTIONS
        Ipopt.AddIpoptNumOption(prob,key,value)
    end
    Ipopt.AddIpoptIntOption(prob,"acceptable_iter",0)
    Ipopt.AddIpoptIntOption(prob,"print_level",0)
    Ipopt.AddIpoptNumOption(prob,"max_wall_time",seconds)
    hist=Any[]
    callback=function(am,k,obj,ipr,idu,mu,dnorm,reg,adu,apr,ls)
        xx=zeros(c.nc)
        Ipopt.GetIpoptCurrentIterate(prob,false,c.nc,xx,zeros(c.nc),zeros(c.nc),0,Float64[],Float64[])
        push!(hist,Dict("iteration"=>k,"elapsed"=>(time_ns()-t0)*1e-9,"objective"=>obj,
            "x"=>xx,"coordinate_source"=>"GetIpoptCurrentIterate",
            "dual_infeasibility"=>idu,"primal_infeasibility"=>ipr))
        keep=Juqbox.intermediate_par(am,k,obj,ipr,idu,mu,dnorm,reg,adu,apr,ls,p)
        return keep && (time_ns()-t0)*1e-9<seconds
    end
    Ipopt.SetIntermediateCallback(prob,callback)
    prob.x=copy(x)
    Ipopt.IpoptSolve(prob)
    Dict("x"=>copy(prob.x),"history"=>hist,"status"=>prob.status,"elapsed"=>(time_ns()-t0)*1e-9)
end

include(joinpath(@__DIR__,"juqbox_sensitivity.jl"))

function serve()
    task=ARGS[1]
    c=redirect_stdout(stderr) do
        build_case(task)
    end
    println(JSON3.write(Dict("ready"=>true,"metadata"=>c.meta))); flush(stdout)
    for line in eachline(stdin)
        isempty(line) && continue
        req=JSON3.read(line); op=String(req.op)
        op=="close" && break
        try
            out=redirect_stdout(stderr) do
                x=Float64.(req.x)
                if op=="reset"
                    p=c.p
                    empty!(p.objHist);empty!(p.primaryHist);empty!(p.secondaryHist);empty!(p.dualInfidelityHist)
                    empty!(p.pcof_hist)
                    p.last_pcof=fill(1e9,c.nc);p.last_infidelity_grad=zeros(c.nc);p.last_leak_grad=zeros(0)
                    Dict("reset"=>true)
                elseif op=="oracle"
                    a=author_oracle(c,x,gradient=Bool(req.derivatives))
                    if Bool(req.derivatives) && get(req,:curvature,false)
                        merge!(a,full_gn(c,x))
                    end
                    a
                elseif op=="trajectory"
                    export_trajectory(c,x,String(req.path))
                elseif op=="verify"
                    reduced=merge(c,(free=Int[],))
                    f=Juqbox.tikhonov_pen(x,c.p);leak=0.;gate=0.;defects=Float64[]
                    for (node,weight) in zip(c.nodes,c.weights)
                        a=system_gn(reduced,x,node)
                        gate+=weight*a.fidelity;leak+=weight*a.leak
                        push!(defects,a.norm_defect)
                    end
                    Dict("f"=>f+gate+leak,"gate_error"=>gate,"leakage"=>leak,
                        "sv_norm_defects"=>defects,"method"=>"independent SV formulas with LU implicit solves and original all-stage leakage quadrature")
                elseif op=="ipopt"
                    native_ipopt(c,x,Float64(req.seconds),Int(get(req,:maxiter,100000)))
                else
                    error("Unknown command $op")
                end
            end
            println(JSON3.write(Dict("ok"=>true,"result"=>out)))
        catch ex
            println(JSON3.write(Dict("ok"=>false,"error"=>sprint(showerror,ex,catch_backtrace()))))
        end
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    serve()
end

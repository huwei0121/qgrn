# Original author environment: Julia 1.9.2 and third_party/QuantumCollocation2023.jl.
# CLI: OUTPUT SEEDS SECONDS ROLE. Uses the source example's solver options.
using QuantumCollocation, NamedTrajectories, LinearAlgebra, Random, Dates, SHA
import Ipopt
import MathOptInterface as MOI
const AJ = Base.require(Base.PkgId(Base.UUID("682c06a0-de6a-54ab-a142-c8b1cf79cde6"),"JSON"))
BLAS.set_num_threads(1)
const AP_ROOT=@__DIR__
ap_clean(x)=x
ap_clean(x::AbstractFloat)=isfinite(x) ? x : nothing
ap_clean(x::AbstractArray)=map(ap_clean,x)
ap_clean(x::Tuple)=map(ap_clean,x)
ap_clean(x::NamedTuple)=Dict(string(k)=>ap_clean(v) for (k,v) in pairs(x))
ap_clean(x::AbstractDict)=Dict(string(k)=>ap_clean(v) for (k,v) in pairs(x))
ap_arr(x)=(;shape=collect(size(x)),values=vec(x),order="F")
ap_complex(x)=(;shape=collect(size(x)),real=vec(real.(x)),imag=vec(imag.(x)),order="F")
function ap_save(path,x)
    ispath(path) && error("Output file already exists: $path")
    write(path,AJ.json(ap_clean(x)))
end
function ap_build(seed;task="pi")
    Random.seed!(seed)
    if task=="swap"
        aa=ComplexF64[0 1;0 0];E=Matrix{ComplexF64}(I,2,2)
        lift3(A,j)=reduce(kron,[q==j ? A : E for q in 1:3])
        ops=[lift3(aa,j) for j in 1:3]
        ws=2pi*[5.18,5.12,5.06];wd=2pi*5.12
        H0=sum((ws[j]-wd)*(ops[j]'*ops[j])-.34pi*(ops[j]'*ops[j]'*ops[j]*ops[j]) for j in 1:3)
        H0+=.01pi*(ops[1]'*ops[2]+ops[1]*ops[2]'+ops[2]'*ops[3]+ops[2]*ops[3]')
        Hs=vcat([a+a' for a in ops],[im*(a-a') for a in ops]);G=zeros(ComplexF64,8,8)
        for j in 0:7
            G[((j&1)<<2)+(j&2)+((j>>2)&1)+1,j+1]=1
        end
        bounds=fill(.08pi,6);sys=QuantumSystem(H0,Hs)
        opts=Options(watchdog_shortened_iter_trigger=0,watchdog_trial_iter_max=10)
        prob=UnitarySmoothPulseProblem(sys,G,500,.4;Q=200.,R=.001,a_bound=.08pi,
            dda_bound=.05,Δt_min=.2,Δt_max=.4,max_iter=10000,linear_solver="mumps",ipopt_options=opts)
        return (;prob,H0,Hs,G,bounds,K=500,h=.4,Q=200.,R=.001,acceleration=.05,free_time=true,
            h_bounds=(.2,.4),task_id="author2023_swap13",source="examples/three_qubit_swap/cold_start.jl",
            source_note="Uses the example's cold-start branch (load_trajectory=false), with its model and Ipopt settings. The example's default trajectory path points to a PNG file.")
    end
    task=="pi" || error("Unknown author task")
    levels=2; alpha=-2pi*.143
    H0=alpha/2*quad(levels)
    Hs=[create(levels)+annihilate(levels),im*(create(levels)-annihilate(levels))]
    G=-im*ComplexF64[0 1;1 0]
    bounds=fill(2pi*.019,2)
    sys=QuantumSystem(H0,Hs)
    opts=Options(watchdog_shortened_iter_trigger=0)
    prob=UnitarySmoothPulseProblem(sys,G,100,.4;free_time=false,
        a_bounds=bounds,dda_bound=.05,max_iter=500,Q=200.,R=.01,ipopt_options=opts)
    (;prob,H0,Hs,G,bounds,K=100,h=.4,Q=200.,R=.01,acceleration=.05,free_time=false,
        h_bounds=(0.,Inf),task_id="author2023_transmon_pi",source="examples/transmon/pi_gate.jl",source_note="Original author example")
end
function ap_slice(prob,z)
    t=prob.trajectory;Z=reshape(z,t.dim,t.T)
    dt=t.timestep isa Symbol ? vec(Z[t.components[t.timestep],:]) : fill(t.timestep,t.T)
    (;u=ap_arr(Z[t.components.a,:]),du=ap_arr(Z[t.components.da,:]),
      ddu=ap_arr(Z[t.components.dda,:]),states=ap_arr(Z[t.components.Ũ⃗,:]),dt)
end
function ap_run(seed,seconds;warmup=false,task="pi")
    started=time_ns()/1e9
    model=ap_build(seed;task);p=model.prob
    build=time_ns()/1e9-started
    initial=copy(p.trajectory.datavec);history=Any[];callback_errors=Any[]
    # Observation is a stock Ipopt callback. It only saves the iterate and enforces
    # the common external wall allowance; no tolerance, derivative or linear solver change.
    MOI.set(p.optimizer,Ipopt.CallbackFunction(),(mode,k,f,pr,du,mu,dn,reg,ad,ap,ls)->begin
        try
            opt=p.optimizer; n=opt.inner.n;x=zeros(n)
            # Request primal values only. Asking for fixed-variable bound
            # multipliers would trigger additional author derivative work.
            Ipopt.GetIpoptCurrentIterate(opt.inner,false,n,x,C_NULL,C_NULL,0,C_NULL,C_NULL)
            push!(history,(;k,elapsed=time_ns()/1e9-started,objective=f,inf_pr=pr,inf_du=du,
                mu,regularization=reg,alpha=ap,mode,slice=ap_slice(p,x)))
        catch err
            push!(callback_errors,(;k,error=sprint(showerror,err)))
            return false
        end
        warmup ? k<2 : time_ns()/1e9-started<seconds
    end)
    native_options=Dict(string(k)=>getfield(p.options,k) for k in fieldnames(typeof(p.options)))
    solve!(p)
    z=copy(p.trajectory.datavec)
    # Ipopt.jl 1.4.1 implements setting NLPBlock, but not its MOI getter.
    # Read the exact evaluator already used by the unmodified native optimizer.
    ev=p.optimizer.nlp_data.evaluator
    f=MOI.eval_objective(ev,z)
    final_slice=ap_slice(p,z);initial_slice=ap_slice(p,initial)
    U=Matrix{ComplexF64}(I,size(model.H0,1),size(model.H0,1))
    for k in 1:model.K-1
        u=reshape(final_slice.u.values,Tuple(final_slice.u.shape))
        H=model.H0+sum(u[j,k]*model.Hs[j] for j in eachindex(model.Hs))
        U=exp(-im*final_slice.dt[k]*H)*U
    end
    physical_ell=abs(1-abs(tr(model.G'*U))/size(U,1))
    reg=.5model.R*(sum(abs2,final_slice.u.values)+sum(abs2,final_slice.du.values)+sum(abs2,final_slice.ddu.values))
    record=(;task=model.task_id,method="author_ipopt",seed,role=warmup ? "warmup" : "main",
      budget_seconds=seconds,build_seconds=build,
      status=string(MOI.get(p.optimizer,MOI.TerminationStatus())),
      raw_status=MOI.get(p.optimizer,MOI.RawStatusString()),
      native_options,final_objective=f,initial=initial_slice,final=final_slice,
      physical=(;infidelity=physical_ell,objective=model.Q*physical_ell+reg,unitarity=norm(U'*U-I)),
      initial_full=ap_arr(initial),final_full=ap_arr(z),history,callback_errors,
      model=(;K=model.K,h=model.h,T=(model.K-1)*model.h,Q=model.Q,R=model.R,
        amplitude=model.bounds,acceleration=model.acceleration,free_time=model.free_time,h_bounds=model.h_bounds,
        H0=ap_complex(model.H0),Hs=ap_complex.(model.Hs),target=ap_complex(model.G),
        components=Dict(string(k)=>collect(v) for (k,v) in pairs(p.trajectory.components)),
        propagation="pade22",loss="unsquared",regularizer="node_sum",state_layout="column_interleaved_real_imag",
        objective="Q*abs(1-abs(tr(G' U))/d)+.5*R*sum(u^2+du^2+ddu^2)",
        source="third_party/QuantumCollocation2023.jl/"*model.source,source_note=model.source_note,
        author_commit="d3b33912ebaf7b335f9e9a8619399cbc59ec89b2",
        initialization="Unmodified author template: geodesic U, uniform independent amplitudes, random independent derivatives"))
    merge(record,(;elapsed=time_ns()/1e9-started))
end
function ap_main()
    length(ARGS) in (4,5) || error("OUTPUT SEEDS SECONDS ROLE [pi|swap]")
    out=abspath(ARGS[1]);mkpath(out)
    seeds=parse.(Int,split(ARGS[2],','));seconds=parse(Float64,ARGS[3]);role=ARGS[4]
    task=length(ARGS)==5 ? ARGS[5] : "pi"
    ap_save(joinpath(out,"protocol.json"),(;task,seeds,seconds,role,
        julia=string(VERSION),ipopt_jl=string(pkgversion(Ipopt)),threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),created_utc=string(now(UTC)),
        author_commit="d3b33912ebaf7b335f9e9a8619399cbc59ec89b2",
        scope="Author original transmon pi or free-time SWAP cold-start branch; original package and locked Manifest, unchanged NLP and mathematical Ipopt settings",
        observation="Stock callback records iterates and enforces external common wall allowance; two-iteration warmup excluded; full build+solve counted"))
    srcs=["pico_template.jl","third_party/QuantumCollocation2023.jl/Project.toml",
        "third_party/QuantumCollocation2023.jl/Manifest.toml"]
    append!(srcs,[replace(relpath(joinpath(r,f),AP_ROOT),'\\'=>'/') for (r,_,fs) in walkdir(joinpath(AP_ROOT,"third_party/QuantumCollocation2023.jl/src")) for f in fs if endswith(f,".jl")])
    hashes=Dict{String,String}()
    for rel in srcs
        src=joinpath(AP_ROOT,rel);dst=joinpath(out,"source",rel);mkpath(dirname(dst));cp(src,dst)
        hashes[rel]=bytes2hex(sha256(read(src)))
    end
    ap_save(joinpath(out,"source_manifest.json"),hashes)
    println("OUTPUT="*out);flush(stdout)
    warm=ap_run(29,seconds;warmup=true,task);ap_save(joinpath(out,"warmup.json"),warm)
    for seed in seeds
        println("START seed="*string(seed));flush(stdout)
        result=merge(ap_run(seed,seconds;task),(;role))
        ap_save(joinpath(out,"s$(seed)_author_ipopt.json"),result)
        println("DONE seed=$seed status=$(result.status) wall=$(result.elapsed)");flush(stdout)
    end
    ap_save(joinpath(out,"complete.json"),(;complete=true,runs=length(seeds),finished_utc=string(now(UTC))))
end
abspath(PROGRAM_FILE)==(@__FILE__) && ap_main()

# Native SWAP driver for Julia 1.9.2 / QuantumCollocation2023.
# Commands: generate INPUT SEED | run INPUT OUTPUT SECONDS [MAXITER]
#           batch INPUTDIR OUTDIR SEEDS SECONDS [WARMSEED]
# Input generation and optimization use explicit subcommands.
include("pico_input.jl")

const SWAP_OBJECTIVE_SEMANTICS = "physical_trace_unsquared_with_original_node_regularizer"
sw_array(a)=reshape(Float64.(a["values"]),Tuple(Int.(a["shape"])))
sw_complex(a)=reshape(ComplexF64.(a["real"],a["imag"]),Tuple(Int.(a["shape"])))

"""Construct the physical trace loss and its analytic derivatives.

For column-interleaved x=iso(U), a=iso(G), b=iso(iG), the physical loss is
abs(1-hypot(a'x,b'x)/d). The original author endpoint Hessian sparsity and
node-summed quadratic regularizers are retained. The loss keeps its absolute-value
cusp and zero-overlap singularity. Both comparison methods use this corrected loss.
"""
function sw_terminal_objective(traj,G,Q)
    O=QuantumCollocation.Objectives
    a=operator_to_iso_vec(G);b=operator_to_iso_vec(im*G);d=size(G,1)
    ids=(traj.T-1)*traj.dim .+ traj.components.Ũ⃗
    local_pairs=[(i,j) for (i,j) in Iterators.product(1:length(a),1:length(a)) if i<=j]
    pairs=[(ids[i],ids[j]) for (i,j) in local_pairs]
    function data(x)
        s=dot(a,x);t=dot(b,x);r=hypot(s,t);F=r/d
        (;r,F,v=s*a+t*b)
    end
    L=(z,Z)->Q*abs(1-data(z[ids]).F)
    function grad(z,Z)
        v=data(z[ids]);g=zeros(length(z))
        g[ids]=-Q*sign(1-v.F)*v.v/(d*v.r)
        g
    end
    function hess(z,Z)
        v=data(z[ids]);H=-Q*sign(1-v.F)*((a*a'+b*b')/(d*v.r)-v.v*v.v'/(d*v.r^3))
        [H[i,j] for (i,j) in local_pairs]
    end
    O.Objective(L,grad,hess,Z->pairs,Dict[Dict(:type=>:PhysicalTraceInfidelityCorrection,:Q=>Q)])
end

function sw_correct_objective!(p,m)
    ev=ap2_evaluator(p);original=ev.objective;O=QuantumCollocation.Objectives
    corrected=sw_terminal_objective(p.trajectory,m.G,m.Q)
    for name in (:a,:da,:dda)
        corrected+=O.QuadraticRegularizer(name,p.trajectory,m.R)
    end
    corrected.∂²L_structure(p.trajectory)==original.∂²L_structure(p.trajectory) || error("Hessian sparsity changed")
    ev.objective=corrected
    original
end

function sw_disable_precision_stops!(p)
    # Use the benchmark time budget as the stopping rule. The original
    # convergence settings are recorded in paper_native_options.
    p.options.tol=1e-300
    p.options.acceptable_iter=0
    # This original Ipopt option is absent from the author's Options struct.
    # solve! sets listed fields only, so this native attribute is retained.
    MOI.set(p.optimizer,MOI.RawOptimizerAttribute("tiny_step_tol"),0.)
    nothing
end

function sw_source_hashes()
    files=["pico_native.jl","pico_input.jl",
        "pico_template.jl","third_party/QuantumCollocation2023.jl/Project.toml",
        "third_party/QuantumCollocation2023.jl/Manifest.toml"]
    append!(files,[replace(relpath(joinpath(r,f),AP_ROOT),'\\'=>'/') for (r,_,fs) in walkdir(joinpath(AP_ROOT,"third_party/QuantumCollocation2023.jl/src")) for f in fs if endswith(f,".jl")])
    Dict(f=>bytes2hex(sha256(read(joinpath(AP_ROOT,f)))) for f in files)
end

function sw_generate(path,seed;role="benchmark")
    ispath(path) && error("Existing input: $path")
    input=ap2_generate(seed;task="swap")
    raw=ap2_input_record(input,seed,role)
    mkpath(dirname(abspath(path)))
    ap_save(path,merge(raw,(;comparison_objective_semantics=SWAP_OBJECTIVE_SEMANTICS,
        objective_correction="Original template loss has a column/half-vector layout defect. Physical trace loss is explicitly corrected for both compared algorithms.",
        source_hashes=sw_source_hashes())))
    raw
end

function sw_load(path)
    raw=AJ.parsefile(path)
    raw["schema"]=="author2023_shared_input_v2" && raw["task"]=="author2023_swap13" || error("Wrong task/input")
    s=raw["initial"];m=ap2_definition("swap");a=raw["model"]
    (a["K"]==m.K && a["Q"]==m.Q && a["R"]==m.R && a["h"]==m.h && a["free_time"]==m.free_time && a["h_bounds"]==collect(m.h_bounds)) || error("Incomplete/modified model")
    norm(sw_complex(a["H0"])-m.H0)<1e-14 || error("Wrong drift")
    all(norm(sw_complex(v)-H)<1e-14 for (v,H) in zip(a["Hs"],m.Hs)) || error("Wrong drives")
    norm(sw_complex(a["target"])-m.G)<1e-14 || error("Wrong target")
    vals=vcat([vec(sw_array(s[n])) for n in ("states","u","du","ddu")]...,Float64.(s["dt"]))
    bytes2hex(sha256(reinterpret(UInt8,vals)))==raw["initial_sha256"] || error("Input hash mismatch")
    components=(;Ũ⃗=sw_array(s["states"]),a=sw_array(s["u"]),da=sw_array(s["du"]),dda=sw_array(s["ddu"]),Δt=reshape(Float64.(s["dt"]),1,m.K))
    initial=(;Ũ⃗=operator_to_iso_vec(Matrix{ComplexF64}(I,8,8)),a=zeros(6));final=(;a=zeros(6));goal=(;Ũ⃗=operator_to_iso_vec(m.G))
    bounds=(;a=m.bounds,dda=fill(m.acceleration,6),Δt=m.h_bounds)
    traj=NamedTrajectory(components;controls=(:dda,:Δt),timestep=:Δt,bounds,initial,final,goal)
    length(traj.datavec)==length(raw["initial_full"]["values"]) || error("Full coordinates mismatch")
    maximum(abs.(traj.datavec-Float64.(raw["initial_full"]["values"])))==0 || error("Imported coordinates changed")
    raw,(;m,system=QuantumSystem(m.H0,m.Hs),traj)
end

function sw_physical_slice(m,s)
    u=reshape(s.u.values,Tuple(s.u.shape));du=reshape(s.du.values,Tuple(s.du.shape));ddu=reshape(s.ddu.values,Tuple(s.ddu.shape))
    isos=reshape(s.states.values,Tuple(s.states.shape));states=[iso_vec_to_operator(isos[:,k]) for k in 1:m.K]
    maximum_abs(x)=maximum(abs,x;init=0.)
    err(U)=abs(1-abs(dot(m.G,U))/8)
    reg=.5*m.R*(sum(abs2,u)+sum(abs2,du)+sum(abs2,ddu))
    checks=Dict("initial_state"=>maximum_abs(states[1]-I),
        "control_chain"=>maximum_abs(u[:,2:end]-u[:,1:end-1]-du[:,1:end-1].*s.dt[1:end-1]'),
        "derivative_chain"=>maximum_abs(du[:,2:end]-du[:,1:end-1]-ddu[:,1:end-1].*s.dt[1:end-1]'),
        "endpoints"=>max(maximum_abs(u[:,1]),maximum_abs(u[:,end])),
        "amplitude"=>max(0.,maximum(abs.(u).-m.bounds)),"acceleration"=>max(0.,maximum_abs(ddu)-m.acceleration),
        "time_bounds"=>max(0.,m.h_bounds[1]-minimum(s.dt),maximum(s.dt)-m.h_bounds[2]),"equal_time"=>maximum_abs(s.dt.-s.dt[1]))
    Up=Matrix{ComplexF64}(I,8,8);Ue=copy(Up);defect=0.;pade_difference=0.;expm_difference=0.
    for k in 1:m.K-1
        H=m.H0+sum(u[j,k]*m.Hs[j] for j in 1:6);A=-im*s.dt[k]*H
        B=I-A/2+A*A/12;C=I+A/2+A*A/12
        defect=max(defect,maximum_abs(B*states[k+1]-C*states[k]))
        Up=(B\C)*Up;Ue=exp(A)*Ue
        pade_difference=max(pade_difference,maximum_abs(states[k+1]-Up));expm_difference=max(expm_difference,maximum_abs(states[k+1]-Ue))
    end
    checks["quantum_dynamics"]=defect;checks["primal"]=maximum(values(checks))
    half=QuantumCollocation.Losses.unitary_infidelity(isos[:,end],operator_to_iso_vec(m.G))
    (;final_objective=m.Q*err(states[end])+reg,terminal_loss=err(states[end]),regularizer=reg,
        historical_half_objective=m.Q*half+reg,historical_half_loss=half,
        pade_objective=m.Q*err(Up)+reg,pade_gate_error=err(Up),physical_objective=m.Q*err(Ue)+reg,physical_gate_error=err(Ue),
        feasibility=checks["primal"],constraints=checks,stored_pade_difference=pade_difference,stored_expm_difference=expm_difference,
        physical_unitarity=norm(Ue'*Ue-I))
end

function sw_audit_history(m,history)
    [try
        check=sw_physical_slice(m,h.slice)
        merge(h,(;raw_callback_constraint_violation=h.inf_pr,
            constraint_violation=check.feasibility,feasibility=check.feasibility,
            constraint_violation_semantics="Independent all-node original-model residual/bound maximum; complex quantum residual entry modulus, hence at least as strict as separate real/imag maxima; not scaled Ipopt inf_pr",
            physical_objective=check.physical_objective,pade_objective=check.pade_objective,
            historical_half_objective=check.historical_half_objective,
            physical_gate_error=check.physical_gate_error,original_constraints=check.constraints,
            offline_audit_valid=abs(check.final_objective-h.objective)<=1e-7*max(1.,abs(h.objective))))
    catch err
        merge(h,(;constraint_violation=nothing,feasibility=nothing,offline_audit_valid=false,
            offline_audit_error=sprint(showerror,err)))
    end for h in history]
end

function sw_run(input_path,seconds;maxiter=10000)
    isfinite(seconds) && seconds>=0 || error("Finite nonnegative seconds required")
    maxiter>=0 || error("Nonnegative maxiter required")
    hashes=sw_source_hashes();started=time_ns();clock()=(time_ns()-started)/1e9
    raw=nothing;history=Any[];errors=Any[];stage="load_complete_shared_input"
    try
        raw,input=sw_load(input_path);stage="native_template_build"
        p=ap2_native_problem(input);legacy=sw_correct_objective!(p,input.m)
        paper_native_options=Dict(string(k)=>getfield(p.options,k) for k in fieldnames(typeof(p.options)))
        sw_disable_precision_stops!(p)
        initial_slice=ap_slice(p,p.trajectory.datavec);initial=sw_physical_slice(input.m,initial_slice)
        native_options=Dict(string(k)=>getfield(p.options,k) for k in fieldnames(typeof(p.options)))
        native_options["tiny_step_tol"]=0.
        setup_seconds=clock();stage="native_solve";external_time_stop=Ref(false)
        MOI.set(p.optimizer,Ipopt.CallbackFunction(),(mode,k,f,pr,du,mu,dn,reg,ad,alpha,ls)->begin
            try
                x=zeros(p.optimizer.inner.n)
                Ipopt.GetIpoptCurrentIterate(p.optimizer.inner,false,length(x),x,C_NULL,C_NULL,0,C_NULL,C_NULL)
                # Pure observation of original native values; no objective/constraint replay here.
                push!(history,(;iteration=k,k,elapsed=clock(),objective=f,inf_pr=pr,
                    constraint_violation=pr,constraint_violation_semantics="Raw Ipopt intermediate-callback inf_pr; internal scaling is not independently removed. Not a certificate of original-model primal feasibility.",
                    inf_du=du,mu,regularization=reg,alpha,mode,slice=ap_slice(p,x)))
                external_time_stop[]=clock()>=seconds
                return !external_time_stop[]
            catch err
                push!(errors,sprint(showerror,err));return false
            end
        end)
        if setup_seconds>=seconds
            status="SETUP_TIME_LIMIT";raw_status="Complete-clock budget exhausted before optimize!"
        else
            # All native options are preserved; maxiter only changes for explicitly marked smoke.
            solve!(p;max_iter=maxiter)
            status=string(MOI.get(p.optimizer,MOI.TerminationStatus()));raw_status=MOI.get(p.optimizer,MOI.RawStatusString())
        end
        solve_return_seconds=clock();stage="full_return_and_independent_validation"
        z=copy(p.trajectory.datavec);final_slice=ap_slice(p,z);returned=sw_physical_slice(input.m,final_slice)
        f=MOI.eval_objective(ap2_evaluator(p),z)
        legacy_f=legacy.L(z,p.trajectory)
        record=(;schema="pico_swap_candidate_v1",task="pico_swap13",seed=raw["seed"],method="author_ipopt_physical_loss_corrected",
            budget_seconds=seconds,target_early_stopping=false,precision_early_stopping=false,
            stopping_policy="user_requested_time_budget_only",objective_semantics=SWAP_OBJECTIVE_SEMANTICS,
            objective_correction="Terminal column-wise complex structure and its analytic derivatives are corrected. Dynamics, controls, original solve! API and update settings are retained; user-requested stopping-policy overrides are separately enumerated.",
            input_file=abspath(input_path),input_file_sha256=bytes2hex(sha256(read(input_path))),initial_coordinate_sha256=raw["initial_sha256"],
            model=raw["model"],initial,initial_slice,initial_full=raw["initial_full"],returned,final_slice,final_full=ap_arr(z),
            final_objective=f,historical_half_objective=legacy_f,native_options,paper_native_options,actual_maxiter=p.options.max_iter,
            stopping_policy_changes=(;tol=1e-300,acceptable_iter=0,tiny_step_tol=0.,
                preserved_absolute_tolerances=("dual_inf_tol","constr_viol_tol","compl_inf_tol"),
                reason="User disabled precision stopping: overall optimality and all absolute tests are AND conditions. Preserve absolute tolerances, including the bound-relaxation cap; tiny positive tol is an API-compatible disable approximation, not an accuracy claim. Retain numerical failure and time/iteration safeguards"),
            history,callback_errors=errors,status,raw_status,external_time_stop=external_time_stop[],
            return_valid=isempty(errors) && isfinite(f) && abs(f-returned.final_objective)<=1e-7*max(1.,abs(f)),
            returned_primal_feasible=returned.feasibility<=1e-6,failure=!isempty(errors),
            setup_seconds,solve_return_seconds,shared_input_generation_seconds=raw["generation_seconds"],
            timing_note="Complete clock starts before input loading/model construction. Native first-call compilation, callback recording, full return and independent final all-node Pade/expm replay are included; import/process startup and previously generated shared input excluded. Warmup is separate. Offline per-history physical audit occurs after complete_seconds and is separately timed. External wall budget is checked at native iteration boundaries, so overrun is retained.",
            source_hashes=hashes,source_unchanged=hashes==sw_source_hashes(),julia=string(VERSION),ipopt_jl=string(pkgversion(Ipopt)))
        complete_seconds=clock()
        history_audit_started=clock()
        audited_history=sw_audit_history(input.m,history)
        offline_history_validation_seconds=clock()-history_audit_started
        merge(record,(;history=audited_history,complete_seconds,over_budget_seconds=max(0.,complete_seconds-seconds),
            offline_history_validation_seconds,record_ready_seconds=clock()))
    catch err
        complete_seconds=clock()
        (;schema="pico_swap_candidate_v1",task="pico_swap13",seed=raw===nothing ? nothing : raw["seed"],method="author_ipopt_physical_loss_corrected",
            budget_seconds=seconds,status="EXCEPTION",failure=true,return_valid=false,stage,error=sprint(showerror,err,catch_backtrace()),
            initial_coordinate_sha256=raw===nothing ? nothing : raw["initial_sha256"],input_file=abspath(input_path),
            history,complete_seconds,over_budget_seconds=max(0.,complete_seconds-seconds),source_hashes=hashes,source_unchanged=hashes==sw_source_hashes())
    end
end

function sw_main()
    isempty(ARGS) && return println("Usage: generate INPUT SEED | run INPUT OUTPUT SECONDS [MAXITER] | batch INPUTDIR OUTDIR SEEDS SECONDS [WARMSEED]")
    cmd=ARGS[1]
    if cmd=="generate"
        length(ARGS)==3 || error("generate INPUT SEED")
        sw_generate(ARGS[2],parse(Int,ARGS[3]))
    elseif cmd=="run"
        length(ARGS) in (4,5) || error("run INPUT OUTPUT SECONDS [MAXITER]")
        ispath(ARGS[3]) && error("Existing output")
        r=sw_run(ARGS[2],parse(Float64,ARGS[4]);maxiter=length(ARGS)==5 ? parse(Int,ARGS[5]) : 10000)
        mkpath(dirname(abspath(ARGS[3])));ap_save(ARGS[3],r)
        println("DONE $(r.status) $(r.complete_seconds)")
    elseif cmd=="batch"
        length(ARGS) in (5,6) || error("batch INPUTDIR OUTDIR SEEDS SECONDS [WARMSEED]")
        indir,out,seedsarg,secarg=ARGS[2:5];ispath(out) && error("Existing output directory")
        seeds=parse.(Int,split(seedsarg,','));length(unique(seeds))==length(seeds) || error("Duplicate seeds")
        seconds=parse(Float64,secarg);warmseed=length(ARGS)==6 ? parse(Int,ARGS[6]) : 85200
        mkpath(out);mkpath(indir)
        warmfile=joinpath(indir,"s$(warmseed).input.json")
        ispath(warmfile) || sw_generate(warmfile,warmseed;role="warmup")
        warm=sw_run(warmfile,120.;maxiter=1);ap_save(joinpath(out,"warmup.json"),warm)
        warm.status=="EXCEPTION" && error("Warmup failed; see warmup.json for details")
        for seed in seeds
            path=joinpath(indir,"s$(seed).input.json");ispath(path) || sw_generate(path,seed)
            r=sw_run(path,seconds);ap_save(joinpath(out,"pico_swap13_author_ipopt_physical_loss_corrected_$(seed).json"),r)
            println("DONE $seed $(r.status) $(r.complete_seconds)");flush(stdout)
        end
    else
        error("Unknown command; no optimization started")
    end
end

abspath(PROGRAM_FILE)==(@__FILE__) && sw_main()

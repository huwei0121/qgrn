# Shared input construction in the Julia 1.9.2 / QuantumCollocation2023 environment.
include("pico_template.jl")

# Ipopt.jl 1.4.1 stores the original evaluator here but has no MOI.NLPBlock getter.
# This is the same evaluator used internally by its native objective callbacks.
ap2_evaluator(problem)=problem.optimizer.nlp_data.evaluator

function ap2_definition(task)
    if task=="swap"
        aa=ComplexF64[0 1;0 0];E=Matrix{ComplexF64}(I,2,2)
        lift3(A,j)=reduce(kron,[q==j ? A : E for q in 1:3])
        ops=[lift3(aa,j) for j in 1:3];ws=2pi*[5.18,5.12,5.06];wd=2pi*5.12
        H0=sum((ws[j]-wd)*(ops[j]'*ops[j])-.34pi*(ops[j]'*ops[j]'*ops[j]*ops[j]) for j in 1:3)
        H0+=.01pi*(ops[1]'*ops[2]+ops[1]*ops[2]'+ops[2]'*ops[3]+ops[2]*ops[3]')
        Hs=vcat([a+a' for a in ops],[im*(a-a') for a in ops]);G=zeros(ComplexF64,8,8)
        for j in 0:7
            G[((j&1)<<2)+(j&2)+((j>>2)&1)+1,j+1]=1
        end
        return (;H0,Hs,G,bounds=fill(.08pi,6),K=500,h=.4,Q=200.,R=.001,
            acceleration=.05,free_time=true,h_bounds=(.2,.4),task_id="author2023_swap13",
            max_iter=10000,source="examples/three_qubit_swap/cold_start.jl",
            source_note="Original load_trajectory=false cold-start branch",
            options=Options(watchdog_shortened_iter_trigger=0,watchdog_trial_iter_max=10))
    end
    task=="pi" || error("Unknown author task: $task")
    H0=(-2pi*.143)/2*quad(2)
    Hs=[create(2)+annihilate(2),im*(create(2)-annihilate(2))]
    G=-im*ComplexF64[0 1;1 0]
    (;H0,Hs,G,bounds=fill(2pi*.019,2),K=100,h=.4,Q=200.,R=.01,
        acceleration=.05,free_time=false,h_bounds=(0.,Inf),task_id="author2023_transmon_pi",
        max_iter=500,source="examples/transmon/pi_gate.jl",source_note="Original author example",
        options=Options(watchdog_shortened_iter_trigger=0))
end

"""Reproduce only the original template initialization, without any NLP object."""
function ap2_generate(seed;task="pi")
    started=time_ns()/1e9
    Random.seed!(seed);m=ap2_definition(task);system=QuantumSystem(m.H0,m.Hs)
    G=Matrix{ComplexF64}(m.G);U0=operator_to_iso_vec(1.0I(size(G,1)))
    n=length(system.G_drives);h=m.free_time ? fill(m.h,1,m.K) : m.h
    # Exactly the original RNG ordering, including geodesic fallback.
    geodesic_success=true;local Uiso
    try
        Uiso=QuantumCollocation.ProblemTemplates.unitary_geodesic(G,m.K)
    catch err
        @warn "Could not find geodesic. Using random initial guess."
        geodesic_success=false
    end
    if !geodesic_success
        Uiso=2*rand(length(U0),m.K).-1
    end
    dists=[QuantumCollocation.ProblemTemplates.Uniform(-m.bounds[i],m.bounds[i]) for i in 1:n]
    a=hcat([zeros(n),vcat([rand(dists[i],1,m.K-2) for i in 1:n]...),zeros(n)]...)
    da=randn(n,m.K)*.01;dda=randn(n,m.K)*.01
    initial=(;Ũ⃗=U0,a=zeros(n));final=(;a=zeros(n));goal=(;Ũ⃗=operator_to_iso_vec(G))
    if m.free_time
        components=(;Ũ⃗=Uiso,a,da,dda,Δt=h)
        bounds=(;a=m.bounds,dda=fill(m.acceleration,n),Δt=m.h_bounds)
        traj=NamedTrajectory(components;controls=(:dda,:Δt),timestep=:Δt,bounds,initial,final,goal)
    else
        components=(;Ũ⃗=Uiso,a,da,dda)
        bounds=(;a=m.bounds,dda=fill(m.acceleration,n))
        traj=NamedTrajectory(components;controls=(:dda,),timestep=h,bounds,initial,final,goal)
    end
    generation_seconds=time_ns()/1e9-started
    (;m,system,traj,generation_seconds,geodesic_success)
end

"""Invoke the unchanged author template, skipping ONLY its initialization branch."""
function ap2_native_problem(input)
    m=input.m
    if m.free_time
        UnitarySmoothPulseProblem(input.system,m.G,m.K,m.h;init_trajectory=input.traj,
            Q=m.Q,R=m.R,a_bound=m.bounds[1],dda_bound=m.acceleration,
            Δt_min=m.h_bounds[1],Δt_max=m.h_bounds[2],max_iter=m.max_iter,
            linear_solver="mumps",ipopt_options=m.options)
    else
        UnitarySmoothPulseProblem(input.system,m.G,m.K,m.h;init_trajectory=input.traj,
            free_time=false,a_bounds=m.bounds,dda_bound=m.acceleration,max_iter=m.max_iter,
            Q=m.Q,R=m.R,ipopt_options=m.options)
    end
end

function ap2_model(input)
    m=input.m;t=input.traj
    (;K=m.K,h=m.h,T=(m.K-1)*m.h,Q=m.Q,R=m.R,amplitude=m.bounds,
        acceleration=m.acceleration,free_time=m.free_time,h_bounds=m.h_bounds,
        H0=ap_complex(m.H0),Hs=ap_complex.(m.Hs),target=ap_complex(m.G),
        components=Dict(string(k)=>collect(v) for (k,v) in pairs(t.components)),
        propagation="pade22",loss="unsquared",regularizer="node_sum",
        state_layout="column_interleaved_real_imag",
        objective="Q*abs(1-abs(tr(G' U))/d)+.5*R*sum(u^2+du^2+ddu^2)",
        source="third_party/QuantumCollocation2023.jl/"*m.source,source_note=m.source_note,
        author_commit="d3b33912ebaf7b335f9e9a8619399cbc59ec89b2",
        initialization="Exact original template initialization, isolated before original NLP construction")
end

function ap2_initial_hash(s)
    Base.ENDIAN_BOM==0x04030201 || error("Initial hash requires little-endian Float64")
    values=vcat(s.states.values,s.u.values,s.du.values,s.ddu.values,s.dt)
    bytes2hex(sha256(reinterpret(UInt8,Float64.(values))))
end

function ap2_input_record(input,seed,role)
    initial=ap_slice((;trajectory=input.traj),input.traj.datavec)
    (;schema="author2023_shared_input_v2",task=input.m.task_id,seed,role,
        model=ap2_model(input),initial,initial_full=ap_arr(copy(input.traj.datavec)),
        generation_seconds=input.generation_seconds,geodesic_success=input.geodesic_success,
        initial_sha256=ap2_initial_hash(initial),
        initial_hash_encoding="Float64 little-endian concatenation: column-major states,u,du,ddu,then dt",
        generation_scope="seed + original physical system + author geodesic/random arrays + NamedTrajectory; excludes native objective, integrators, optimizer and all NLP assembly")
end

function ap2_validate_seeds(seeds,role)
    isempty(seeds) && error("Empty seed list")
    length(unique(seeds))==length(seeds) || error("Duplicate seeds")
    if role=="final20"
        length(seeds)==20 || error("Exactly twenty held-out seeds required")
        isempty(intersect(seeds,[29,42,1234,61001,61007,61027,61031,61043,61051])) || error("Development seed in final20")
    end
    nothing
end

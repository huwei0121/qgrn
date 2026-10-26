module PiccoloModels

using Piccolo
import DirectTrajOpt
using LinearAlgebra, Random, SHA

export task_ids, task_spec, build_task, build_leakage_stage, snapshot, metadata,
       solve_native!, json_safe, source_provenance

const PICCOLO_COMMIT = "5e8f209a4a2216a94e3cf1b755dd451ad3403d99"
const DTO_COMMIT = "99caf3519a13dea58d662eaa9b7c99187fa84e08"
const PROJECT_ROOT = @__DIR__

"""Task identifiers from the pinned Piccolo software tutorials."""
task_ids() = ["first_gate", "two_qubit_zoh", "multilevel_transmon"]

function task_spec(id::AbstractString)
    common = (Q=100.0, R=0.01, ddu_bound=1.0,
              source_kind="author_software_tutorial", solver="MadNLP",
              fixed_final_time=false, timestep_bounds=(0.0, Inf),
              callback=false, cache_replay=false)
    if id == "first_gate"
        return merge(common, (id=String(id), seed=42, N=100, initial_duration=10.0,
            max_iter=20, print_level=1, timesteps_all_equal=true,
            source="docs/literate/first_gate.jl", source_lines="28-125",
            target_metric="process_fidelity", leakage_constraint=false))
    elseif id == "two_qubit_zoh"
        return merge(common, (id=String(id), seed=1234, N=200, initial_duration=10.0,
            max_iter=150, print_level=nothing, timesteps_all_equal=true,
            source="docs/literate/two_qubit_gate_validation.jl", source_lines="24-149",
            target_metric="embedded_average_gate_fidelity", leakage_constraint=false))
    elseif id in ("multilevel_transmon", "multilevel_transmon_leakage")
        return merge(common, (id=String(id), seed=123, N=50, initial_duration=10.0,
            max_iter=50, print_level=nothing, timesteps_all_equal=false,
            source="docs/literate/multilevel_transmon.jl", source_lines="30-112",
            target_metric="embedded_average_gate_fidelity",
            leakage_constraint=(id == "multilevel_transmon_leakage")))
    end
    error("Unknown author task: $id")
end

struct AuthorTask{Q,S,G,P}
    id::String
    qcp::Q
    system::S
    goal::G
    spec::P
    seed::Int
    initialization::String
end

function build_task(id::AbstractString; seed::Union{Nothing,Int}=nothing)
    id == "multilevel_transmon_leakage" && error(
        "The author leakage stage starts from the solved multilevel qtraj. Use build_leakage_stage(parent).")
    spec = task_spec(id)
    actual_seed = isnothing(seed) ? spec.seed : seed
    Random.seed!(actual_seed)
    if id == "first_gate"
        sys = QuantumSystem(0.5 * PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]], [1.0, 1.0])
        goal = GATES[:X]
        controls = 0.1 * randn(2, spec.N)
    elseif id == "two_qubit_zoh"
        sys = MultiTransmonSystem([4.0, 4.1], [0.2, 0.2], [0.0 0.1; 0.1 0.0];
                                 levels_per_transmon=2, drive_bounds=0.1)
        goal = EmbeddedOperator(GATES[:CX], sys)
        controls = 0.02 * randn(sys.n_drives, spec.N)
    elseif id == "multilevel_transmon"
        sys = TransmonSystem(drive_bounds=[0.2, 0.2], levels=5, δ=0.2)
        goal = EmbeddedOperator(:X, sys)
        controls = 0.1 * randn(2, spec.N)
    else
        error("Unknown task: $id")
    end
    pulse = ZeroOrderPulse(controls, collect(range(0, spec.initial_duration; length=spec.N)))
    qtraj = UnitaryTrajectory(sys, pulse, goal)
    qcp = if spec.timesteps_all_equal
        SmoothPulseProblem(qtraj, spec.N; Q=spec.Q, R=spec.R,
            ddu_bound=spec.ddu_bound,
            piccolo_options=PiccoloOptions(timesteps_all_equal=true))
    else
        # Omit PiccoloOptions exactly as in the tutorial; independent free timesteps.
        SmoothPulseProblem(qtraj, spec.N; Q=spec.Q, R=spec.R, ddu_bound=spec.ddu_bound)
    end
    initialization = actual_seed == spec.seed ? "author_default_seed_and_native_construction" :
                                               "author_model_with_declared_seed_variant"
    return AuthorTask(String(id), qcp, sys, goal, spec, actual_seed, initialization)
end

"""Build the author's second stage from the SAME qtraj updated by stage-one solve!."""
function build_leakage_stage(parent::AuthorTask)
    parent.id == "multilevel_transmon" || error("Leakage parent must be multilevel_transmon")
    spec = task_spec("multilevel_transmon_leakage")
    qcp = SmoothPulseProblem(quantum_trajectory(parent.qcp), spec.N;
        Q=spec.Q, R=spec.R, ddu_bound=spec.ddu_bound,
        piccolo_options=PiccoloOptions(leakage_constraint=true,
            leakage_constraint_value=1e-2, leakage_cost=1e-2))
    return AuthorTask(spec.id, qcp, parent.system, parent.goal, spec, parent.seed,
                      "author_warm_start_from_multilevel_transmon_stage_one")
end

# JSON arrays are explicitly row-major nested arrays; datavec is Julia column-major.
matrix_rows(A::AbstractMatrix) = [collect(A[i, :]) for i in axes(A, 1)]
complex_record(A::AbstractMatrix) = Dict("re"=>matrix_rows(real.(A)),
    "im"=>matrix_rows(imag.(A)), "shape"=>collect(size(A)))

function json_safe(x)
    x === nothing && return nothing
    x isa Bool && return x
    x isa Enum && return string(x)
    x isa AbstractFloat && return isfinite(x) ? x : string(x)
    x isa Real && return x
    x isa Complex && return Dict("re"=>json_safe(real(x)), "im"=>json_safe(imag(x)))
    x isa Union{AbstractString,Symbol} && return string(x)
    x isa AbstractMatrix{<:Complex} && return json_safe(complex_record(x))
    x isa AbstractMatrix && return [json_safe(collect(x[i, :])) for i in axes(x, 1)]
    x isa Union{NamedTuple,AbstractDict} && return Dict(string(k)=>json_safe(v) for (k,v) in pairs(x))
    x isa Union{AbstractVector,Tuple,AbstractRange} && return [json_safe(v) for v in x]
    return Dict("julia_type"=>string(typeof(x)))
end

function struct_record(x)
    return Dict("julia_type"=>string(typeof(x)),
                "fields"=>Dict(string(k)=>json_safe(getfield(x,k)) for k in fieldnames(typeof(x))))
end

function objective_record(obj, traj)
    record = struct_record(obj)
    record["value_at_export"] = DirectTrajOpt.objective_value(obj,traj)
    if obj isa DirectTrajOpt.CompositeObjective
        record["terms"] = [objective_record(o,traj) for o in obj.objectives]
        record["weights"] = copy(obj.weights)
    end
    return record
end

function source_provenance()
    relpaths = ["third_party/Piccolo.jl/docs/literate/first_gate.jl",
        "third_party/Piccolo.jl/docs/literate/two_qubit_gate_validation.jl",
        "third_party/Piccolo.jl/docs/literate/multilevel_transmon.jl",
        "third_party/Piccolo.jl/src/control/templates/smooth_pulse_problem.jl",
        "third_party/Piccolo.jl/src/control/objectives.jl",
        "third_party/DirectTrajOpt.jl/src/solvers/madnlp_solver/options.jl",
        "third_party/DirectTrajOpt.jl/src/solvers/madnlp_solver/solver.jl",
        "piccolo_models.jl", "Manifest.toml"]
    files = Dict(rel=>bytes2hex(sha256(read(joinpath(PROJECT_ROOT, rel)))) for rel in relpaths)
    return Dict("piccolo_commit"=>PICCOLO_COMMIT, "directtrajopt_commit"=>DTO_COMMIT,
                "files_sha256"=>files)
end

function metadata(task::AuthorTask)
    traj = get_trajectory(task.qcp)
    prob = direct_problem(task.qcp)
    embedded = task.goal isa EmbeddedOperator
    goalmat = embedded ? task.goal.operator : task.goal
    subspace = embedded ? collect(task.goal.subspace) : collect(1:size(goalmat,1))
    leakage_indices = embedded ? get_iso_vec_leakage_indices(subspace, size(goalmat,1)) : Int[]
    return Dict("schema"=>"author_native_task_v1", "task"=>task.id,
        "spec"=>json_safe(task.spec), "actual_seed"=>task.seed,
        "initialization"=>task.initialization, "provenance"=>source_provenance(),
        "hamiltonian"=>Dict("drift"=>complex_record(Matrix(get_drift(task.system))),
            "drives"=>[complex_record(Matrix(H)) for H in get_drives(task.system)],
            "convention"=>"U[k+1] = exp(-im*dt[k]*(H0+sum(u[j,k]*Hj)))*U[k]"),
        "target"=>Dict("full"=>complex_record(Matrix(goalmat)), "subspace"=>subspace,
            "metric"=>task.spec.target_metric, "embedded"=>embedded),
        "components"=>json_safe(traj.components), "dimensions"=>json_safe(traj.dims),
        "N"=>traj.N, "node_dimension"=>traj.dim, "variable_count"=>length(traj.datavec),
        "state_name"=>string(state_name(task.qcp)), "drive_name"=>string(drive_name(task.qcp)),
        "bounds"=>json_safe(traj.bounds), "initial_constraints"=>json_safe(traj.initial),
        "final_constraints"=>json_safe(traj.final), "soft_goal"=>json_safe(traj.goal),
        "constraints"=>[struct_record(c) for c in prob.constraints],
        "integrators"=>[struct_record(i) for i in prob.integrators],
        "objective"=>objective_record(prob.objective,traj),
        "leakage_iso_indices"=>leakage_indices,
        "leakage_semantics"=>"if enabled: each selected real/imag coordinate squared <= 0.01; soft 0.01*mean(abs2(selected)) at every node",
        "solver_options"=>struct_record(DirectTrajOpt.MadNLPOptions()),
        "solver_kwargs"=>(isnothing(task.spec.print_level) ? Dict("max_iter"=>task.spec.max_iter) :
            Dict("max_iter"=>task.spec.max_iter,"print_level"=>task.spec.print_level)),
        "callback"=>nothing, "cached_solve_bypassed"=>true,
        "comparison_scope"=>"source nonlinear program with free duration and source solver defaults")
end

function _fidelity(task::AuthorTask, U)
    if task.goal isa EmbeddedOperator
        indices = task.goal.subspace
        W = unembed(task.goal)
        M = W' * U[indices, indices]
        n = length(indices)
        return real((abs(tr(M' * M)) + abs2(tr(M))) / (n*(n+1)))
    end
    return real(abs2(tr(task.goal' * U)) / size(U,1)^2)
end

function snapshot(task::AuthorTask)
    traj = get_trajectory(task.qcp)
    prob = direct_problem(task.qcp)
    sn, un = state_name(task.qcp), drive_name(task.qcp)
    dn, ddn = Symbol("d", un), Symbol("dd", un)
    states = [Matrix(iso_vec_to_operator(traj[sn][:,k])) for k in 1:traj.N]
    u, du, ddu = Matrix(traj[un]), Matrix(traj[dn]), Matrix(traj[ddn])
    dt, times = vec(copy(traj[traj.timestep])), vec(copy(traj[:t]))
    H0, Hs = Matrix(get_drift(task.system)), Matrix.(get_drives(task.system))
    d = size(H0,1)
    rolled = Matrix{ComplexF64}[Matrix{ComplexF64}(I,d,d)]
    local_defect = 0.0
    for k in 1:traj.N-1
        H = H0 + sum(u[j,k] * Hs[j] for j in eachindex(Hs))
        P = exp(-im*dt[k]*H)
        push!(rolled, P*rolled[end])
        local_defect = max(local_defect, maximum(abs, P*states[k]-states[k+1]))
    end
    bound_violation = 0.0
    for (name,(lo,hi)) in pairs(traj.bounds)
        X = traj[name]
        bound_violation = max(bound_violation, maximum(lo .- X), maximum(X .- hi))
    end
    boundary_violation = 0.0
    for (name,v) in pairs(traj.initial)
        boundary_violation = max(boundary_violation, maximum(abs, traj[name][:,1]-v))
    end
    for (name,v) in pairs(traj.final)
        boundary_violation = max(boundary_violation, maximum(abs, traj[name][:,end]-v))
    end
    derivative_defect = maximum(abs, diff(u;dims=2)-du[:,1:end-1].*permutedims(dt[1:end-1]))
    derivative_defect = max(derivative_defect,
        maximum(abs, diff(du;dims=2)-ddu[:,1:end-1].*permutedims(dt[1:end-1])))
    time_defect = maximum(abs, diff(times)-dt[1:end-1])
    equal_dt_defect = task.spec.timesteps_all_equal ? maximum(abs, dt .- dt[end]) : 0.0
    indices = task.goal isa EmbeddedOperator ?
        get_iso_vec_leakage_indices(task.goal.subspace,d) : Int[]
    leakage_coordinate_max = isempty(indices) ? 0.0 : maximum(abs2, traj[sn][indices,:])
    leakage_violation = task.spec.leakage_constraint ? max(0.0, leakage_coordinate_max-0.01) : 0.0
    feasible_defect = max(local_defect,bound_violation,boundary_violation,
                          derivative_defect,time_defect,equal_dt_defect,leakage_violation)
    return Dict("datavec"=>copy(traj.datavec), "datavec_order"=>"column-major node-major",
        "components"=>Dict(string(n)=>matrix_rows(traj[n]) for n in traj.names),
        "states"=>complex_record.(states), "dense_rollout_states"=>complex_record.(rolled),
        "controls"=>matrix_rows(u), "du"=>matrix_rows(du), "ddu"=>matrix_rows(ddu),
        "times"=>times, "dt"=>dt, "duration"=>sum(dt[1:end-1]),
        "native_objective"=>DirectTrajOpt.objective_value(prob.objective,traj),
        "native_stored_fidelity"=>_fidelity(task,states[end]),
        "physical_fidelity"=>_fidelity(task,rolled[end]),
        "physical_infidelity"=>abs(1-_fidelity(task,rolled[end])),
        "dense_vs_stored_final"=>norm(rolled[end]-states[end]),
        "max_stored_unitarity"=>maximum(norm(U'*U-I) for U in states),
        "max_dense_unitarity"=>maximum(norm(U'*U-I) for U in rolled),
        "quantum_dynamics_defect"=>local_defect, "control_derivative_defect"=>derivative_defect,
        "time_consistency_defect"=>time_defect,"equal_timestep_defect"=>equal_dt_defect,
        "box_violation"=>max(0.0,bound_violation),"boundary_violation"=>boundary_violation,
        "leakage_coordinate_max_abs2"=>leakage_coordinate_max,
        "leakage_constraint_violation"=>leakage_violation,"max_primal_defect"=>feasible_defect)
end

"""No callback, no backend options override, no cache. This is the tutorial's effective solve!."""
function solve_native!(task::AuthorTask)
    if isnothing(task.spec.print_level)
        return Piccolo.solve!(task.qcp; max_iter=task.spec.max_iter)
    end
    return Piccolo.solve!(task.qcp; max_iter=task.spec.max_iter, print_level=task.spec.print_level)
end

end # module

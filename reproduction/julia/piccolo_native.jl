# Complete modern author tasks with measured primal histories and time budgets.
# This entry adds instrumentation and experiment stopping only; author sources
# and all objective, dynamics, bounds and derivative implementations are reused.
using LinearAlgebra, SHA, JSON3
using Piccolo
import DirectTrajOpt
import NamedTrajectories
include("piccolo_adapter.jl")
const B = AuthorGRNBenchmarks
const AN = B.AN
const NT = B.NativeIterationTiming
const MN = NT.MadNLP
const EXT = DirectTrajOpt.MadNLPSolverExt
const MOI = EXT.MOI
const TASKS = ("first_gate", "two_qubit_zoh")
clock() = time_ns()/1e9

function save_record(path, value)
    mkpath(dirname(path))
    ispath(path) && error("Existing evidence: $path")
    started=clock()
    open(path,"w") do io
        JSON3.write(io,B.clean(value))
    end
    return clock()-started
end

function input_record(task, build_seconds)
    traj=get_trajectory(task.qcp)
    values=Vector{Float64}(vec(traj))
    metadata=AN.metadata(task)
    (;schema="modern_piccolo_shared_input_v1",task=task.id,seed=task.seed,
      initial=B.trajectory_record(task),initial_primal=values,
      initial_sha256=bytes2hex(sha256(reinterpret(UInt8,values))),
      initial_coordinate_sha256=bytes2hex(sha256(reinterpret(UInt8,values))),
      model=B.model_record(task,metadata),native_initial=AN.snapshot(task),
      generation_seconds=build_seconds)
end

mutable struct FullTrace <: MN.AbstractUserCallback
    base::NT.NativeTrace
    primals::Vector{Vector{Float64}}
    deadline::Float64
    snapshot_seconds::Float64
end

function (trace::FullTrace)(solver, mode::MN.AbstractUserCallbackStatus)
    NT_before=clock()
    trace.base(solver,mode)
    cb=MN.get_cb(solver)
    x=MN.variable(MN.get_x(solver))
    handler=cb.fixed_handler
    # Reinsert parameters without modifying MadNLP's cached full vector or hash.
    full = if handler isa MN.MakeParameter
        data=copy(handler.x_full)
        data[handler.free] .= x
        data
    else
        copy(x)
    end
    push!(trace.primals,Vector{Float64}(full))
    trace.snapshot_seconds += clock()-NT_before
    return clock() < trace.deadline
end

function native_one(id, seed, seconds, input_path; maxiter=100000000)
    sources=copy(AN.source_provenance()["files_sha256"])
    project=@__DIR__
    for rel in ("piccolo_native.jl",
                "piccolo_adapter.jl","native_iteration_timing.jl")
        sources[rel]=bytes2hex(sha256(read(joinpath(project,rel))))
    end
    started=clock()
    task=AN.build_task(id;seed)
    build_seconds=clock()-started
    prepared=JSON3.read(read(input_path,String),Dict{String,Any})
    trj=get_trajectory(task.qcp)
    rebuilt=Vector{Float64}(vec(trj))
    signature=bytes2hex(sha256(reinterpret(UInt8,rebuilt)))
    signature==prepared["initial_sha256"] || error("Author initialization changed")
    # Reinsert the exact exported coordinates; no initial feasibility repair.
    NamedTrajectories.update!(trj,Float64.(prepared["initial_primal"]);type=:both)
    trace=FullTrace(NT.NativeTrace(started),Vector{Vector{Float64}}(),started+seconds,0.)
    options=DirectTrajOpt.MadNLPOptions(max_iter=maxiter,print_level=6,
        intermediate_callback=trace,
        barrier=MN.QualityFunctionUpdate(1e-8,10.))
    prob=direct_problem(task.qcp)
    optimizer,variables=EXT.get_optimizer_and_variables(prob,options,nothing;verbose=false)
    # Keep the author barrier floor and native linear refinement accuracy when
    # disabling success tolerances. Otherwise tol also changes numerical work.
    optimizer.options[:tol]=1e-300
    optimizer.options[:acceptable_tol]=0.
    optimizer.options[:acceptable_iter]=typemax(Int)
    optimizer.options[:richardson_tol]=1e-10
    optimizer.options[:richardson_acceptable_tol]=1e-5
    optimizer.options[:max_wall_time]=max(seconds-(clock()-started),1e-9)
    setup_seconds=clock()-started
    solve_started=clock()
    MOI.optimize!(optimizer)
    solver_seconds=clock()-solve_started
    EXT.update_trajectory!(prob,optimizer,variables)
    sync_started=clock()
    Piccolo.sync_trajectory!(task.qcp;check_divergence=false)
    sync_seconds=clock()-sync_started
    final=B.trajectory_record(task)
    verification_started=clock()
    physical=AN.snapshot(task)
    verification_seconds=clock()-verification_started
    elapsed=clock()-started
    history=[merge(event,(;primal=trace.primals[i])) for (i,event) in enumerate(trace.base.events)]
    raw_status=string(MOI.get(optimizer,MOI.RawStatusString()))
    status=string(MOI.get(optimizer,MOI.TerminationStatus()))
    (;schema="modern_piccolo_probe_native_v1",task=id,seed,method="author_madnlp",
      budget_seconds=seconds,status,raw_status,stop_on_target=false,
      stopping_policy="user_requested_time_budget_only",time_budget_only=true,
      precision_stopping=false,target_early_stopping=false,
      initial_coordinate_sha256=signature,initial_sha256=signature,
      initial_slice=prepared["initial"],initial_primal=prepared["initial_primal"],
      model=prepared["model"],initial=prepared["native_initial"],
      history,final_slice=final,physical,
      returned=(;final_objective=physical["native_objective"],
                  feasibility=physical["max_primal_defect"],
                  physical_gate_error=physical["physical_infidelity"]),
      final_objective=physical["native_objective"],
      return_valid=isfinite(physical["native_objective"]),source_hashes=sources,
      source_unchanged=all(bytes2hex(sha256(read(joinpath(project,rel))))==digest for (rel,digest) in sources),
      failure=false,build_seconds,setup_seconds,solver_seconds,sync_seconds,
      verification_seconds,complete_seconds=elapsed,solve_return_seconds=setup_seconds+solver_seconds,
      shared_input_generation_seconds=prepared["generation_seconds"],
      over_budget_seconds=max(0.,elapsed-seconds),
      observer_seconds=trace.snapshot_seconds,
      native_timing=merge(NT.timing_summary(trace.base),(;
          mechanism="raw MadNLP callback; full primal observation and experiment time-budget termination",
          callback_semantics="record full primal and cached objective; return false at the complete-clock deadline",
          solver_controls_modified=true,
          control_changes="experiment termination settings only; original mathematical model and native derivative/linear solver retained")),
      native_options=(;tol=1e-300,acceptable_tol=0.,acceptable_iter=typemax(Int),
          max_iter=maxiter,max_wall_time=optimizer.options[:max_wall_time],print_level=6,
          barrier="Author QualityFunctionUpdate(1e-8,10.0)",
          richardson_tol=1e-10,richardson_acceptable_tol=1e-5,
          fixed_variable_treatment="author default MakeParameter",
          hessian="author exact",linear_solver="author MUMPS"),
      history_semantics="Full original primal including fixed variables; objective/time are raw MadNLP cached events; independently audit each saved primal offline",
      timing_note="Complete time includes author build, native setup, solver, observation, synchronization and final verification; export/import startup, warmup, and saving separate")
end

function main()
    length(ARGS)==5 || error("TASK SEED INPUT OUTPUT SECONDS")
    task,seed,input,output,seconds=ARGS
    warm_started=clock()
    warm=native_one(task,parse(Int,seed),120.,input;maxiter=1)
    warm_seconds=clock()-warm_started
    result=native_one(task,parse(Int,seed),parse(Float64,seconds),input;maxiter=100000000)
    save_record(output,merge(result,(;warm_seconds,warm_maxiter=1,warm_status=warm.status)))
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()

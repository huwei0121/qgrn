module NativeIterationTiming

import DirectTrajOpt
const MadNLP = DirectTrajOpt.MadNLPSolverExt.MadNLP

export NativeTrace, timing_summary

const TraceEvent = NamedTuple{
    (:event_index, :iteration, :elapsed, :objective, :objective_scaled,
     :objective_scale, :mode),
    Tuple{Int,Int,Float64,Float64,Float64,Float64,String},
}

"""
    NativeTrace(started::Real; capacity=1024)

MadNLP iteration observer for the source `Piccolo.solve!` path.
`started` is the caller's complete-cost origin in `time_ns()/1e9` seconds.
Pass the object as `intermediate_callback=trace` and preserve all other
solver options. The callback stores cached objective scalars and a monotonic
timestamp in memory, then returns `true` to continue the solve.

Subtyping `MadNLP.AbstractUserCallback` preserves the solver's default
fixed-variable treatment. DTO's `AbstractIntermediateCallback` adapter sets
that treatment to `RelaxBound`.

Events include the initial regular loop (iteration zero), regular iterations
before the solver's own termination checks, and native restoration callbacks.
Restoration modes remain explicit, and duplicate iteration numbers are kept.
`objective` has exactly MadNLP's printed objective semantics: the already
cached native objective divided by its scaling factor. Feasible trajectory
quality is evaluated separately by the benchmark analysis. The benchmark runner
also records the final return. Logging time is included in the caller's clock.
"""
mutable struct NativeTrace <: MadNLP.AbstractUserCallback
    started::Float64
    events::Vector{TraceEvent}
    callback_nanoseconds::UInt64
end

function NativeTrace(started::Real; capacity::Integer=1024)
    isfinite(started) || throw(ArgumentError("The timing origin must be finite"))
    capacity >= 0 || throw(ArgumentError("Trace capacity must be nonnegative"))
    events = TraceEvent[]
    sizehint!(events, capacity)
    return NativeTrace(Float64(started), events, UInt64(0))
end

mode_name(::MadNLP.UserCallbackRegular) = "regular"
mode_name(::MadNLP.UserCallbackRestore) = "restore"
mode_name(::MadNLP.UserCallbackRobust) = "robust"

function (trace::NativeTrace)(solver, mode::MadNLP.AbstractUserCallbackStatus)
    observed = time_ns()
    scale = Float64(MadNLP.get_cb(solver).obj_scale[])
    scaled = Float64(MadNLP.get_obj_val(solver))
    push!(trace.events, (
        event_index=length(trace.events)+1,
        iteration=Int(MadNLP.get_cnt(solver).k),
        elapsed=Float64(observed)/1e9-trace.started,
        objective=scaled/scale,
        objective_scaled=scaled,
        objective_scale=scale,
        mode=mode_name(mode),
    ))
    trace.callback_nanoseconds += time_ns()-observed
    return true
end

function timing_summary(trace::NativeTrace)
    modes = Dict(mode=>count(e->e.mode==mode, trace.events)
        for mode in ("regular", "restore", "robust"))
    return (
        mechanism="raw MadNLP.AbstractUserCallback; always continue",
        clock="time_ns monotonic wall clock; shared complete-cost origin",
        objective_semantics="cached native objective divided by obj_scale; same as MadNLP print_iter",
        event_count=length(trace.events), mode_counts=modes,
        callback_seconds=Float64(trace.callback_nanoseconds)/1e9,
        callback_cost_included=true,
        extra_objective_evaluations=0, extra_derivative_evaluations=0,
        solver_controls_modified=false,
        all_elapsed_finite=all(e->isfinite(e.elapsed), trace.events),
        elapsed_monotonic=issorted(getproperty.(trace.events, :elapsed)),
        contains_initial_iteration=any(e->e.mode=="regular" && e.iteration==0, trace.events),
        final_return_is_separate=true,
    )
end

end

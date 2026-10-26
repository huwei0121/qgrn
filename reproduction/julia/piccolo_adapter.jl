module AuthorGRNBenchmarks

using LinearAlgebra, SHA, JSON3
using Piccolo
import NamedTrajectories
include("piccolo_models.jl")
include("native_iteration_timing.jl")
const AN = PiccoloModels

array_record(x) = (; shape=collect(size(x)), values=vec(x), order="F")
complex_record(x) = (; shape=collect(size(x)), real=vec(real.(x)), imag=vec(imag.(x)), order="F")
clean(x) = x
clean(x::AbstractFloat) = isfinite(x) ? x : nothing
clean(x::AbstractArray) = map(clean, x)
clean(x::Tuple) = map(clean, x)
clean(x::NamedTuple) = Dict(string(k)=>clean(v) for (k,v) in pairs(x))
clean(x::AbstractDict) = Dict(string(k)=>clean(v) for (k,v) in pairs(x))
function save_new(path, data)
    ispath(path) && error("Refusing to overwrite evidence: $path")
    write(path, JSON3.write(clean(data)))
end

function trajectory_record(task)
    trj=get_trajectory(task.qcp); un=drive_name(task.qcp); sn=state_name(task.qcp)
    (; u=array_record(copy(trj[un])), du=array_record(copy(trj[Symbol("d",un)])),
       ddu=array_record(copy(trj[Symbol("dd",un)])), states=array_record(copy(trj[sn])),
       dt=vec(copy(trj[trj.timestep])), times=vec(copy(trj[:t])))
end

function model_record(task, metadata)
    s=task.spec; trj=get_trajectory(task.qcp)
    H0=Matrix(get_drift(task.system)); Hs=Matrix.(get_drives(task.system))
    G=task.goal isa EmbeddedOperator ? Matrix(task.goal.operator) : Matrix(task.goal)
    (; K=s.N, h=trj[trj.timestep][1,1], T=s.initial_duration, Q=s.Q, R=s.R,
       amplitude=trj.bounds[drive_name(task.qcp)][2], acceleration=s.ddu_bound,
       free_time=true, h_bounds=(0.,Inf), H0=complex_record(H0),
       Hs=complex_record.(Hs), target=complex_record(G),
       loss=task.goal isa EmbeddedOperator ? "average" : "process",
       regularizer="time_weighted", propagation="exact",
       state_layout="column_interleaved_real_imag", source=s.source,
       source_kind=s.source_kind, initialization=task.initialization,
       native_metadata=metadata)
end


end

include("pico_native.jl")
length(ARGS)==3 || error("INPUT OUTPUT SECONDS")
input,output,seconds=ARGS
ispath(output) && error("Existing output")
warm_started=time_ns()/1e9
warm=sw_run(input,120.;maxiter=1)
warm_seconds=time_ns()/1e9-warm_started
result=sw_run(input,parse(Float64,seconds);maxiter=100000)
ap_save(output,merge(result,(;warm_seconds,warm_maxiter=1,warm_status=warm.status)))

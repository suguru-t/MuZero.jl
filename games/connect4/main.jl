
# Required packages
using Distributed
using Dates
using Flux
using ParameterSchedulers
using Distributions

try
	import CUDA
	import cuDNN
catch err
	@warn "CUDA/cuDNN are unavailable; running without explicit GPU setup." exception = err
end

# --- Configuration for Workers ---
const REQUIRED_SELF_PLAYERS = 1
const REQUIRED_LEARNERS = 1
const TARGET_DEDICATED_WORKERS = REQUIRED_SELF_PLAYERS + REQUIRED_LEARNERS

# --- Worker Setup ---
current_procs = nprocs()
required_total_procs = 1 + TARGET_DEDICATED_WORKERS

if current_procs < required_total_procs
	missing = required_total_procs - current_procs
	# println("Warning: Detected $(current_procs) processes. Need $required_total_procs (1 Master + $TARGET_DEDICATED_WORKERS Workers).")
	# println("   Adding $missing worker(s)...")
	addprocs(missing, exeflags = "--project")
end

worker_ids = workers()
dedicated_ids = filter(x -> x != 1, worker_ids)

for pid in worker_ids
    remotecall_fetch(() -> begin
		Base.eval(Main, :(using Flux))
		try
			Base.eval(Main, :(import CUDA))
		catch
		end
	end, pid)
end

# println("Dedicated PIDs: $dedicated_ids")

if length(dedicated_ids) < TARGET_DEDICATED_WORKERS
	println("ERROR: Failed to acquire enough dedicated workers.")
	exit(1)
end

@everywhere begin

    using Logging
    _old_logger = global_logger(NullLogger())
    Base.eval(Main, :(using Flux))
	try
		Base.eval(Main, :(import CUDA))
	catch
	end
    global_logger(_old_logger)
	using Distributed
	using ParameterSchedulers

	const SRC_DIR = joinpath(@__DIR__, "../../src")

	include(joinpath(SRC_DIR, "Constructors.jl"))
	include(joinpath(SRC_DIR, "SelfPlay.jl"))
	include(joinpath(SRC_DIR, "ReplayBuffer.jl"))
	include(joinpath(SRC_DIR, "Learning.jl"))
	include("game.jl")
end

include("params.jl")

env = Connect4()

training_step = RemoteChannel(() -> Channel{Int}(1))
num_played_games = RemoteChannel(() -> Channel{Int}(1))
num_played_steps = RemoteChannel(() -> Channel{Int}(1))
num_reanalysed_games = RemoteChannel(() -> Channel{Int}(1))
total_samples = RemoteChannel(() -> Channel{Int}(1))

remote_NNs = RemoteChannel(() -> Channel{NamedTuple{(:representation, :prediction, :dynamics), Tuple{Any, Any, Any}}}(1))

# NEW: The Game Queue (Replaces RemoteBufferChannel)
game_queue = RemoteChannel(() -> Channel{GameHistory}(200))

# println("Initializing networks...")
rep = init_representation(hyper, conf)
pred = init_prediction(hyper, conf)
dyn = init_dynamics(hyper, conf)

function count_params(model)
	p, _ = Flux.destructure(model)
	return length(p)
end

function print_structure(model, indent = 0, prefix = "")
	sp = " " ^ indent
	if model isa Chain
		println("$(sp)$(prefix)Chain")
		for (i, layer) in enumerate(model.layers)
			print_structure(layer, indent + 3, "[$i] ")
		end
		return
	end
	if model isa Flux.Parallel
        println("$(sp)$(prefix)Parallel Head")
        for (i, path) in enumerate(model.layers)
            print_structure(path, indent + 3, "Path $i: ")
        end
        return
    end
	if model isa Dense
		w = size(model.weight)
		activation = getproperty(model, Symbol("\u03c3"))
		println("$(sp)$(prefix)Dense($(w[2]) -> $(w[1])) | activation: $(activation)")
		return
	end
	println("$(sp)$(prefix)$(typeof(model))")
end

# println("\n" * "="^60)
# println("Connect 4 Network Architecture")
# println("="^60)
# println("\nRepresentation Network:"); print_structure(rep); println("   Params: $(count_params(rep))")
# println("\nPrediction Network:"); print_structure(pred); println("   Params: $(count_params(pred))")
# println("\nDynamics Network:"); print_structure(dyn); println("   Params: $(count_params(dyn))")
# println("="^60 * "\n")

put!(remote_NNs, (representation = rep, prediction = pred, dynamics = dyn))

put!(training_step, 0)
put!(num_played_games, 0)
put!(num_played_steps, 0)
put!(num_reanalysed_games, 0)
put!(total_samples, 0)

learner_pid = pop!(dedicated_ids)
self_play_pids = dedicated_ids

# println("Assignments:")
# println("   Learner PID:   $learner_pid")
# println("   Self-Play PIDs: $self_play_pids")

self_play_jobs = NamedTuple[]

# println("Starting Self-Play...")
for pid in self_play_pids
	future = @spawnat pid begin
		self_play!(env,
			training_step,
			remote_NNs,
			game_queue,
			conf)
	end
	push!(self_play_jobs, (pid = pid, future = future))
end

# println("Starting Learner...")
learn = @spawnat learner_pid learning!(
	training_step,
	remote_NNs,
	game_queue,
	conf,
	hyper)

function current_training_step_text(training_step)
	try
		return string(fetch(training_step))
	catch
		return "unknown"
	end
end

function wait_for_training!(learn, self_play_jobs, training_step, conf)
	completed_self_play_workers = Set{Int}()
	while true
		if isready(learn)
			fetch(learn)
			return true
		end

		for job in self_play_jobs
			job.pid in completed_self_play_workers && continue

			if isready(job.future)
				try
					fetch(job.future)
				catch err
					step_text = current_training_step_text(training_step)
					details = sprint(showerror, err, catch_backtrace())
					error("Self-play worker $(job.pid) failed while learner was active at training step $step_text:\n$details")
				end
				step_text = current_training_step_text(training_step)
				if tryparse(Int, step_text) !== nothing && parse(Int, step_text) > conf.training_steps
					push!(completed_self_play_workers, job.pid)
				else
					error("Self-play worker $(job.pid) exited before learner completed at training step $step_text")
				end
			end
		end

		sleep(1.0)
	end
end

try
	wait_for_training!(learn, self_play_jobs, training_step, conf)
catch e
	if e isa InterruptException
		println("\nTraining stopped by user.")
	else
		println("Learner process failed: $e")

		# Save error to log file
		try
			log_path = joinpath(conf.results_path, "learner_error.log")
			open(log_path, "w") do io
				println(io, "\n" * "="^60)
			println(io, "TIMESTAMP: $(now())")
			println(io, "ERROR TYPE: $(typeof(e))")
			println(io, "-"^30)
			showerror(io, e, catch_backtrace())
			println(io, "\n" * "="^60)
		end
			println("Error details saved to: $log_path")
		catch log_err
			println("Failed to write error log: $log_err")
		end
	end
finally
	# println("Cleaning up workers...")
	rmprocs(workers())
end

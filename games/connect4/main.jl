
# 必要なパッケージをまとめてusing
using Distributed
using Dates
using Flux
using ParameterSchedulers
using CUDA
using cuDNN
using Distributions

# --- Configuration for Workers ---
const REQUIRED_SELF_PLAYERS = 1
const REQUIRED_LEARNERS = 1
const TARGET_DEDICATED_WORKERS = REQUIRED_SELF_PLAYERS + REQUIRED_LEARNERS

# --- Worker Setup ---
current_procs = nprocs()
required_total_procs = 1 + TARGET_DEDICATED_WORKERS

if current_procs < required_total_procs
	missing = required_total_procs - current_procs
	# println("⚠️  Detected $(current_procs) processes. Need $required_total_procs (1 Master + $TARGET_DEDICATED_WORKERS Workers).")
	# println("   Adding $missing worker(s)...")
	addprocs(missing, exeflags = "--project")
end

worker_ids = workers()
dedicated_ids = filter(x -> x != 1, worker_ids)

# ====== 以下の3行をここに追加！ ======
for pid in worker_ids
    remotecall_fetch(() -> Base.eval(Main, :(using Flux, CUDA)), pid) # これで全てのワーカーでFluxとCUDAが利用可能になります
end
# ====================================

# println("✅ Dedicated PIDs: $dedicated_ids")

if length(dedicated_ids) < TARGET_DEDICATED_WORKERS
	println("❌ ERROR: Failed to acquire enough dedicated workers.")
	exit(1)
end

# ===== ここにこの1行を追加！ =====
using Flux, CUDA, cuDNN
# =================================

@everywhere begin

	# --- ここから追加 ---
    using Logging 	  # ロギングのために必要
    _old_logger = global_logger(NullLogger()) # ログ出力を完全にミュート
    Base.eval(Main, :(using Flux, CUDA))      # ミュート状態でFluxを読み込む
    global_logger(_old_logger)                # ミュートを解除して元に戻す
    # --- ここまで ---
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

# println("🧠 Initializing Networks...")
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
        for (i, path) in enumerate(model.layers) # paths を layers に変更
            print_structure(path, indent + 3, "Path $i: ")
        end
        return
    end
	if model isa Dense
		w = size(model.weight)
		println("$(sp)$(prefix)Dense($(w[2]) ➡️  $(w[1])) | σ: $(model.σ)")
		return
	end
	println("$(sp)$(prefix)$(typeof(model))")
end

# println("\n" * "="^60)
# println("🏗️  Connect 4 Network Architecture")
# println("="^60)
# println("\n🔹 Representation Network:"); print_structure(rep); println("   ↳ Params: $(count_params(rep))")
# println("\n🔹 Prediction Network:"); print_structure(pred); println("   ↳ Params: $(count_params(pred))")
# println("\n🔹 Dynamics Network:"); print_structure(dyn); println("   ↳ Params: $(count_params(dyn))")
# println("="^60 * "\n")

put!(remote_NNs, (representation = rep, prediction = pred, dynamics = dyn))

put!(training_step, 0)
put!(num_played_games, 0)
put!(num_played_steps, 0)
put!(num_reanalysed_games, 0)
put!(total_samples, 0)

learner_pid = pop!(dedicated_ids)
self_play_pids = dedicated_ids

# println("📋 Assignments:")
# println("   Learner PID:   $learner_pid")
# println("   Self-Play PIDs: $self_play_pids")

# println("🚀 Starting Self-Play...")
for pid in self_play_pids
	@spawnat pid begin
		try
			# Updated signature: No counters, use game_queue
			self_play!(env,
				training_step,
				remote_NNs,
				game_queue,
				conf)
		catch e
			println("❌ Worker $pid failed: $e")
		end
	end
end

# println("📚 Starting Learner...")
learn = @spawnat learner_pid learning!(
	training_step,
	remote_NNs,
	game_queue,
	conf,
	hyper)

try
	wait(learn)
catch e
	if e isa InterruptException
		println("\n🛑 Training stopped by user.")
	else
		println("❌ Learner process failed: $e")

		# Save error to log file
		try
			log_path = joinpath(conf.results_path, "learner_error.log")
			open(log_path, "w") do io
				println(io, "\n" * "="^60)
				println(io, "TIMESTAMP: $(now())")
				println(io, "ERROR TYPE: $(typeof(e))")
				println(io, "-"^30)
				showerror(io, e)
				println(io, "\n" * "="^60)
			end
			println("📝 Error details saved to: $log_path")
		catch log_err
			println("❌ Failed to write error log: $log_err")
		end
	end
finally
	# println("🧹 Cleaning up workers...")
	rmprocs(workers())
end

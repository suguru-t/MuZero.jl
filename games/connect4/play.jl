using Distributed
using Serialization
using Flux
using ReinforcementLearningBase
using JLD2

const SRC_DIR = joinpath(@__DIR__, "../../src")

include(joinpath(SRC_DIR, "Constructors.jl"))
include(joinpath(SRC_DIR, "SelfPlay.jl"))
include(joinpath(SRC_DIR, "ReplayBuffer.jl"))
include(joinpath(SRC_DIR, "Learning.jl"))
include("game.jl")
include("params.jl")

function parse_args(args)
	options = Dict(
		"step" => "latest",
		"human-player" => "1",
		"num-iters" => string(conf.num_iters),
	)

	i = 1
	while i <= length(args)
		arg = args[i]
		if startswith(arg, "--")
			key = arg[3:end]
			if key == "help"
				options["help"] = "true"
				i += 1
			else
				i == length(args) && error("Missing value for --$key")
				options[key] = args[i + 1]
				i += 2
			end
		else
			error("Unexpected argument: $arg")
		end
	end
	return options
end

function print_help()
	println("""
	Usage:
	  julia --project games/connect4/play.jl [options]

	Options:
	  --step latest|N          Checkpoint to load. Default: latest
	  --human-player 1|2      Human player side. Default: 1
	  --num-iters N           MCTS simulations per MuZero move. Default: conf.num_iters
	""")
end

function load_networks(conf, step_arg::String)
	filename = step_arg == "latest" || step_arg == "0" ? "latest_checkpoint.jld2" : "$(parse(Int, step_arg))_checkpoint.jld2"
	path = joinpath(conf.networks_path, filename)

	isfile(path) || error("Checkpoint not found: $path")
	println("Loading networks from: $path")

	model_data = load(path)
	return set_inference_mode!((
		representation = model_data["representation"],
		prediction = model_data["prediction"],
		dynamics = model_data["dynamics"],
	))
end

function main()
	options = parse_args(ARGS)
	if get(options, "help", "false") == "true"
		print_help()
		return
	end

	human_player = parse(Int, options["human-player"])
	human_player in (1, 2) || error("--human-player must be 1 or 2")
	muzero_player = human_player == 1 ? 2 : 1
	num_iters = parse(Int, options["num-iters"])
	num_iters > 0 || error("--num-iters must be positive")

	play_conf = Config(conf;
		muzero_player = muzero_player,
		opponent = "human",
		num_iters = num_iters,
		allow_nonfinite_mcts = false,
	)

	NNs = load_networks(play_conf, options["step"])
	env = Connect4()

	println("\n" * "="^40)
	println(" Connect 4 MuZero Agent")
	println("="^40)
	println("You are Player $human_player.")
	println("MuZero is Player $muzero_player.")
	println("Enter column (1-7) to play.")
	println("="^40 * "\n")

	competitive_play!(env, NNs, play_conf; buffer_to_disk = false)
end

main()

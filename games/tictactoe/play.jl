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

LOAD_STEP = 0

function load_networks(conf, step)
	filename = step == 0 ? "latest_checkpoint.jld2" : "$(step)_checkpoint.jld2"
	path = joinpath(conf.networks_path, filename)

	if !isfile(path)
		# println("Warning: Checkpoint not found at: $path")
		# println("   Initializing fresh random networks instead.")
		return (
			representation = init_representation(hyper, conf),
			prediction = init_prediction(hyper, conf),
			dynamics = init_dynamics(hyper, conf),
		)
	end

	# println("Loading networks from: $path")
	model_data = load(path)
	return (
		representation = model_data["representation"],
		prediction = model_data["prediction"],
		dynamics = model_data["dynamics"],
	)
end

NNs = load_networks(conf, LOAD_STEP)
env = TicTacToe()

println("\n" * "="^40)
println(" Tic-Tac-Toe MuZero Agent")
println("="^40)
println("You are playing against the MuZero agent.")
println("Enter a number (1-9) to place your mark.")
println("The agent searches $(conf.num_iters) moves ahead.")
println("="^40 * "\n")

competitive_play!(env, NNs, conf; buffer_to_disk = false)

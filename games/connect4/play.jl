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
		# println("⚠️  Checkpoint not found at: $path")
		# println("   Initializing fresh random networks.")
		return (
			representation = init_representation(hyper, conf),
			prediction = init_prediction(hyper, conf),
			dynamics = init_dynamics(hyper, conf),
		)
	end

	# println("✅ Loading networks from: $path")
	model_data = load(path)
	return (
		representation = model_data["representation"],
		prediction = model_data["prediction"],
		dynamics = model_data["dynamics"],
	)
end

NNs = load_networks(conf, LOAD_STEP)
env = Connect4()

println("\n" * "="^40)
println(" 🔴 Connect 4 MuZero Agent")
println("="^40)
println("You are Player 1.")
println("Enter column (1-7) to play.")
println("="^40 * "\n")

competitive_play!(env, NNs, conf; buffer_to_disk = false)

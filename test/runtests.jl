using Test
using Flux
using Distributed
using ReinforcementLearningBase

# Local includes
const SRC_DIR = joinpath(@__DIR__, "../src")
const GAME_DIR = joinpath(@__DIR__, "../games/tictactoe")

# Include order matters
include(joinpath(SRC_DIR, "Constructors.jl"))
include(joinpath(SRC_DIR, "SelfPlay.jl"))
include(joinpath(SRC_DIR, "ReplayBuffer.jl"))
include(joinpath(SRC_DIR, "Learning.jl"))
include(joinpath(GAME_DIR, "game.jl"))
include(joinpath(GAME_DIR, "params.jl"))

@testset "MuZero Integration Tests" begin

	@testset "Initialization" begin
		# Test Network Initialization
		rep = init_representation(hyper, conf)
		pred = init_prediction(hyper, conf)
		dyn = init_dynamics(hyper, conf)

		@test rep isa Chain
		@test pred isa Chain
		@test dyn isa Chain

		# Test Environment
		env = TicTacToe()
		obs = ReinforcementLearningBase.reset!(env)
		@test size(obs) == (3, 3, 3)
	end

	@testset "Game Queue & Local Buffer" begin
		# Create a local buffer (Learner side)
		local_buffer = Dict{Int, GameHistory}()

		# Create a dummy game history
		history = GameHistory(
			observation_history = rand(Float32, 3, 3, 3, 10),
			action_history = collect(1:10),
			reward_history = zeros(Float32, 10),
			to_play_history = fill(1, 10),
			child_visits = rand(Float32, 9, 10),
			root_values = rand(Float32, 10),
			reanalysed_predicted_root_values = nothing,
			priorities = nothing,
			game_priority = nothing,
		)

		# Test insertion logic
		insert_game!(local_buffer, history, 1, conf)

		@test length(local_buffer) == 1
		@test haskey(local_buffer, 1)

		# Test Queue mechanics (Simulating SelfPlay pushing to Learner)
		game_queue = Channel{GameHistory}(10)
		put!(game_queue, history)
		@test isready(game_queue)
		taken_hist = take!(game_queue)
		@test taken_hist === history
	end

	@testset "Gradient Step (Smoke Test)" begin
		# Setup Data
		local_buffer = Dict{Int, GameHistory}()

		# Lower batch size for testing
		test_conf = Config(conf; batch_size = 4, training_steps = 5, num_unroll_steps = 2)

		# Fill buffer
		for i in 1:10
			history = GameHistory(
				observation_history = rand(Float32, 3, 3, 3, 10),
				action_history = collect(1:10),
				reward_history = zeros(Float32, 10),
				to_play_history = fill(1, 10),
				child_visits = rand(Float32, 9, 10),
				root_values = rand(Float32, 10),
				reanalysed_predicted_root_values = nothing,
				priorities = nothing,
				game_priority = nothing,
			)
			insert_game!(local_buffer, history, i, test_conf)
		end

		# Setup Networks
		rep = init_representation(hyper, test_conf)
		pred = init_prediction(hyper, test_conf)
		dyn = init_dynamics(hyper, test_conf)

		# Get Batch
		next_batch = get_batch(local_buffer, test_conf)
		_, batch = next_batch
		obs_b, act_b, t_v, t_r, t_p, w_b, g_scale = batch

		# Adjust dims
		g_scale = permutedims(g_scale)
		test_conf.PER ? w_b = permutedims(w_b) : nothing

		# Calc Gradients
		grads = Flux.gradient(rep, pred, dyn) do r, p, d
			compute_total_loss(r, p, d, obs_b, act_b, t_v, t_r, t_p, w_b, g_scale, test_conf)
		end

		@test !isnothing(grads)
		@test !isnothing(grads[1])

		println("Gradient calculation successful.")
	end
end

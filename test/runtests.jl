using Test
using Flux
using Distributed
using ReinforcementLearningBase
using JLD2

const SRC_DIR = joinpath(@__DIR__, "../src")
const GAME_DIR = joinpath(@__DIR__, "../games/tictactoe")

include(joinpath(SRC_DIR, "Constructors.jl"))
include(joinpath(SRC_DIR, "SelfPlay.jl"))
include(joinpath(SRC_DIR, "ReplayBuffer.jl"))
include(joinpath(SRC_DIR, "Learning.jl"))
include(joinpath(GAME_DIR, "game.jl"))
include(joinpath(GAME_DIR, "params.jl"))

@testset "MuZero Integration Tests" begin
	@testset "Initialization" begin
		rep = init_representation(hyper, conf)
		pred = init_prediction(hyper, conf)
		dyn = init_dynamics(hyper, conf)

		@test rep isa Chain
		@test pred isa Chain
		@test dyn isa Chain

		env = TicTacToe()
		obs = ReinforcementLearningBase.reset!(env)
		@test size(obs) == (3, 3, 3)

		obs_batch = rand(Float32, 3, 3, 7, conf.batch_size)
		hidden = rep(obs_batch)
		@test all(isfinite, hidden)
		@test minimum(hidden) >= 0.0f0
		@test maximum(abs.(hidden)) <= 1.0f0

		hidden_state = reshape(hidden, (conf.observation_shape..., conf.batch_size))
		action_batch = fill(Float32(1), conf.batch_size)
		next_hidden, reward = dyn(make_dynamics_input(hidden_state, action_batch, conf))
		@test all(isfinite, next_hidden)
		@test minimum(next_hidden) >= 0.0f0
		@test maximum(abs.(next_hidden)) <= 1.0f0
		@test all(isfinite, reward)
	end

	@testset "Gradient scaling helper" begin
		x = Float32[1, 2, 3]
		@test scale_gradient(x, 0.5f0) == x
		g = Flux.gradient(y -> sum(scale_gradient(y, 0.5f0)), x)[1]
		@test g == fill(0.5f0, length(x))
	end

	@testset "Game Queue & Local Buffer" begin
		local_buffer = Dict{Int, GameHistory}()

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

		insert_game!(local_buffer, history, 1, conf)

		@test length(local_buffer) == 1
		@test haskey(local_buffer, 1)

		game_queue = Channel{GameHistory}(10)
		put!(game_queue, history)
		@test isready(game_queue)
		taken_hist = take!(game_queue)
		@test taken_hist === history
	end

	@testset "Two-player value targets" begin
		test_conf = Config(conf; td_steps = 2, num_unroll_steps = 2)
		history = GameHistory(
			observation_history = rand(Float32, 3, 3, 3, 3),
			action_history = [1, 2, 3],
			reward_history = Float32[0, 0, 1],
			to_play_history = [1, 2, 1],
			child_visits = fill(1.0f0 / 9, 9, 3),
			root_values = Float32[0.2, -0.3, 0.4],
			reanalysed_predicted_root_values = nothing,
			priorities = nothing,
			game_priority = nothing,
		)

		@test isapprox(compute_target_value(history, 1, test_conf), 0.4f0 * test_conf.discount^2)
		@test isapprox(compute_target_value(history, 3, test_conf), 1.0f0)

		target_values, target_rewards, target_policies, actions = make_target(history, 2, test_conf)
		@test isapprox(target_values[1], -test_conf.discount)
		@test isapprox(target_values[2], 1.0f0)
		@test isapprox(target_rewards[1], 0.0f0)
		@test isapprox(target_rewards[2], 0.0f0)
		@test isapprox(target_rewards[3], 1.0f0)
		@test target_policies[:, 2] == history.child_visits[:, 3]
		@test actions[2] == 3
	end

	@testset "MCTS strict finite checks" begin
		strict_conf = Config(conf; allow_nonfinite_mcts = false)
		lenient_conf = Config(conf; allow_nonfinite_mcts = true)
		bad_logits = Float32[NaN32 for _ in conf.action_space]

		@test_throws ErrorException safe_policy_values(bad_logits, conf.action_space, strict_conf)
		@test safe_policy_values(bad_logits, conf.action_space, lenient_conf) == fill(1.0f0 / length(conf.action_space), length(conf.action_space))

		root = Node(prior = 0.0f0)
		root.children = Dict(
			3 => Node(prior = 0.1f0, visit_count = 2),
			7 => Node(prior = 0.1f0, visit_count = 9),
			1 => Node(prior = 0.1f0, visit_count = 4),
		)
		@test select_action(root, 0.0f0) == 7
	end

	@testset "MCTS updates legal actions after simulated moves" begin
		test_conf = Config(conf; stacked_observations = 0, num_iters = 1)
		env = TicTacToe()
		observation = Float32.(ReinforcementLearningBase.reset!(env))
		legal_actions = Int.(collect(ReinforcementLearningBase.legal_action_space(env, ReinforcementLearningBase.current_player(env))))
		rep = init_representation(hyper, test_conf)
		pred = init_prediction(hyper, test_conf)
		dyn = init_dynamics(hyper, test_conf)

		root = run_mcts(
			env,
			observation,
			legal_actions,
			ReinforcementLearningBase.current_player(env),
			false,
			(representation = rep, prediction = pred, dynamics = dyn),
			test_conf,
		)

		expanded_children = [(action, child) for (action, child) in root.children if !isnothing(child.children)]
		@test length(expanded_children) == 1
		action, child = only(expanded_children)
		@test !(action in child.legal_actions)
		@test length(child.legal_actions) == length(legal_actions) - 1
	end

	@testset "Two-player MCTS backprop signs" begin
		test_conf = Config(conf; discount = 0.9f0)
		root = Node(prior = 0.0f0, to_play = 1)
		child = Node(prior = 1.0f0, to_play = 2, reward = 1.0f0)
		stats = MinMaxStats(Inf32, -Inf32)

		backpropagate!([root, child], 0.4f0, 2, stats, test_conf)

		@test isapprox(node_value(child), 0.4f0)
		@test isapprox(node_value(root), 1.0f0 - 0.9f0 * 0.4f0)
	end

	@testset "Fixed aspiration RS search score" begin
		test_conf = Config(conf; use_rs = true, rs_R = 0.6f0, discount = 1.0f0)
		root = Node(prior = 0.0f0, to_play = 1, value_prior = 0.8f0, visit_count = 3, value_sum = 1.2f0)
		child = Node(prior = 1.0f0, to_play = 2, reward = 0.0f0, visit_count = 2, value_sum = -1.0f0)
		stats = MinMaxStats(Inf32, -Inf32)

		parent_mean = (root.value_prior / 2 + root.value_sum) / (root.visit_count + 1)
		child_value = child.reward - test_conf.discount * node_value(child, test_conf)
		expected = (child.visit_count + 1) * ((parent_mean + child.visit_count * child_value) / (child.visit_count + 1) - test_conf.rs_R) / (root.visit_count + 1)

		@test isapprox(rs_score(root, child, test_conf), expected)
		@test isapprox(search_score(root, child, stats, test_conf), expected)
	end

	@testset "Reward loss is always learned" begin
		test_conf = Config(conf; intermediate_rewards = false, batch_size = 2, num_unroll_steps = 1)
		value = zeros(Float32, 2, 2)
		reward = zeros(Float32, 2, 2)
		policy = zeros(Float32, length(test_conf.action_space), 2, 2)
		target_values = zeros(Float32, 2, 2)
		target_rewards = ones(Float32, 2, 2)
		target_policies = fill(1.0f0 / length(test_conf.action_space), length(test_conf.action_space), 2, 2)
		weights = ones(Float32, 1, 2)
		gradient_scale = ones(Float32, 1, 2)

		loss_with_reward_error = loss_base((value, reward, policy), (target_values, target_rewards, target_policies), weights, gradient_scale, test_conf)
		loss_without_reward_error = loss_base((value, target_rewards, policy), (target_values, target_rewards, target_policies), weights, gradient_scale, test_conf)

		@test loss_with_reward_error > loss_without_reward_error
	end

	@testset "Connect4 checkpoint selection" begin
		eval_mod = Module(:Connect4EvaluateForTests)
		Core.eval(eval_mod, :(include(path::AbstractString) = Base.include($eval_mod, path)))
		Base.include(eval_mod, joinpath(@__DIR__, "../games/connect4/evaluate.jl"))

		tmp = mktempdir()
		JLD2.jldsave(joinpath(tmp, "100_checkpoint.jld2"); step = 100)
		JLD2.jldsave(joinpath(tmp, "200_checkpoint.jld2"); step = 200)
		JLD2.jldsave(joinpath(tmp, "latest_checkpoint.jld2"); step = 200)

		eval_conf = eval_mod.Config(eval_mod.conf; networks_path = tmp)
		specs = eval_mod.checkpoint_specs(eval_conf, "all")
		@test [spec.requested_step for spec in specs] == [100, 200]

		JLD2.jldsave(joinpath(tmp, "latest_checkpoint.jld2"); step = 300)
		specs = eval_mod.checkpoint_specs(eval_conf, "all")
		@test [spec.requested_step for spec in specs] == [100, 200, 300]

		svg_path = joinpath(tmp, "checkpoint_winrate.svg")
		summaries = [
			(label = "100", step = 100, games = 10, wins = 5, win_rate = 0.5),
			(label = "200", step = 200, games = 10, wins = 6, win_rate = 0.6),
			(label = "300", step = 300, games = 10, wins = 7, win_rate = 0.7),
		]
		eval_mod.write_checkpoint_winrate_svg(svg_path, summaries)
		svg = read(svg_path, String)
		@test occursin(">100</text>", svg)
		@test occursin(">200</text>", svg)
		@test occursin(">300</text>", svg)
	end

	@testset "Gradient Step (Smoke Test)" begin
		local_buffer = Dict{Int, GameHistory}()
		test_conf = Config(conf; batch_size = 4, training_steps = 5, num_unroll_steps = 2)

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

		rep = init_representation(hyper, test_conf)
		pred = init_prediction(hyper, test_conf)
		dyn = init_dynamics(hyper, test_conf)

		_, batch = get_batch(local_buffer, test_conf)
		obs_b, act_b, t_v, t_r, t_p, w_b, g_scale = batch

		g_scale = permutedims(g_scale)
		test_conf.PER ? w_b = permutedims(w_b) : nothing

		grads = Flux.gradient(rep, pred, dyn) do r, p, d
			compute_total_loss(r, p, d, obs_b, act_b, t_v, t_r, t_p, w_b, g_scale, test_conf)
		end

		@test !isnothing(grads)
		@test !isnothing(grads[1])

		opt_rep = Flux.setup(Flux.AdamW(2e-4), rep)
		opt_pred = Flux.setup(Flux.AdamW(2e-4), pred)
		opt_dyn = Flux.setup(Flux.AdamW(2e-4), dyn)
		Flux.update!(opt_rep, rep, grads[1])
		Flux.update!(opt_pred, pred, grads[2])
		Flux.update!(opt_dyn, dyn, grads[3])
		assert_finite_networks((representation = rep, prediction = pred, dynamics = dyn), 1)

		println("Gradient calculation successful.")
	end
end

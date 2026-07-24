using BSON
using Serialization
using ReinforcementLearningBase
using Flux

drop_singleton_dims(x) = dropdims(x, dims = (findall(size(x) .== 1)...,))
insert_singleton_dim(xs::AbstractArray, dim::Integer) = reshape(xs, (size(xs)[1:(dim-1)]..., 1, size(xs)[dim:end]...))

function make_state_action(state::Array{Float32, 3}, action::Int, conf::Config)::Array{Float32, 3}
	norm_action = action / length(conf.action_space)
	action_plane = fill(Float32(norm_action), (conf.observation_shape[1], conf.observation_shape[2], 1))
	state_action = cat(state, action_plane, dims = 3)
	return state_action
end

mutable struct MinMaxStats
	min::Float32
	max::Float32
end

struct SearchState
	legal_actions::Vector{Int}
	to_play::Int
	terminal::Bool
end

function update_tree!(treeminmax::MinMaxStats, value::Float32)::Nothing
	treeminmax.min = treeminmax.min < value ? treeminmax.min : value
	treeminmax.max = treeminmax.max > value ? treeminmax.max : value
	return nothing
end

function normalize_tree_value(treeminmax::MinMaxStats, value::Float32)::Float32
	if treeminmax.max > treeminmax.min
		return (value - treeminmax.min) / (treeminmax.max - treeminmax.min)
	else
		return value
	end
end

function visit_softmax_temperature_fn(trained_steps::Int, conf::Config)::Float32
	decay_steps = isnothing(conf.temperature_decay_steps) ? conf.training_steps : conf.temperature_decay_steps
	decay_steps = max(decay_steps, 1)
	progress = clamp(Float32(trained_steps) / Float32(decay_steps), 0.0f0, 1.0f0)
	return conf.temperature_initial + progress * (conf.temperature_final - conf.temperature_initial)
end

function visit_softmax_temperature_fn(trained_steps::Int)::Float32
	if trained_steps < 500_000
		return 1.0f0
	elseif trained_steps < 750_000
		return 0.5f0
	else
		return 0.25f0
	end
end

using Distributions: Dirichlet, Categorical
using Parameters: @with_kw
using Flux: softmax

@with_kw mutable struct Node
	visit_count::Int = 0
	to_play::Int = 1
	prior::Float32
	# 展開時のネットワーク価値。RSではAlphaZeRSのノード初期価値として使う。
	value_prior::Float32 = 0.0f0
	value_sum::Float32 = 0.0
	children::Union{Dict{Int, Node}, Nothing} = nothing
	hidden_state::Union{Array{Float32, 3}, Nothing} = nothing
	legal_actions::Vector{Int} = Int[]
	reward::Union{Float32, Int} = 0
end

function expanded(node::Node)::Bool
	return !isnothing(node.children) && !isempty(node.children)
end

function node_hidden_state(node::Node)::Array{Float32, 3}
	hidden_state = node.hidden_state
	isnothing(hidden_state) && error("MCTS node hidden state was not initialized.")
	return hidden_state
end

first_scalar(x::Number)::Float32 = Float32(x)
first_scalar(x::AbstractArray)::Float32 = Float32(first(x))

function current_player_index(env)::Int
	player = ReinforcementLearningBase.current_player(env)
	player isa Int || error("MCTS current player must be an Int. Got $(typeof(player)).")
	return player
end

function legal_action_list(env, to_play::Int)::Vector{Int}
	return Int.(collect(ReinforcementLearningBase.legal_action_space(env, to_play)))
end

function mcts_hidden_state(hidden_state::AbstractArray, conf::Config)::Array{Float32, 3}
	reshaped = ndims(hidden_state) == 2 ? reshape(hidden_state, (conf.observation_shape..., 1)) : hidden_state
	return Array{Float32, 3}(drop_singleton_dims(reshaped))
end

function sanitize_mcts_value(x, conf::Config, context::String)::Float32
	value = Float32(x)
	if isfinite(value)
		return value
	end
	conf.allow_nonfinite_mcts && return 0.0f0
	error("$context became non-finite during MCTS: $value")
end

function node_value(node::Node)::Float32
	if node.visit_count == 0
		return 0
	else
		return node.value_sum / node.visit_count
	end
end

function node_value(node::Node, conf::Config)::Float32
	return sanitize_mcts_value(node_value(node), conf, "node value")
end

function safe_policy_values(policy_logits::Vector{Float32}, actions, conf::Config)::Vector{Float32}
	selected_logits = Float32[policy_logits[a] for a in actions]
	if isempty(selected_logits)
		return Float32[]
	end
	if !all(isfinite, selected_logits)
		conf.allow_nonfinite_mcts || error("policy logits became non-finite during MCTS")
		return fill(1.0f0 / length(selected_logits), length(selected_logits))
	end
	policy_values = softmax(selected_logits)
	if !all(isfinite, policy_values) || sum(policy_values) <= 0
		conf.allow_nonfinite_mcts || error("policy probabilities became non-finite during MCTS")
		return fill(1.0f0 / length(selected_logits), length(selected_logits))
	end
	return Float32.(policy_values ./ sum(policy_values))
end

function expand_node!(node, actions, to_play, reward, policy_logits, hidden_state, conf::Config; value_prior = 0.0f0)
	node.legal_actions = Int.(collect(actions))
	policy_values = safe_policy_values(policy_logits, actions, conf)
	policy = Dict([(a, policy_values[i]) for (i, a) in enumerate(actions)])
	node.children = Dict([(action, Node(prior = prob)) for (action, prob) in policy])
	node.to_play = Int(to_play)
	node.reward = sanitize_mcts_value(reward, conf, "reward")
	node.value_prior = sanitize_mcts_value(value_prior, conf, "value prior")
	hidden_state_array = Array{Float32, 3}(hidden_state)
	if !all(isfinite, hidden_state_array)
		conf.allow_nonfinite_mcts || error("hidden state became non-finite during MCTS")
		hidden_state_array = sanitize_mcts_value.(hidden_state_array, Ref(conf), Ref("hidden state"))
	end
	node.hidden_state = hidden_state_array
	return nothing
end

function mark_terminal_node!(node, to_play, reward, hidden_state, conf::Config; value_prior = 0.0f0)
	node.legal_actions = Int[]
	node.children = Dict{Int, Node}()
	node.to_play = Int(to_play)
	node.reward = sanitize_mcts_value(reward, conf, "reward")
	node.value_prior = sanitize_mcts_value(value_prior, conf, "value prior")
	hidden_state_array = Array{Float32, 3}(hidden_state)
	if !all(isfinite, hidden_state_array)
		conf.allow_nonfinite_mcts || error("hidden state became non-finite during MCTS")
		hidden_state_array = sanitize_mcts_value.(hidden_state_array, Ref(conf), Ref("hidden state"))
	end
	node.hidden_state = hidden_state_array
	return nothing
end

function search_state_after_path(env, parent::Node, actions::Vector{Int}, virtual_to_play::Int)::SearchState
	isnothing(env) && return SearchState(parent.legal_actions, virtual_to_play, false)

	search_env = deepcopy(env)
	for action in actions
		ReinforcementLearningBase.is_terminated(search_env) && break
		search_env(action)
	end

	terminal::Bool = ReinforcementLearningBase.is_terminated(search_env)
	to_play::Int = current_player_index(search_env)
	legal_actions = terminal ? Int[] : legal_action_list(search_env, to_play)
	return SearchState(legal_actions, to_play, terminal)
end

function add_exploration_noise!(node::Node, dirichlet_alpha::Float32, exploration_epsilon::Float32)::Nothing
	actions = collect(keys(node.children))
	noise = rand(Dirichlet(length(actions), dirichlet_alpha))
	for (a, n) in zip(actions, noise)
		node.children[a].prior = node.children[a].prior * (1 - exploration_epsilon) + n * exploration_epsilon
	end
	return nothing
end

function store_search_stats!(history::GameHistory, root::Node, action_space::Array{Int}, conf::Config)
	children=collect(values(root.children))
	children=filter!(x->!isnothing(x), children)
	sum_visits = sum([child.visit_count for child in children])
	history.child_visits = hcat(history.child_visits, [haskey(root.children, a) ? root.children[a].visit_count / sum_visits : 0.0f0 for a in action_space])
	value=node_value(root, conf)
	append!(history.root_values, value)
end

function store_unsearched_stats!(history::GameHistory, action_space::Array{Int})
	history.child_visits = hcat(history.child_visits, fill(1.0f0 / length(action_space), length(action_space)))
	append!(history.root_values, 0.0f0)
end

function get_stacked_observations(history::GameHistory, index::Int, num_stacked_observations::Int, conf::Config)::Array{Float32, 3}
	stacked_observations = copy(history.observation_history[:, :, :, index])
	for past_observation_index in (index-1):-1:(index-num_stacked_observations)
		if 1 <= past_observation_index
			action_plane = fill(Float32(history.action_history[past_observation_index]), (conf.observation_shape[1], conf.observation_shape[2], 1))
			previous_observation = cat(
				action_plane,
				history.observation_history[:, :, :, past_observation_index],
				dims = 3,
			)
		else
			previous_observation = cat(
				zeros(Float32, (conf.observation_shape[1], conf.observation_shape[2], 1)),
				zeros(Float32, conf.observation_shape),
				dims = 3,
			)
		end
		stacked_observations = cat(stacked_observations, previous_observation, dims = 3)
	end
	return stacked_observations
end

using Random
global rng = MersenneTwister(1234)

function select_child(node::Node, treeminmax::MinMaxStats, conf::Config)::Tuple{Int, Node}
	entries = collect(node.children)
	if isempty(entries)
		error("Cannot select a child from an unexpanded node.")
	end
	actions = [entry.first for entry in entries]
	children = [entry.second for entry in entries]
	# 設定に応じて通常のMuZero UCBか、固定希求水準RSで子ノードを評価する。
	scores = [search_score(node, child, treeminmax, conf) for child in children]
	if !any(isfinite, scores)
		conf.allow_nonfinite_mcts || error("all search scores became non-finite during MCTS")
		i = rand(eachindex(children))
		return actions[i], children[i]
	end
	scores = [isfinite(score) ? score : -Inf32 for score in scores]
	max_score = maximum(scores)
	max_scores = findall(x -> x == max_score, scores)
	i = rand(max_scores)
	return actions[i], children[i]
end

function child_value_from_parent(child::Node, conf::Config)::Float32
	value = child.reward + conf.discount * (length(conf.players) == 1 ? node_value(child, conf) : -node_value(child, conf))
	return sanitize_mcts_value(value, conf, "child value")
end

function ucb_score(parent_node::Node, child::Node, treeminmax::MinMaxStats, conf::Config)::Float32
	pb_c = (log2((parent_node.visit_count + conf.pb_c_base + 1) / conf.pb_c_base)
			+
			conf.pb_c_init)
	pb_c *= sqrt(parent_node.visit_count) / (child.visit_count + 1)
	prior_score = pb_c * child.prior

	if child.visit_count > 0
		value_score = normalize_tree_value(treeminmax, child_value_from_parent(child, conf))
	else
		value_score = 0
	end
	return sanitize_mcts_value(prior_score + value_score, conf, "UCB score")
end

function rs_score(parent_node::Node, child::Node, conf::Config)::Float32
	# AlphaZeRSでは展開済みノードを n_all = 1, q_sum_all = v / 2 で初期化している。
	# MuZero側では訪問数と価値和を直接持つため、ここで同じ基準値を再構成する。
	parent_visit_count = parent_node.visit_count + 1
	parent_value_sum = parent_node.value_prior / 2 + parent_node.value_sum
	parent_mean_value = parent_value_sum / parent_visit_count

	# RS = n * (Q_mean - R) / n_all。子の価値はUCBの価値項と同じく親視点へ変換する。
	child_visit_count = child.visit_count + 1
	child_value_sum = child.visit_count > 0 ? child.visit_count * child_value_from_parent(child, conf) : 0.0f0
	score_value_sum = parent_mean_value + child_value_sum
	rs = child_visit_count * (score_value_sum / child_visit_count - conf.rs_R) / parent_visit_count
	return sanitize_mcts_value(rs, conf, "RS score")
end

function search_score(parent_node::Node, child::Node, treeminmax::MinMaxStats, conf::Config)::Float32
	if conf.use_rs
		return rs_score(parent_node, child, conf)
	else
		return ucb_score(parent_node, child, treeminmax, conf)
	end
end

function backpropagate!(search_path::Vector{Node}, value::Real, to_play::Integer, treeminmax::MinMaxStats, conf::Config)::Nothing
	value32::Float32 = Float32(value)
	if length(conf.players) == 1
		for node in reverse(search_path)
			node.value_sum += value32
			node.visit_count += 1
			update_tree!(treeminmax, node.reward + conf.discount * node_value(node, conf))
			value32 = node.reward + conf.discount * value32
		end
	elseif length(conf.players) == 2
		for node in reverse(search_path)
			node.value_sum += value32
			node.visit_count += 1
			update_tree!(treeminmax, node.reward - conf.discount * node_value(node, conf))
			value32 = node.reward - conf.discount * value32
		end
	else
		ErrorException("backpropagate for more than 2 players is not implemented")
	end
	return nothing
end

function run_mcts(observation::Array{Float32, 3}, legal_actions::Vector{Int}, to_play::Int, exploration::Bool, NNs, conf::Config)::Node
	return run_mcts(nothing, observation, legal_actions, to_play, exploration, NNs, conf)
end

function run_mcts(env, observation::Array{Float32, 3}, legal_actions::Vector{Int}, to_play::Int, exploration::Bool, NNs, conf::Config)::Node
	root = Node(prior = 0.0)
	observation = insert_singleton_dim(observation, 4)
	hidden_state = NNs.representation(observation)
	if ndims(hidden_state)==2
		hidden_state=reshape(hidden_state, (conf.observation_shape..., 1))
	end

	root_predicted_value, policy_logits = NNs.prediction(hidden_state)
	hidden_state, root_predicted_value, policy_logits = drop_singleton_dims.([hidden_state, root_predicted_value, policy_logits])
	hidden_state_array::Array{Float32, 3} = Array{Float32, 3}(hidden_state)
	policy_logits_vector::Vector{Float32} = vec(Float32.(policy_logits))
	reward = 0.0f0

	if isempty(legal_actions)
		error("Legal actions should not be an empty array. Got $(legal_actions)")
	end
	@assert issubset(Set(legal_actions), Set(conf.action_space)) "Legal actions should be a subset of the action space."
	expand_node!(root, legal_actions, to_play, reward, policy_logits_vector, hidden_state_array, conf; value_prior = root_predicted_value[1])

	if exploration
		add_exploration_noise!(root, conf.dirichlet_alpha, conf.exploration_epsilon)
	end

	treeminmax::MinMaxStats = MinMaxStats(Float32(Inf), -Float32(Inf))

	max_tree_depth = 0
	for iter in 1:conf.num_iters
		node = root
		parent = root
		virtual_to_play = to_play
		search_path = Vector{Node}()
		push!(search_path, node)
		path_actions = Vector{Int}()
		current_tree_depth = 0
		action=0
		while expanded(node)
			current_tree_depth += 1
			parent = node
			action, node = select_child(node, treeminmax, conf)
			push!(path_actions, action)
			push!(search_path, node)
			virtual_to_play = mod1(virtual_to_play + 1, length(conf.players))
		end

		search_state::SearchState = search_state_after_path(env, parent, path_actions, virtual_to_play)
		state_action = make_state_action(node_hidden_state(parent), action, conf)
		state_action = insert_singleton_dim(state_action, 4)
		next_hidden_state, reward = NNs.dynamics(state_action)
		next_hidden_state_3d::Array{Float32, 3} = mcts_hidden_state(next_hidden_state, conf)
		reward = drop_singleton_dims(reward)
		reward_value = first_scalar(reward)

		if search_state.terminal || isempty(search_state.legal_actions)
			mark_terminal_node!(node, search_state.to_play, reward_value, next_hidden_state_3d, conf)
			terminal_value::Float32 = 0.0f0
			terminal_to_play::Int = search_state.to_play
			backpropagate!(search_path, terminal_value, terminal_to_play, treeminmax, conf)
			max_tree_depth = maximum([max_tree_depth, current_tree_depth])
			continue
		end

		obs_w::Int = conf.observation_shape[1]
		obs_h::Int = conf.observation_shape[2]
		obs_c::Int = conf.observation_shape[3]
		prediction_input = fill(0.0f0, (obs_w, obs_h, obs_c, 1))
		prediction_input[:, :, :, 1] .= next_hidden_state_3d
		value, policy_logits = NNs.prediction(prediction_input)
		value, policy_logits = drop_singleton_dims.([value, policy_logits])
		policy_logits = vec(Float32.(policy_logits))
		value_prior = first_scalar(value)

		expand_node!(node, search_state.legal_actions, search_state.to_play, reward_value, policy_logits, next_hidden_state_3d, conf; value_prior = value_prior)
		backpropagate!(search_path, value_prior, search_state.to_play, treeminmax, conf)
		max_tree_depth = maximum([max_tree_depth, current_tree_depth])
	end
	return root
end

function select_action(node::Node, temperature::Real)::Int
	entries = collect(node.children)
	actions = [entry.first for entry in entries]
	visit_counts = Int32[entry.second.visit_count for entry in entries]
	if temperature == 0.0f0
		action = actions[argmax(visit_counts)]
	elseif temperature == Inf
		action = rand(rng, actions)
	else
		visit_count_distribution = visit_counts .^ (1 / temperature)
		visit_count_distribution = visit_count_distribution ./ sum(visit_count_distribution)
		action = actions[rand(rng, Categorical(visit_count_distribution))]
	end
	return action
end

function expert_agent()
	error("Expert agent not implemented for this game.")
end

function select_opponent_action(env, opponent::String, stacked_observations::Array{Float32, 3}, conf::Config)::Int
	if opponent == "human"
		p = ReinforcementLearningBase.current_player(env)
		las = ReinforcementLearningBase.legal_action_space(env, p)
		return human_input()
	elseif opponent == "expert"
		return expert_agent()
	elseif opponent == "random"
		p = ReinforcementLearningBase.current_player(env)
		las = ReinforcementLearningBase.legal_action_space(env, p)
		@assert !isempty(las) "Legal actions should not be an empty array. Got $(las)"
		return rand(rng, las)
	else
		error("Wrong argument: opponent argument should be self, human, expert or random")
	end
end

function play_game(env, temperature, render::Bool, opponent::String, muzero_player::Int, NNs, conf::Config)::GameHistory
	history = GameHistory(Array{Float32}(undef, conf.observation_shape..., 0), Vector{Int}(), Vector{Float32}(), Vector{Int}(), Matrix{Float32}(undef, length(conf.action_space), 0), Vector{Float32}(), nothing, nothing, nothing)

	done = false
	if render
		;
		render_game(env);
	end
	observation = ReinforcementLearningBase.reset!(env)

	while !done && length(history.action_history) <= conf.max_moves
		if !isnothing(conf.temperature_threshold) && length(history.action_history) >= conf.temperature_threshold
			temperature = 0.0f0
		end
		p = ReinforcementLearningBase.current_player(env)
		history.observation_history = cat(history.observation_history, observation, dims = 4)
		stacked_observations = get_stacked_observations(history, lastindex(history.observation_history, 4), conf.stacked_observations, conf)
		root = nothing

		action = if opponent == "self" || muzero_player == p
			root = run_mcts(env, stacked_observations, ReinforcementLearningBase.legal_action_space(env, p), p, true, NNs, conf)
			select_action(root, temperature)
		else
			select_opponent_action(env, opponent, stacked_observations, conf)
		end

		observation = env(action)
		reward = convert(Float32, ReinforcementLearningBase.reward(env, p))
		done = ReinforcementLearningBase.is_terminated(env)

		if render
			println("Played action: $(action)")
			render_game(env)
		end

		if isnothing(root)
			store_unsearched_stats!(history, conf.action_space)
		else
			store_search_stats!(history, root, conf.action_space, conf)
		end
		append!(history.action_history, action)
		append!(history.reward_history, reward)
		push!(history.to_play_history, p)
	end
	return history
end

function self_play!(env, training_step, remote_NNs, game_queue::RemoteChannel, conf::Config)::Bool
	Random.seed!(conf.seed + myid())
	global rng = MersenneTwister(conf.seed + myid())

	NNs = set_inference_mode!(fetch(remote_NNs))
	last_network_update_step = 0
	last_sync_time = time()
	SYNC_INTERVAL = 2.0

	try
		while true
			if time() - last_sync_time > SYNC_INTERVAL
				current_step = fetch(training_step)

				if current_step > conf.training_steps
					break
				end

				if current_step > last_network_update_step + conf.checkpoint_interval
					NNs = set_inference_mode!(fetch(remote_NNs))
					last_network_update_step = current_step
					# @info "Networks synced in SelfPlay (Step: $current_step)" 
				end

				last_sync_time = time()
			end

			current_step = fetch(training_step)
			temperature = visit_softmax_temperature_fn(current_step, conf)

			history = play_game(
				env,
				temperature,
				false,
				"self",
				conf.muzero_player,
				NNs,
				conf,
			)

			put!(game_queue, history)
		end
	catch err
		step_text = try
			string(fetch(training_step))
		catch
			"unknown"
		end
		details = sprint(showerror, err, catch_backtrace())
		error("Self-play worker $(myid()) failed at training step $step_text with last synced checkpoint step $last_network_update_step:\n$details")
	end
	return true
end

function competitive_play!(env, NNs, conf::Config; buffer_to_disk = false)::Nothing
	history = play_game(env, 0.0f0, true, length(conf.players) == 1 ? "self" : conf.opponent, conf.muzero_player, NNs, conf)
	return nothing
end

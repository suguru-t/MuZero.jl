using Distributions: Categorical
using Random

"""
The value target is the discounted root value of the search tree td_steps into the
future, plus the discounted sum of all rewards until then.
"""
function compute_target_value(history::GameHistory, index::Int, conf::Config)::Float32
	bootstrap_index = index + conf.td_steps
	current_player = history.to_play_history[index]
	value = 0.0f0

	last_reward_index = min(bootstrap_index - 1, length(history.reward_history))
	for reward_index in index:last_reward_index
		reward = history.reward_history[reward_index]
		signed_reward = history.to_play_history[reward_index] == current_player ? reward : -reward
		value += signed_reward * conf.discount^(reward_index - index)
	end

	if bootstrap_index <= length(history.root_values)
		root_values = isnothing(history.reanalysed_predicted_root_values) ? history.root_values : history.reanalysed_predicted_root_values
		bootstrap_value = history.to_play_history[bootstrap_index] == current_player ? root_values[bootstrap_index] : -root_values[bootstrap_index]
		value += bootstrap_value * conf.discount^(bootstrap_index - index)
	end
	return value
end

function make_target(history::GameHistory, state_index::Int, conf::Config)::Tuple{Vector{Float32}, Vector{Float32}, Array{Float32, 2}, Vector{Int}}
	target_values, target_rewards, target_policies, actions = Vector{Float32}(), Vector{Float32}(), Matrix{Float32}(undef, length(conf.action_space), 0), Vector{Int}()
	for current_index in state_index:(state_index+conf.num_unroll_steps)
		if current_index <= length(history.root_values)
			value = compute_target_value(history, current_index, conf)
			append!(target_values, value)
			reward = current_index == state_index ? 0.0f0 : history.reward_history[current_index-1]
			append!(target_rewards, reward)
			target_policies = hcat(target_policies, history.child_visits[:, current_index])
			append!(actions, history.action_history[current_index])
		else
			append!(target_values, 0)
			reward_index = current_index - 1
			reward = reward_index <= length(history.reward_history) ? history.reward_history[reward_index] : 0.0f0
			append!(target_rewards, reward)
			target_policies = hcat(target_policies, fill(1.0f0 / length(conf.action_space), length(conf.action_space)))
			append!(actions, rand(rng, conf.action_space))
		end
	end
	return target_values, target_rewards, target_policies, actions
end

function normalize_probs(probs::Vector{Float32})::Vector{Float32}
	total = sum(probs)
	if isempty(probs)
		return Float32[]
	end
	if !isfinite(total) || total <= 0 || !all(isfinite, probs)
		return fill(1.0f0 / length(probs), length(probs))
	end
	return probs ./ total
end

function initialized_priorities(history::GameHistory)::Vector{Float32}
	priorities = history.priorities
	isnothing(priorities) && error("PER priorities were not initialized before sampling.")
	return priorities
end

function initialized_game_priority(history::GameHistory)::Float32
	priority = history.game_priority
	isnothing(priority) && error("PER game priority was not initialized before sampling.")
	return priority
end

function sample_position(history::GameHistory, conf::Config; force_uniform = false)::Tuple{Int, Float32}
	position_prob = 0.0f0
	if conf.PER && !force_uniform
		position_probs = normalize_probs(initialized_priorities(history))
		position_index = rand(rng, Categorical(position_probs))
		position_prob = position_probs[position_index]
	else
		position_index = rand(1:length(history.root_values))
	end
	return position_index, position_prob
end

function sample_n_games(buffer::Dict{Int, GameHistory}, conf::Config; force_uniform = false)::Vector{Tuple{Int, GameHistory, Float32}}
	if conf.PER && !force_uniform
		game_id_list = Vector{Int}()
		game_probs = Vector{Float32}()
		for (game_id, history) in buffer
			push!(game_id_list, game_id)
			push!(game_probs, initialized_game_priority(history))
		end
		game_probs_total = sum(game_probs)
		normalized_game_probs = if isempty(game_probs)
			Float32[]
		elseif !isfinite(game_probs_total) || game_probs_total <= 0 || !all(isfinite, game_probs)
			fill(1.0f0 / length(game_probs), length(game_probs))
		else
			game_probs ./ game_probs_total
		end
		game_prob_dict = Dict(game_id => prob for (game_id, prob) in zip(game_id_list, normalized_game_probs))
		selected_games = [game_id_list[i] for i in rand(rng, Categorical(normalized_game_probs), conf.batch_size)]
		n_games = [(game_id, buffer[game_id], game_prob_dict[game_id]) for game_id in selected_games]
	else
		selected_games = rand(collect(keys(buffer)), conf.batch_size)
		n_games = [(game_id, buffer[game_id], 0.0f0) for game_id in selected_games]
	end
	return n_games
end

function sample_game(buffer::Dict{Int, GameHistory}, num_played_games_count::Int, conf::Config; force_uniform = false)::Tuple{Int, GameHistory, Float32}
	game_prob = 0.0f0
	if conf.PER && !force_uniform
		game_probs = Vector{Float32}()
		for (_, history) in buffer
			push!(game_probs, initialized_game_priority(history))
		end
		game_probs_total = sum(game_probs)
		normalized_game_probs = if isempty(game_probs)
			Float32[]
		elseif !isfinite(game_probs_total) || game_probs_total <= 0 || !all(isfinite, game_probs)
			fill(1.0f0 / length(game_probs), length(game_probs))
		else
			game_probs ./ game_probs_total
		end
		game_index = rand(rng, Categorical(normalized_game_probs))
		game_prob = normalized_game_probs[game_index]
	else
		game_index = rand(1:length(buffer))
	end
	game_id = num_played_games_count - length(buffer) + game_index
	return game_id, buffer[game_id], game_prob
end

"""
Inserts a game into the LOCAL buffer (Learner side).
Handles PER initialization and buffer size limits.
"""
function insert_game!(buffer::Dict{Int, GameHistory}, history::GameHistory, next_game_id::Int, conf::Config)
	if conf.PER
		priorities = Vector{Float32}()
		for (i, root_value) in enumerate(history.root_values)
			priority = abs(root_value - compute_target_value(history, i, conf))^conf.PER_alpha
			append!(priorities, priority)
		end
		if isempty(priorities) || !all(isfinite, priorities) || sum(priorities) <= 0
			priorities = fill(1.0f0, length(history.root_values))
		end
		history.priorities = priorities
		history.game_priority = maximum(history.priorities)
	end

	buffer[next_game_id] = history

	if length(buffer) > conf.replay_buffer_size
		oldest_id = next_game_id - conf.replay_buffer_size
		if haskey(buffer, oldest_id)
			delete!(buffer, oldest_id)
		end
	end
end

function update_priorities!(buffer::Dict{Int, GameHistory}, priorities::Matrix{Float32}, index_batch::AbstractVector{Tuple{Int, Int}})::Nothing
	for i in eachindex(index_batch)
		game_id, game_pos = index_batch[i]
		if haskey(buffer, game_id)
			maybe_priorities = buffer[game_id].priorities
			if isnothing(maybe_priorities)
				continue
			end
			stored_priorities = maybe_priorities::Vector{Float32}

			priority = priorities[:, i]
			start_index = game_pos
			end_index = min(game_pos + size(priorities, 1) - 1, lastindex(stored_priorities))
			update_len = end_index - start_index + 1

			if update_len > 0
				stored_priorities[start_index:end_index] = priority[1:update_len]
				buffer[game_id].game_priority = maximum(stored_priorities)
			end
		end
	end
end

function get_batch(buffer::Dict{Int, GameHistory}, conf::Config)::Tuple{Vector{Tuple{Int, Int}}, Tuple{Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Array{Float32, 3}, Any, Vector{Float32}}}
	total_samples = sum([length(history.root_values) for history in values(buffer)])
	index_batch = Vector{Tuple{Int, Int}}()
	observation_batch = Array{Float32}(undef, conf.observation_shape[1], conf.observation_shape[2], (conf.observation_shape[3]*(conf.stacked_observations+1)+conf.stacked_observations), 0)
	action_batch = Array{Float32}(undef, conf.num_unroll_steps+1, 0)
	reward_batch = Array{Float32}(undef, conf.num_unroll_steps+1, 0)
	value_batch = Array{Float32}(undef, conf.num_unroll_steps+1, 0)
	policy_batch = Array{Float32}(undef, length(conf.action_space), conf.num_unroll_steps+1, 0)
	gradient_scale_batch = Vector{Float32}()
	weight_batch = conf.PER ? Vector{Float32}() : nothing
	n_games = sample_n_games(buffer, conf)
	for (game_id, history, game_prob) in n_games
		game_pos, pos_prob = sample_position(history, conf)
		target_values, target_rewards, target_policies, actions = make_target(history, game_pos, conf)
		push!(index_batch, (game_id, game_pos))
		observation_batch = cat(observation_batch, get_stacked_observations(history, game_pos, conf.stacked_observations, conf), dims = ndims(observation_batch))
		action_batch = hcat(action_batch, actions)
		reward_batch = hcat(reward_batch, target_rewards)
		value_batch = hcat(value_batch, target_values)
		policy_batch = cat(policy_batch, target_policies, dims = ndims(target_policies) + 1)
		push!(gradient_scale_batch, min(conf.num_unroll_steps, length(history.action_history)+1 - game_pos))
		conf.PER ? push!(weight_batch, 1 / (total_samples * game_prob * pos_prob)) : nothing
	end
	if conf.PER
		max_weight = maximum(weight_batch)
		weight_batch = (!isfinite(max_weight) || max_weight <= 0) ? ones(Float32, length(weight_batch)) : weight_batch ./ max_weight
	end
	return index_batch, (observation_batch, action_batch, value_batch, reward_batch, policy_batch, weight_batch, gradient_scale_batch)
end

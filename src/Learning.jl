using Serialization
using Parameters: @with_kw
using Statistics: mean
using Flux: cpu, Parallel
using Flux: Flux
using Base.Threads

using Flux: relu, softmax, flatten
using Flux.Losses: mse, logitcrossentropy, crossentropy
using ParameterSchedulers
using TensorBoardLogger
using Logging
using JLD2
using Dates

using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, MeanPool, MaxPool, AdaptiveMeanPool
using Zygote: Zygote

invert_scaling(x) = convert(Float32, sign(x) * (((sqrt(1 + 4 * 0.001 * (abs.(x) + 1 + 0.001)) - 1) / (2 * 0.001))^2 - 1))
scaling(x) = convert(Float32, sign(x) * (sqrt(abs(x) + 1) - 1 + 0.001 * x))

to_singletons(x) = reshape(x, size(x)..., 1)
squeeze(x) = reshape(x, size(x)[1:(end-1)])
unsqueeze(xs::AbstractArray, dim::Integer) = reshape(xs, (size(xs)[1:(dim-1)]..., 1, size(xs)[dim:end]...))

struct Split{T}
	paths::T
end
Split(paths...) = Split(paths)
Flux.@layer Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

function make_dense(indim::Int, outdim::Int, bnmom::Float32, hyper::FeedForwardHP)
	if hyper.use_batch_norm
		Chain(Dense(indim, outdim), BatchNorm(outdim, relu, momentum = bnmom))
	else
		Dense(indim, outdim, relu)
	end
end

hlayers(depth::Int, hsize, bnmom, hyper) = [make_dense(hsize, hsize, bnmom, hyper) for _ in 1:depth]

function init_representation(hyper::FeedForwardHP, conf::Config)
	indim = prod([conf.observation_shape[1], conf.observation_shape[2], (conf.observation_shape[3] * (conf.stacked_observations + 1) + conf.stacked_observations)])
	outdim = hyper.hidden_state_size
	bnmom = hyper.batch_norm_momentum
	hsize = hyper.width_hidden
	layers = Chain(flatten,
		make_dense(indim, hsize, bnmom, hyper),
		hlayers(hyper.depth_representation, hsize, bnmom, hyper)...,
		Dense(hsize, outdim),
	)
	return layers
end

function init_prediction(hyper::FeedForwardHP, conf::Config)
	bnmom = hyper.batch_norm_momentum
	indim = hyper.hidden_state_size
	outdim = length(conf.action_space)
	hsize = hyper.width_hidden
	common = Chain(flatten,
		make_dense(indim, hsize, bnmom, hyper),
		hlayers(hyper.depth_prediction, hsize, bnmom, hyper)...)
	value_head = Chain(
		hlayers(hyper.depth_value, hsize, bnmom, hyper)...,
		Dense(hsize, 1, tanh))
	policy_head = Chain(
		hlayers(hyper.depth_policy, hsize, bnmom, hyper)...,
		Dense(hsize, outdim),
		softmax)
	return Chain(common, Split(value_head, policy_head))
end

function init_dynamics(hyper::FeedForwardHP, conf::Config)
	bnmom = hyper.batch_norm_momentum
	indim = prod([conf.observation_shape[1], conf.observation_shape[2], (conf.observation_shape[3] + 1)])
	outdim = hyper.hidden_state_size
	hsize = hyper.width_hidden
	common = Chain(
		flatten,
		make_dense(indim, hsize, bnmom, hyper),
		hlayers(hyper.depth_dynamics, hsize, bnmom, hyper)...,
	)
	state_head = Chain(
		hlayers(hyper.depth_state_head, hsize, bnmom, hyper)...,
		Dense(hsize, outdim),
	)
	reward_head = Chain(
		hlayers(hyper.depth_reward, hsize, bnmom, hyper)...,
		Dense(hsize, 1, hyper.reward_activation))
	return Chain(common, Split(state_head, reward_head))
end

function resnet_block(size::Tuple{Int, Int}, n::Int, bnmom::Float32)
	;
end
function init_representation(hyper::ResNetHP, conf::Config)
	;
end
function init_prediction(hyper::ResNetHP, conf::Config)
	;
end
function init_dynamics(hyper::ResNetHP, conf::Config)
	;
end

function make_dynamics_input(states::Array{Float32, 4}, actions::AbstractVector, conf::Config)::Array{Float32, 4}
	norm_actions = actions ./ length(conf.action_space)
	reshaped_actions = reshape(norm_actions, 1, 1, 1, :)
	w, h = conf.observation_shape[1], conf.observation_shape[2]
	action_planes = repeat(reshaped_actions, w, h, 1, 1)
	scaled_states = states .* 2.0f0
	return cat(scaled_states, action_planes, dims = 3)
end

function learning!(training_step, remote_NNs, game_queue::RemoteChannel, conf::Config, hyper)::Bool
	local_buffer = Dict{Int, GameHistory}()
	next_game_id = 1

	timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	log_dir = joinpath(conf.results_path, "tensorboard_logs", timestamp)
	mkpath(log_dir)
	logger = TBLogger(log_dir, tb_overwrite, min_level = Logging.Info)

	println("Learner: Waiting for initial games...")

	while length(local_buffer) < conf.batch_size
		if isready(game_queue)
			hist = take!(game_queue)
			insert_game!(local_buffer, hist, next_game_id, conf)
			next_game_id += 1
		else
			sleep(1.0)
		end
	end
	println("Learner: Buffer has $(length(local_buffer)) games. Starting training. Logs at: $log_dir")

	# FIX: Increased Learning Rate to 2e-3 (was 1e-4)
	opt_def = Flux.AdamW(2e-3, (0.9, 0.999), 1e-4)
	schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(λ0 = 2e-3, λ1 = 1e-4, period = 50))

	NNs = fetch(remote_NNs)
	representation = deepcopy(NNs.representation)
	prediction = deepcopy(NNs.prediction)
	dynamics = deepcopy(NNs.dynamics)

	opt_state_rep = Flux.setup(opt_def, representation)
	opt_state_pred = Flux.setup(opt_def, prediction)
	opt_state_dyn = Flux.setup(opt_def, dynamics)

	training_step_ = 0

	while training_step_ ≤ conf.training_steps

		while isready(game_queue)
			hist = take!(game_queue)
			insert_game!(local_buffer, hist, next_game_id, conf)
			next_game_id += 1
		end

		next_batch = get_batch(local_buffer, conf)
		index_batch, batch = next_batch
		observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch = batch

		gradient_scale_batch = permutedims(gradient_scale_batch)
		conf.PER ? weight_batch = permutedims(weight_batch) : nothing

		current_eta = ParameterSchedulers.next!(schedule)
		Flux.adjust!(opt_state_rep, current_eta)
		Flux.adjust!(opt_state_pred, current_eta)
		Flux.adjust!(opt_state_dyn, current_eta)

		val, grads = Flux.withgradient(representation, prediction, dynamics) do m_rep, m_pred, m_dyn
			compute_total_loss(m_rep, m_pred, m_dyn, observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch, conf)
		end

		Flux.update!(opt_state_rep, representation, grads[1])
		Flux.update!(opt_state_pred, prediction, grads[2])
		Flux.update!(opt_state_dyn, dynamics, grads[3])

		if conf.PER
			(final_pred_values, _, _) = unroll_network(representation, prediction, dynamics, observation_batch, action_batch, conf)
			priorities = (abs.(final_pred_values - target_values)) .^ conf.PER_alpha
			update_priorities!(local_buffer, priorities, index_batch)
		end

		training_step_ += 1

		# Detailed logging
		if training_step_ <= 100 || training_step_ % 50 == 0
			# Recalculate component losses for display (no gradient needed)
			# This helps debugging which part isn't learning
			(v_loss, p_loss, r_loss) = compute_loss_breakdown(representation, prediction, dynamics, observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch, conf)

			with_logger(logger) do
				@info "train" loss=val value_loss=v_loss policy_loss=p_loss reward_loss=r_loss learning_rate=current_eta
			end
		end

		take!(training_step)
		put!(training_step, training_step_)

		if training_step_ % conf.checkpoint_interval == 0 && training_step_ > 1
			take!(remote_NNs)
			put!(remote_NNs, (representation = representation, prediction = prediction, dynamics = dynamics))

			rep_cpu = cpu(representation)
			pred_cpu = cpu(prediction)
			dyn_cpu = cpu(dynamics)

			jldsave(joinpath(conf.networks_path, "latest_checkpoint.jld2");
				representation = rep_cpu,
				prediction = pred_cpu,
				dynamics = dyn_cpu,
				step = training_step_,
			)

			if training_step_ % (conf.checkpoint_interval * 5) == 0
				jldsave(joinpath(conf.networks_path, "$(training_step_)_checkpoint.jld2");
					representation = rep_cpu,
					prediction = pred_cpu,
					dynamics = dyn_cpu,
					step = training_step_,
				)
			end
		end
	end
	return true
end

function compute_total_loss(rep, pred, dyn, obs_batch, act_batch, t_vals, t_rews, t_pols, w_batch, g_scale, conf)
	(p_vals, p_rews, p_pols) = unroll_network(rep, pred, dyn, obs_batch, act_batch, conf)
	preds = (p_vals, p_rews, p_pols)
	targets = (t_vals, t_rews, t_pols)
	base_loss = loss_base(preds, targets, w_batch, g_scale, conf)
	return base_loss
end

# New helper for breakdown
function compute_loss_breakdown(rep, pred, dyn, obs_batch, act_batch, t_vals, t_rews, t_pols, w_batch, g_scale, conf)
	(p_vals, p_rews, p_pols) = unroll_network(rep, pred, dyn, obs_batch, act_batch, conf)

	!conf.PER ? w_batch = 1.0f0 : nothing

	v_loss = mse(p_vals, t_vals, agg = x -> mean((sum(x, dims = 1) ./ g_scale) .* w_batch))

	if conf.intermediate_rewards
		r_loss = mse(p_rews, t_rews, agg = x -> mean((sum(x, dims = 1) ./ g_scale) .* w_batch))
	else
		r_loss = 0.0f0
	end

	p_loss = logitcrossentropy(p_pols, t_pols, agg = x -> mean((sum(x, dims = 2) ./ g_scale) .* w_batch))

	return (v_loss, p_loss, r_loss)
end

function loss_base(predictions, targets, weight_batch, gradient_scale_batch, conf)
	value, reward, policy_logits = predictions
	target_values, target_rewards, target_policies = targets
	!conf.PER ? weight_batch = 1.0f0 : nothing
	value_loss = mse(value, target_values, agg = x -> mean((sum(x, dims = 1) ./ gradient_scale_batch) .* weight_batch))
	if conf.intermediate_rewards
		reward_loss = mse(reward, target_rewards, agg = x -> mean((sum(x, dims = 1) ./ gradient_scale_batch) .* weight_batch))
	else
		reward_loss = 0.0f0
	end
	policy_loss = logitcrossentropy(policy_logits, target_policies, agg = x -> mean((sum(x, dims = 2) ./ gradient_scale_batch) .* weight_batch))
	return sum([value_loss, reward_loss, policy_loss])
end

function unroll_network(representation, prediction, dynamics, observation_batch, action_batch, conf)
	hidden_state = representation(observation_batch)
	if ndims(hidden_state) == 2
		hidden_state = reshape(hidden_state, (conf.observation_shape..., conf.batch_size))
	end

	p0_val, p0_pol = prediction(hidden_state)
	p0_pol = Flux.unsqueeze(p0_pol, 2)

	vals_buf = Zygote.Buffer(Vector{AbstractArray}(undef, conf.num_unroll_steps + 1))
	pols_buf = Zygote.Buffer(Vector{AbstractArray}(undef, conf.num_unroll_steps + 1))
	rews_buf = Zygote.Buffer(Vector{AbstractArray}(undef, conf.num_unroll_steps + 1))

	vals_buf[1] = p0_val
	pols_buf[1] = p0_pol
	zero_rew = zero(p0_val)
	rews_buf[1] = zero_rew

	curr_state = hidden_state

	for k ∈ 1:conf.num_unroll_steps
		state_action = make_dynamics_input(curr_state, action_batch[k, :], conf)
		curr_state, reward = dynamics(state_action)
		if ndims(curr_state) == 2
			curr_state = reshape(curr_state, (conf.observation_shape..., conf.batch_size))
		end
		val, pol = prediction(curr_state)
		pol = Flux.unsqueeze(pol, 2)
		vals_buf[k+1] = val
		pols_buf[k+1] = pol
		rews_buf[k+1] = reward
	end

	vals = copy(vals_buf)
	pols = copy(pols_buf)
	rews = copy(rews_buf)

	final_vals = reduce(vcat, vals)
	final_rews = reduce(vcat, rews)
	final_pols = reduce((x, y) -> cat(x, y, dims = 2), pols)

	return (final_vals, final_rews, final_pols)
end

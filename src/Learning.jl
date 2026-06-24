using Serialization
using Parameters: @with_kw
using Statistics: mean
using Flux: cpu, gpu, Parallel
using Flux: Flux
using Base.Threads

using Flux: relu, sigmoid, softmax, flatten
using Flux.Losses: mse, logitcrossentropy, crossentropy
using ParameterSchedulers
using TensorBoardLogger
using Logging
using JLD2
using Dates

using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, MeanPool, MaxPool, AdaptiveMeanPool
using Zygote: Zygote

apply_tanh(x) = tanh.(x)
apply_sigmoid(x) = sigmoid.(x)
scale_gradient(x, scale::Real) = x .* scale .+ Zygote.dropgrad(x) .* (1 - scale)

function maybe_gpu(x, conf::Config)
	x === nothing && return nothing
	return getproperty(conf, :selfplay_on_gpu) ? gpu(x) : x
end

function reclaim_cuda_if_loaded()
	if isdefined(Main, :CUDA)
		Main.CUDA.reclaim()
	end
	return nothing
end

function set_training_mode!(NNs)
	Flux.trainmode!(NNs.representation)
	Flux.trainmode!(NNs.prediction)
	Flux.trainmode!(NNs.dynamics)
	return NNs
end

function set_inference_mode!(NNs)
	Flux.testmode!(NNs.representation)
	Flux.testmode!(NNs.prediction)
	Flux.testmode!(NNs.dynamics)
	return NNs
end

function assert_finite_scalar(name::String, value, step::Int)
	scalar = Float32(cpu(value))
	isfinite(scalar) || error("$name became non-finite at training step $step: $scalar")
	return scalar
end

function assert_finite_array(name::String, value, step::Int)
	values = cpu(value)
	all(isfinite, values) || error("$name became non-finite at training step $step")
	return value
end

function assert_finite_tree(name::String, value, step::Int)
	values, _ = Flux.destructure(value)
	values = cpu(values)
	all(isfinite, values) || error("$name became non-finite at training step $step")
	return value
end

function assert_finite_networks(NNs, step::Int)
	assert_finite_tree("representation parameters", NNs.representation, step)
	assert_finite_tree("prediction parameters", NNs.prediction, step)
	assert_finite_tree("dynamics parameters", NNs.dynamics, step)
	return NNs
end

function assert_finite_prediction_outputs(representation, prediction, dynamics, observation_batch, action_batch, conf::Config, step::Int)
	values, rewards, policies = unroll_network(representation, prediction, dynamics, observation_batch, action_batch, conf)
	assert_finite_array("checkpoint value output", values, step)
	assert_finite_array("checkpoint reward output", rewards, step)
	assert_finite_array("checkpoint policy output", policies, step)
	return nothing
end

invert_scaling(x) = convert(Float32, sign(x) * (((sqrt(1 + 4 * 0.001 * (abs.(x) + 1 + 0.001)) - 1) / (2 * 0.001))^2 - 1))
scaling(x) = convert(Float32, sign(x) * (sqrt(abs(x) + 1) - 1 + 0.001 * x))

to_singletons(x) = reshape(x, size(x)..., 1)
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
		Dense(hsize, outdim), apply_sigmoid,
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
		Dense(hsize, 1), apply_tanh)
	policy_head = Chain(
		hlayers(hyper.depth_policy, hsize, bnmom, hyper)...,
		Dense(hsize, outdim))
	return Chain(common, Flux.Parallel(tuple, value_head, policy_head))
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
		Dense(hsize, outdim), apply_sigmoid,
	)
	reward_head = Chain(
		hlayers(hyper.depth_reward, hsize, bnmom, hyper)...,
		Dense(hsize, 1), apply_tanh)
	return Chain(common, Flux.Parallel(tuple, state_head, reward_head))
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

function make_dynamics_input(states::AbstractArray{Float32, 4}, actions::AbstractVector, conf::Config)
	norm_actions = actions ./ length(conf.action_space)
	reshaped_actions = reshape(norm_actions, 1, 1, 1, :)
	w, h = conf.observation_shape[1], conf.observation_shape[2]
	action_planes = repeat(reshaped_actions, w, h, 1, 1)
	return cat(states, action_planes, dims = 3)
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

	opt_def = Flux.AdamW(2e-4, (0.9, 0.999), 1e-4)
	schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(l0 = 2e-4, l1 = 2e-5, period = max(conf.training_steps, 1)))

	NNs = fetch(remote_NNs)
	representation = maybe_gpu(deepcopy(NNs.representation), conf)
	prediction = maybe_gpu(deepcopy(NNs.prediction), conf)
	dynamics = maybe_gpu(deepcopy(NNs.dynamics), conf)
	set_training_mode!((representation = representation, prediction = prediction, dynamics = dynamics))

	opt_state_rep = Flux.setup(opt_def, representation)
	opt_state_pred = Flux.setup(opt_def, prediction)
	opt_state_dyn = Flux.setup(opt_def, dynamics)

	training_step_ = 0

	while training_step_ <= conf.training_steps

		while isready(game_queue)
			hist = take!(game_queue)
			insert_game!(local_buffer, hist, next_game_id, conf)
			next_game_id += 1
		end

		next_batch = get_batch(local_buffer, conf)
		index_batch, batch = next_batch
		observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch = batch

		observation_batch = maybe_gpu(observation_batch, conf)
		action_batch = maybe_gpu(action_batch, conf)
		target_values = maybe_gpu(target_values, conf)
		target_rewards = maybe_gpu(target_rewards, conf)
		target_policies = maybe_gpu(target_policies, conf)
		weight_batch = maybe_gpu(weight_batch, conf)
		gradient_scale_batch = maybe_gpu(gradient_scale_batch, conf)

		gradient_scale_batch = permutedims(gradient_scale_batch)
		conf.PER ? weight_batch = permutedims(weight_batch) : nothing

		current_eta = ParameterSchedulers.next!(schedule)
		Flux.adjust!(opt_state_rep, current_eta)
		Flux.adjust!(opt_state_pred, current_eta)
		Flux.adjust!(opt_state_dyn, current_eta)

		val, grads = Flux.withgradient(representation, prediction, dynamics) do m_rep, m_pred, m_dyn
			compute_total_loss(m_rep, m_pred, m_dyn, observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch, conf)
		end
		assert_finite_scalar("loss", val, training_step_)
		assert_finite_tree("representation gradient", grads[1], training_step_)
		assert_finite_tree("prediction gradient", grads[2], training_step_)
		assert_finite_tree("dynamics gradient", grads[3], training_step_)

		Flux.update!(opt_state_rep, representation, grads[1])
		Flux.update!(opt_state_pred, prediction, grads[2])
		Flux.update!(opt_state_dyn, dynamics, grads[3])
		assert_finite_networks((representation = representation, prediction = prediction, dynamics = dynamics), training_step_)

		grads = nothing
		GC.gc(true)
		reclaim_cuda_if_loaded()

		if conf.PER
			(final_pred_values, _, _) = unroll_network(representation, prediction, dynamics, observation_batch, action_batch, conf)
			priorities = (abs.(final_pred_values - target_values)) .^ conf.PER_alpha
			assert_finite_array("PER priorities", priorities, training_step_)
			update_priorities!(local_buffer, cpu(priorities), index_batch)
		end

		training_step_ += 1

		# Detailed logging
		if training_step_ <= 100 || training_step_ % 50 == 0
			# Recalculate component losses for display (no gradient needed)
			# This helps debugging which part isn't learning
			(v_loss, p_loss, r_loss) = compute_loss_breakdown(representation, prediction, dynamics, observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch, conf)

			with_logger(logger) do
				set_step!(logger, training_step_)
				@info "train" loss=val value_loss=v_loss policy_loss=p_loss reward_loss=r_loss learning_rate=current_eta log_step_increment=0
			end
		end
		if training_step_ % conf.checkpoint_interval == 0
			println("Step: $training_step_")
		end

		take!(training_step)
		put!(training_step, training_step_)

		if training_step_ % conf.checkpoint_interval == 0 && training_step_ > 1
		
			rep_cpu = cpu(deepcopy(representation))
			pred_cpu = cpu(deepcopy(prediction))
			dyn_cpu = cpu(deepcopy(dynamics))
			checkpoint_NNs = set_inference_mode!((representation = rep_cpu, prediction = pred_cpu, dynamics = dyn_cpu))
			assert_finite_networks(checkpoint_NNs, training_step_)
			assert_finite_prediction_outputs(rep_cpu, pred_cpu, dyn_cpu, cpu(observation_batch), cpu(action_batch), conf, training_step_)

			take!(remote_NNs)
			put!(remote_NNs, checkpoint_NNs)

			jldsave(joinpath(conf.networks_path, "latest_checkpoint.jld2");
				representation = checkpoint_NNs.representation,
				prediction = checkpoint_NNs.prediction,
				dynamics = checkpoint_NNs.dynamics,
				step = training_step_,
			)

			if training_step_ % (conf.checkpoint_interval * 5) == 0
				jldsave(joinpath(conf.networks_path, "$(training_step_)_checkpoint.jld2");
					representation = checkpoint_NNs.representation,
					prediction = checkpoint_NNs.prediction,
					dynamics = checkpoint_NNs.dynamics,
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
	policy_g_scale = reshape(g_scale, 1, 1, :)
	policy_weight_batch = conf.PER ? reshape(w_batch, 1, 1, :) : 1.0f0

	v_loss = conf.value_loss_weight * mse(p_vals, t_vals, agg = x -> mean((sum(x, dims = 1) ./ g_scale) .* w_batch))

	r_loss = mse(p_rews, t_rews, agg = x -> mean((sum(x, dims = 1) ./ g_scale) .* w_batch))

	p_loss = logitcrossentropy(p_pols, t_pols, agg = x -> mean((sum(x, dims = 2) ./ policy_g_scale) .* policy_weight_batch))

	return (v_loss, p_loss, r_loss)
end

function loss_base(predictions, targets, weight_batch, gradient_scale_batch, conf)
	value, reward, policy_logits = predictions
	target_values, target_rewards, target_policies = targets
	!conf.PER ? weight_batch = 1.0f0 : nothing
	policy_gradient_scale_batch = reshape(gradient_scale_batch, 1, 1, :)
	policy_weight_batch = conf.PER ? reshape(weight_batch, 1, 1, :) : 1.0f0
	value_loss = conf.value_loss_weight * mse(value, target_values, agg = x -> mean((sum(x, dims = 1) ./ gradient_scale_batch) .* weight_batch))
	reward_loss = mse(reward, target_rewards, agg = x -> mean((sum(x, dims = 1) ./ gradient_scale_batch) .* weight_batch))
	policy_loss = logitcrossentropy(policy_logits, target_policies, agg = x -> mean((sum(x, dims = 2) ./ policy_gradient_scale_batch) .* policy_weight_batch))
	return sum([value_loss, reward_loss, policy_loss])
end

function unroll_network(representation, prediction, dynamics, observation_batch, action_batch, conf)
	hidden_state = representation(observation_batch)
	if ndims(hidden_state) == 2
		hidden_state = reshape(hidden_state, (conf.observation_shape..., conf.batch_size))
	end

	p0_val, p0_pol = prediction(hidden_state)
	p0_pol = Flux.unsqueeze(p0_pol; dims = 2)

	vals_buf = Zygote.Buffer(Vector{AbstractArray}(undef, conf.num_unroll_steps + 1))
	pols_buf = Zygote.Buffer(Vector{AbstractArray}(undef, conf.num_unroll_steps + 1))
	rews_buf = Zygote.Buffer(Vector{AbstractArray}(undef, conf.num_unroll_steps + 1))

	vals_buf[1] = p0_val
	pols_buf[1] = p0_pol
	zero_rew = zero(p0_val)
	rews_buf[1] = zero_rew

	curr_state = hidden_state

	for k in 1:conf.num_unroll_steps
		state_action = make_dynamics_input(curr_state, action_batch[k, :], conf)
		curr_state, reward = dynamics(state_action)
		if ndims(curr_state) == 2
			curr_state = reshape(curr_state, (conf.observation_shape..., conf.batch_size))
		end
		val, pol = prediction(curr_state)
		pol = Flux.unsqueeze(pol; dims = 2)
		vals_buf[k+1] = val
		pols_buf[k+1] = pol
		rews_buf[k+1] = reward
		curr_state = scale_gradient(curr_state, 0.5f0)
	end

	vals = copy(vals_buf)
	pols = copy(pols_buf)
	rews = copy(rews_buf)

	final_vals = reduce(vcat, vals)
	final_rews = reduce(vcat, rews)
	final_pols = reduce((x, y) -> cat(x, y, dims = 2), pols)

	return (final_vals, final_rews, final_pols)
end

const GAME_DIR = @__DIR__

const conf = Config(
	observation_shape = (6, 7, 3),
	action_space = collect(1:7),
	players = collect(1:2),
	stacked_observations = 0,
	num_workers = 9,
	max_moves = 42,
	num_unroll_steps = 20,
	td_steps = 10,
	PER = true,
	opponent = "human",
	training_steps = 100000,
	batch_size = 64,
	num_iters = 100,
	use_rs = true, #true:RS使用/false:RS不使用
	rs_R = 0.6f0, #希求水準の値
	checkpoint_interval = 500,
	temperature_initial = 1.0f0,
	temperature_final = 1.0f0,
	temperature_decay_steps = 10000,
	temperature_threshold = 30,
	intermediate_rewards = true,

	#MuZeroの結果を保存する場合
	#results_path = mkpath(joinpath(GAME_DIR, "results")),
	#networks_path = mkpath(joinpath(GAME_DIR, "networks")),

	#MuzeRSの結果を保存する場合
	results_path = mkpath(joinpath(GAME_DIR, "results_muzers")),
	networks_path = mkpath(joinpath(GAME_DIR, "networks_muzers")),
)

const hyper = FeedForwardHP(
	width_hidden = 64,
	depth_representation = 4,
	depth_prediction = 4,
	depth_dynamics = 4,
	depth_policy = 2,
	depth_value = 2,
	depth_reward = 2,
	depth_state_head = 2,
	use_batch_norm = true,
	hidden_state_size = 126,
	reward_activation = tanh,
)

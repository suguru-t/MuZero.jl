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
	training_steps = 50000,
	batch_size = 128,
	num_iters = 100,
	checkpoint_interval = 500,
	# FIX: Enable reward learning
	intermediate_rewards = false,
	results_path = mkpath(joinpath(GAME_DIR, "results")),
	networks_path = mkpath(joinpath(GAME_DIR, "networks")),
)

const hyper = FeedForwardHP(
	width_hidden = 64,  # Increased from 16 to 64 for better learning capacity          
	depth_representation = 4,
	depth_prediction = 4,
	depth_dynamics = 4,
	depth_policy = 2,
	depth_value = 2,
	depth_reward = 2,
	depth_state_head = 2,
	use_batch_norm = true,     # ← これを追加！
	hidden_state_size = 126, # 6*7*3 = 126      
	reward_activation = tanh,
)

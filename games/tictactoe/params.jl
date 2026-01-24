# Use explicit paths relative to this file
# This ensures data is saved in MuZero.jl/games/tictactoe/results/ even if run from root
const GAME_DIR = @__DIR__

const conf = Config(
	observation_shape = (3, 3, 3),
	action_space = collect(1:9),
	players = collect(1:2),
	stacked_observations = 1,
	num_workers = 2,
	max_moves = 9,
	num_unroll_steps = 5,
	td_steps = 5,
	PER = false,
	opponent = "human",
	training_steps = 1000,
	batch_size = 32,
	num_iters = 50,
	intermediate_rewards = true, # Keep enabled
	results_path = mkpath(joinpath(GAME_DIR, "results")),
	networks_path = mkpath(joinpath(GAME_DIR, "networks")),
)

const hyper = FeedForwardHP(
	width_hidden = 64, # Increased from 16 to 64
	depth_representation = 3, # Increased depth slightly
	depth_prediction = 3,
	depth_dynamics = 3,
	depth_policy = 2,
	depth_value = 2,
	depth_reward = 2,
	depth_state_head = 2,
	hidden_state_size = 27,
	reward_activation = tanh,
)

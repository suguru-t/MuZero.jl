using Dates
using Distributed
using Flux
using JLD2
using Random
using ReinforcementLearningBase

const SRC_DIR = joinpath(@__DIR__, "../../src")

include(joinpath(SRC_DIR, "Constructors.jl"))
include(joinpath(SRC_DIR, "SelfPlay.jl"))
include(joinpath(SRC_DIR, "ReplayBuffer.jl"))
include(joinpath(SRC_DIR, "Learning.jl"))
include("game.jl")
include("params.jl")

function parse_args(args)
	options = Dict(
		"games" => "100",
		"step" => "all",
		"num-iters" => string(conf.num_iters),
		"muzero-player" => "alternate",
		"seed" => "1337",
		"outdir" => joinpath(conf.results_path, "evaluation"),
		"save-details" => "false",
	)

	i = 1
	while i <= length(args)
		arg = args[i]
		if startswith(arg, "--")
			key = arg[3:end]
			if key == "help"
				options["help"] = "true"
				i += 1
			else
				i == length(args) && error("Missing value for --$key")
				options[key] = args[i + 1]
				i += 2
			end
		else
			error("Unexpected argument: $arg")
		end
	end
	return options
end

function print_help()
	println("""
	Usage:
	  julia --project games/connect4/evaluate.jl [options]

	Options:
	  --games N                  Number of evaluation games. Default: 100
	  --step latest|N|all|A,B    Checkpoint(s) to load. Default: all
	  --num-iters N              MCTS simulations per MuZero move. Default: conf.num_iters
	  --muzero-player MODE       1, 2, or alternate. Default: alternate
	  --seed N                   RNG seed. Default: 1337
	  --outdir PATH              Output directory. Default: games/connect4/results/evaluation
	  --save-details true|false  Save per-game CSV/SVG per checkpoint. Default: false
	""")
end

function parse_bool(value::String)
	lower = lowercase(value)
	lower in ("true", "1", "yes", "y") && return true
	lower in ("false", "0", "no", "n") && return false
	error("Expected true or false, got: $value")
end

function read_checkpoint_step(path::String, fallback_step::Int)
	model_data = load(path)
	step = haskey(model_data, "step") ? Int(model_data["step"]) : fallback_step
	return step
end

function checkpoint_path(conf, step_arg::String)
	filename = step_arg == "latest" || step_arg == "0" ? "latest_checkpoint.jld2" : "$(parse(Int, step_arg))_checkpoint.jld2"
	return joinpath(conf.networks_path, filename)
end

function checkpoint_specs(conf, step_arg::String)
	if step_arg == "all"
		specs = NamedTuple[]
		for filename in readdir(conf.networks_path)
			m = match(r"^(\d+)_checkpoint\.jld2$", filename)
			isnothing(m) && continue
			step = parse(Int, m.captures[1])
			push!(specs, (label = string(step), requested_step = step, path = joinpath(conf.networks_path, filename)))
		end
		sort!(specs, by = x -> x.requested_step)

		latest_path = joinpath(conf.networks_path, "latest_checkpoint.jld2")
		if isfile(latest_path)
			latest_step = read_checkpoint_step(latest_path, 0)
			if !any(spec -> spec.requested_step == latest_step, specs)
				push!(specs, (label = "latest", requested_step = latest_step, path = latest_path))
			end
		end
		isempty(specs) && error("No checkpoint files found in $(conf.networks_path)")
		return specs
	elseif occursin(",", step_arg)
		return [only(checkpoint_specs(conf, String(strip(part)))) for part in split(step_arg, ",")]
	else
		path = checkpoint_path(conf, step_arg)
		isfile(path) || error("Checkpoint not found: $path")
		fallback_step = step_arg == "latest" || step_arg == "0" ? 0 : parse(Int, step_arg)
		actual_step = read_checkpoint_step(path, fallback_step)
		return [(label = step_arg, requested_step = actual_step, path = path)]
	end
end

function load_networks(path::String)
	isfile(path) || error("Checkpoint not found: $path")

	model_data = load(path)
	return set_inference_mode!((
		representation = model_data["representation"],
		prediction = model_data["prediction"],
		dynamics = model_data["dynamics"],
	))
end

function choose_random_action(env::Connect4, rng::AbstractRNG)
	p = ReinforcementLearningBase.current_player(env)
	legal_actions = ReinforcementLearningBase.legal_action_space(env, p)
	isempty(legal_actions) && error("Random agent received no legal actions.")
	return rand(rng, legal_actions)
end

function choose_muzero_action(env::Connect4, history::GameHistory, observation, NNs, eval_conf::Config)
	p = ReinforcementLearningBase.current_player(env)
	history.observation_history = cat(history.observation_history, observation, dims = 4)
	stacked_observations = get_stacked_observations(
		history,
		lastindex(history.observation_history, 4),
		eval_conf.stacked_observations,
		eval_conf,
	)
	legal_actions = ReinforcementLearningBase.legal_action_space(env, p)
	root = run_mcts(env, stacked_observations, legal_actions, p, false, NNs, eval_conf)
	return select_action(root, 0.0f0)
end

function resolve_muzero_player(mode::String, game_index::Int)
	if mode == "alternate"
		return isodd(game_index) ? 1 : 2
	elseif mode == "1" || mode == "2"
		return parse(Int, mode)
	else
		error("--muzero-player must be 1, 2, or alternate. Got: $mode")
	end
end

function play_eval_game(NNs, eval_conf::Config, rng::AbstractRNG, muzero_player::Int)
	env = Connect4()
	observation = ReinforcementLearningBase.reset!(env)
	history = GameHistory(
		Array{Float32}(undef, eval_conf.observation_shape..., 0),
		Vector{Int}(),
		Vector{Float32}(),
		Vector{Int}(),
		Matrix{Float32}(undef, length(eval_conf.action_space), 0),
		Vector{Float32}(),
		nothing,
		nothing,
		nothing,
	)

	while !ReinforcementLearningBase.is_terminated(env) && length(history.action_history) < eval_conf.max_moves
		p = ReinforcementLearningBase.current_player(env)
		action = if p == muzero_player
			choose_muzero_action(env, history, observation, NNs, eval_conf)
		else
			choose_random_action(env, rng)
		end

		observation = env(action)
		push!(history.action_history, action)
		push!(history.to_play_history, p)
	end

	winner = env.winner
	result = winner === nothing ? 0 : (winner == muzero_player ? 1 : -1)
	return (
		winner = winner === nothing ? 0 : winner,
		result = result,
		moves = length(history.action_history),
	)
end

function write_csv(path, rows)
	open(path, "w") do io
		println(io, "game,muzero_player,winner,result,cumulative_win_rate,cumulative_draw_rate,cumulative_loss_rate,moves")
		for row in rows
			println(io, join((
				row.game,
				row.muzero_player,
				row.winner,
				row.result,
				row.cumulative_win_rate,
				row.cumulative_draw_rate,
				row.cumulative_loss_rate,
				row.moves,
			), ","))
		end
	end
end

function write_checkpoint_summary_csv(path, summaries)
	open(path, "w") do io
		println(io, "checkpoint_label,checkpoint_step,games,wins,win_rate")
		for summary in summaries
			println(io, join((
				summary.label,
				summary.step,
				summary.games,
				summary.wins,
				summary.win_rate,
			), ","))
		end
	end
end

function svg_polyline(points)
	return join(["$(round(x, digits = 2)),$(round(y, digits = 2))" for (x, y) in points], " ")
end

function axis_tick_indices(n::Int; max_ticks::Int = 8)
	n <= 0 && return Int[]
	n <= max_ticks && return collect(1:n)
	return unique(round.(Int, range(1, n; length = max_ticks)))
end

function write_winrate_svg(path, rows; width = 900, height = 520)
	margin_left = 70
	margin_right = 30
	margin_top = 35
	margin_bottom = 65
	plot_w = width - margin_left - margin_right
	plot_h = height - margin_top - margin_bottom
	n = length(rows)

	function x_for(i)
		n <= 1 && return margin_left
		return margin_left + (i - 1) * plot_w / (n - 1)
	end

	function y_for(rate)
		return margin_top + (1 - rate) * plot_h
	end

	win_points = [(x_for(i), y_for(rows[i].cumulative_win_rate)) for i in 1:n]
	draw_points = [(x_for(i), y_for(rows[i].cumulative_draw_rate)) for i in 1:n]
	loss_points = [(x_for(i), y_for(rows[i].cumulative_loss_rate)) for i in 1:n]
	final_win = isempty(rows) ? 0.0 : rows[end].cumulative_win_rate
	final_draw = isempty(rows) ? 0.0 : rows[end].cumulative_draw_rate
	final_loss = isempty(rows) ? 0.0 : rows[end].cumulative_loss_rate

	open(path, "w") do io
		println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
		println(io, """<rect width="100%" height="100%" fill="white"/>""")
		println(io, """<text x="$(width / 2)" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Connect4 Evaluation vs Random</text>""")
		println(io, """<line x1="$margin_left" y1="$margin_top" x2="$margin_left" y2="$(margin_top + plot_h)" stroke="#333"/>""")
		println(io, """<line x1="$margin_left" y1="$(margin_top + plot_h)" x2="$(margin_left + plot_w)" y2="$(margin_top + plot_h)" stroke="#333"/>""")

		for tick in 0:0.25:1.0
			y = y_for(tick)
			println(io, """<line x1="$margin_left" y1="$y" x2="$(margin_left + plot_w)" y2="$y" stroke="#e6e6e6"/>""")
			println(io, """<text x="$(margin_left - 10)" y="$(y + 4)" text-anchor="end" font-family="sans-serif" font-size="12">$(round(tick * 100; digits = 0))%</text>""")
		end

		baseline_y = y_for(0.5)
		println(io, """<line x1="$margin_left" y1="$baseline_y" x2="$(margin_left + plot_w)" y2="$baseline_y" stroke="#999" stroke-dasharray="4 4"/>""")
		println(io, """<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="$(svg_polyline(win_points))"/>""")
		println(io, """<polyline fill="none" stroke="#2ca02c" stroke-width="2" points="$(svg_polyline(draw_points))"/>""")
		println(io, """<polyline fill="none" stroke="#d62728" stroke-width="2" points="$(svg_polyline(loss_points))"/>""")

		println(io, """<text x="$(margin_left + plot_w / 2)" y="$(height - 20)" text-anchor="middle" font-family="sans-serif" font-size="13">Evaluation game</text>""")
		println(io, """<text x="20" y="$(margin_top + plot_h / 2)" transform="rotate(-90 20,$(margin_top + plot_h / 2))" text-anchor="middle" font-family="sans-serif" font-size="13">Rate</text>""")
		println(io, """<text x="$(margin_left + plot_w - 210)" y="55" font-family="sans-serif" font-size="13" fill="#1f77b4">Win: $(round(final_win * 100; digits = 1))%</text>""")
		println(io, """<text x="$(margin_left + plot_w - 210)" y="75" font-family="sans-serif" font-size="13" fill="#2ca02c">Draw: $(round(final_draw * 100; digits = 1))%</text>""")
		println(io, """<text x="$(margin_left + plot_w - 210)" y="95" font-family="sans-serif" font-size="13" fill="#d62728">Loss: $(round(final_loss * 100; digits = 1))%</text>""")
		println(io, """</svg>""")
	end
end

function write_checkpoint_winrate_svg(path, summaries; width = 900, height = 520)
	margin_left = 80
	margin_right = 30
	margin_top = 35
	margin_bottom = 65
	plot_w = width - margin_left - margin_right
	plot_h = height - margin_top - margin_bottom
	n = length(summaries)
	steps = [summary.step for summary in summaries]
	min_step = minimum(steps)
	max_step = maximum(steps)

	function x_for(step)
		max_step == min_step && return margin_left + plot_w / 2
		return margin_left + (step - min_step) * plot_w / (max_step - min_step)
	end

	function y_for(rate)
		return margin_top + (1 - rate) * plot_h
	end

	win_points = [(x_for(summary.step), y_for(summary.win_rate)) for summary in summaries]

	open(path, "w") do io
		println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
		println(io, """<rect width="100%" height="100%" fill="white"/>""")
		println(io, """<text x="$(width / 2)" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Average Win Rate vs Random</text>""")
		println(io, """<line x1="$margin_left" y1="$margin_top" x2="$margin_left" y2="$(margin_top + plot_h)" stroke="#333"/>""")
		println(io, """<line x1="$margin_left" y1="$(margin_top + plot_h)" x2="$(margin_left + plot_w)" y2="$(margin_top + plot_h)" stroke="#333"/>""")

		for tick in 0:0.25:1.0
			y = y_for(tick)
			println(io, """<line x1="$margin_left" y1="$y" x2="$(margin_left + plot_w)" y2="$y" stroke="#e6e6e6"/>""")
			println(io, """<text x="$(margin_left - 10)" y="$(y + 4)" text-anchor="end" font-family="sans-serif" font-size="12">$(round(tick * 100; digits = 0))%</text>""")
		end

		for i in axis_tick_indices(length(summaries))
			step = summaries[i].step
			x = x_for(step)
			println(io, """<line x1="$x" y1="$(margin_top + plot_h)" x2="$x" y2="$(margin_top + plot_h + 6)" stroke="#333"/>""")
			println(io, """<text x="$x" y="$(margin_top + plot_h + 22)" text-anchor="middle" font-family="sans-serif" font-size="11">$step</text>""")
		end

		baseline_y = y_for(0.5)
		println(io, """<line x1="$margin_left" y1="$baseline_y" x2="$(margin_left + plot_w)" y2="$baseline_y" stroke="#999" stroke-dasharray="4 4"/>""")
		println(io, """<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="$(svg_polyline(win_points))"/>""")

		for summary in summaries
			x = x_for(summary.step)
			y = y_for(summary.win_rate)
			println(io, """<circle cx="$x" cy="$y" r="3" fill="#1f77b4"><title>step $(summary.step): win $(round(summary.win_rate * 100; digits = 1))%</title></circle>""")
		end

		println(io, """<text x="$(margin_left + plot_w / 2)" y="$(height - 20)" text-anchor="middle" font-family="sans-serif" font-size="13">Training step</text>""")
		println(io, """<text x="20" y="$(margin_top + plot_h / 2)" transform="rotate(-90 20,$(margin_top + plot_h / 2))" text-anchor="middle" font-family="sans-serif" font-size="13">Average win rate</text>""")
		println(io, """<text x="$(margin_left + plot_w - 210)" y="55" font-family="sans-serif" font-size="13" fill="#1f77b4">MuZero win rate</text>""")
		println(io, """</svg>""")
	end
end

function checkpoint_dir_name(label, step)
	safe_label = replace(label, r"[^A-Za-z0-9_-]" => "_")
	return "checkpoint_$(safe_label)_step_$(step)"
end

function evaluate_checkpoint(spec, eval_conf::Config, options, base_seed::Int, outdir::String, save_details::Bool)
	rng = MersenneTwister(base_seed)
	NNs = load_networks(spec.path)
	num_games = parse(Int, options["games"])
	rows = NamedTuple[]
	wins = 0
	draws = 0
	losses = 0
	total_moves = 0

	for game_index in 1:num_games
		muzero_player = resolve_muzero_player(options["muzero-player"], game_index)
		outcome = play_eval_game(NNs, eval_conf, rng, muzero_player)
		total_moves += outcome.moves
		if outcome.result == 1
			wins += 1
		elseif outcome.result == 0
			draws += 1
		else
			losses += 1
		end

		push!(rows, (
			game = game_index,
			muzero_player = muzero_player,
			winner = outcome.winner,
			result = outcome.result,
			cumulative_win_rate = wins / game_index,
			cumulative_draw_rate = draws / game_index,
			cumulative_loss_rate = losses / game_index,
			moves = outcome.moves,
		))

	end

	if save_details
		checkpoint_outdir = mkpath(joinpath(outdir, "checkpoints", checkpoint_dir_name(spec.label, spec.requested_step)))
		write_csv(joinpath(checkpoint_outdir, "evaluation.csv"), rows)
		write_winrate_svg(joinpath(checkpoint_outdir, "winrate.svg"), rows)
		open(joinpath(checkpoint_outdir, "summary.txt"), "w") do io
			println(io, "checkpoint=$(spec.label)")
			println(io, "checkpoint_step=$(spec.requested_step)")
			println(io, "checkpoint_path=$(spec.path)")
			println(io, "games=$num_games")
			println(io, "num_iters=$(options["num-iters"])")
			println(io, "muzero_player=$(options["muzero-player"])")
			println(io, "wins=$wins")
			println(io, "draws=$draws")
			println(io, "losses=$losses")
			println(io, "win_rate=$(wins / num_games)")
			println(io, "draw_rate=$(draws / num_games)")
			println(io, "loss_rate=$(losses / num_games)")
			println(io, "average_moves=$(total_moves / num_games)")
		end
	end

	return (
		label = spec.label,
		step = spec.requested_step,
		games = num_games,
		wins = wins,
		draws = draws,
		losses = losses,
		win_rate = wins / num_games,
		draw_rate = draws / num_games,
		loss_rate = losses / num_games,
		average_moves = total_moves / num_games,
	)
end

function main()
	options = parse_args(ARGS)
	if get(options, "help", "false") == "true"
		print_help()
		return
	end

	num_games = parse(Int, options["games"])
	num_games > 0 || error("--games must be positive")
	num_iters = parse(Int, options["num-iters"])
	num_iters > 0 || error("--num-iters must be positive")
	base_seed = parse(Int, options["seed"])
	save_details = parse_bool(options["save-details"])
	eval_conf = Config(conf; num_iters = num_iters, allow_nonfinite_mcts = true)
	specs = checkpoint_specs(eval_conf, options["step"])

	timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	outdir = mkpath(joinpath(options["outdir"], timestamp))

	println("Evaluating Connect4 MuZero vs random: checkpoints=$(length(specs)), games=$num_games")

	summaries = NamedTuple[]
	for spec in specs
		push!(summaries, evaluate_checkpoint(spec, eval_conf, options, base_seed, outdir, save_details))
	end

	summary_csv_path = joinpath(outdir, "checkpoint_summary.csv")
	summary_svg_path = joinpath(outdir, "checkpoint_winrate.svg")
	write_checkpoint_summary_csv(summary_csv_path, summaries)
	write_checkpoint_winrate_svg(summary_svg_path, summaries)

	println("step,win_rate")
	for summary in summaries
		println("$(summary.step),$(round(summary.win_rate * 100; digits = 1))% ($(summary.wins)/$(summary.games))")
	end

	println("Summary CSV: $summary_csv_path")
	println("Summary SVG: $summary_svg_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end

using ReinforcementLearningBase

mutable struct Connect4 <: AbstractEnv
    board::Array{Int, 2} # 6 rows x 7 cols. 0=empty, 1=player1, 2=player2
    player::Int
    winner::Union{Nothing, Int}
end

function Connect4()
    return Connect4(zeros(Int, 6, 7), 1, nothing)
end

function RLBase.reset!(env::Connect4)
    fill!(env.board, 0)
    env.player = 1
    env.winner = nothing
    return get_observation(env)
end

RLBase.action_space(::Connect4) = Base.OneTo(7)

function RLBase.legal_action_space(env::Connect4, p)
    return findall(c -> env.board[6, c] == 0, 1:7)
end

function RLBase.legal_action_space_mask(env::Connect4, p)
    mask = zeros(Bool, 7)
    for c in 1:7
        if env.board[6, c] == 0
            mask[c] = true
        end
    end
    return mask
end

function (env::Connect4)(col::Int)
    if env.winner !== nothing
        return get_observation(env)
    end

    # Find the first empty row in the selected column
    row = findfirst(x -> x == 0, env.board[:, col])
    
    if row === nothing
        # Illegal move (column full), but RL envs usually assume valid actions.
        # If MuZero picks an illegal move, MCTS should filter it out via legal_action_space.
        return get_observation(env)
    end

    env.board[row, col] = env.player
    
    if check_win(env, row, col)
        env.winner = env.player
    else
        env.player = env.player == 1 ? 2 : 1
    end
    
    return get_observation(env)
end

RLBase.current_player(env::Connect4) = env.player
RLBase.players(env::Connect4) = (1, 2)

# Representation for MuZero: 6x7x3
# Plane 1: Current player's pieces
# Plane 2: Opponent's pieces
# Plane 3: Color (Who's turn is it? All 1s for P1, All 0s for P2)
function get_observation(env::Connect4)
    obs = zeros(Float32, 6, 7, 3)
    p = env.player
    opp = p == 1 ? 2 : 1
    
    # Plane 1: Player's pieces
    obs[:, :, 1] .= (env.board .== p)
    
    # Plane 2: Opponent's pieces
    obs[:, :, 2] .= (env.board .== opp)
    
    # Plane 3: To-play indicator
    if p == 1
        obs[:, :, 3] .= 1.0
    end
    # Implicitly 0.0 if p == 2
    
    return obs
end

# Define StateStyle to tell RLBase we use Array{Float32, 3} observations
RLBase.StateStyle(::Connect4) = Observation{Array{Float32,3}}()

RLBase.state(env::Connect4, ::Observation{Array{Float32,3}}, p) = get_observation(env)
RLBase.state_space(env::Connect4, ::Observation{Array{Float32,3}}, p) = Space(fill(0.0..1.0, 6, 7, 3))

# Fallback for Observation{Any} which caused the original error
RLBase.state(env::Connect4, ::Observation{Any}, p) = get_observation(env)

RLBase.is_terminated(env::Connect4) = env.winner !== nothing || all(env.board .!= 0)

function RLBase.reward(env::Connect4, player)
    if env.winner === nothing
        return 0.0
    elseif env.winner === player
        return 1.0
    else
        return -1.0
    end
end

# Check for 4 in a row
function check_win(env::Connect4, r, c)
    p = env.board[r, c]
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Horizontal, Vertical, Diagonal /, Diagonal \

    for (dr, dc) in directions
        count = 1
        # Check forward
        for k in 1:3
            nr, nc = r + k*dr, c + k*dc
            if 1 <= nr <= 6 && 1 <= nc <= 7 && env.board[nr, nc] == p
                count += 1
            else
                break
            end
        end
        # Check backward
        for k in 1:3
            nr, nc = r - k*dr, c - k*dc
            if 1 <= nr <= 6 && 1 <= nc <= 7 && env.board[nr, nc] == p
                count += 1
            else
                break
            end
        end
        
        if count >= 4
            return true
        end
    end
    return false
end

using Crayons

player_color(p) = p == 1 ? crayon"light_red" : crayon"light_yellow"
player_mark(p) = p == 1 ? "O" : "X"

function render_game(env::Connect4)
    print("\n 1 2 3 4 5 6 7\n")
    for r in 6:-1:1
        print("|")
        for c in 1:7
            v = env.board[r, c]
            if v == 0
                print(" ")
            else
                print(player_color(v), player_mark(v), crayon"reset")
            end
            print("|")
        end
        print("\n")
    end
    print(" ---------------\n")
    if env.winner !== nothing
        println("Winner: Player $(env.winner)")
    else
        println("Current Player: $(env.player)")
    end
end

function human_input()::Int
    print("Enter column (1-7): ")
    try
        action_taken = parse(Int, readline())
        return action_taken
    catch
        return 1
    end
end

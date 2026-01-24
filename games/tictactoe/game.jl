using ReinforcementLearningBase

mutable struct TicTacToe <: AbstractEnv
    board::BitArray{3}
    player::Int
end

function TicTacToe()
    board = BitArray{3}(undef, 3, 3, 3)
    fill!(board, false)
    board[:, :, 3] .= true
    return TicTacToe(board, 1)
end

function RLBase.reset!(env::TicTacToe)
    fill!(env.board, false)
    env.board[:, :, 3] .= true
    env.player = 1
    return env.board
end

Base.hash(env::TicTacToe, h::UInt) = hash(env.board, h)
Base.isequal(a::TicTacToe, b::TicTacToe) = isequal(a.board, b.board)

RLBase.action_space(::TicTacToe) = Base.OneTo(9)

RLBase.legal_action_space(env::TicTacToe, p) = findall(legal_action_space_mask(env))

function RLBase.legal_action_space_mask(env::TicTacToe, p=nothing)
    if is_win(env, 1) || is_win(env, 2)
        zeros(Bool, 9)
    else
        vec(view(env.board, :, :, 3))
    end
end

(env::TicTacToe)(action::Int) = env(CartesianIndices((3, 3))[action])

function (env::TicTacToe)(action::CartesianIndex{2})
    env.board[action, 3] = false
    env.board[action, env.player] = true
    env.player = mod1(env.player + 1, 2)
    return env.board
end

RLBase.current_player(env::TicTacToe) = env.player
RLBase.players(env::TicTacToe) = (1, 2)

RLBase.state(env::TicTacToe, ::Observation{BitArray{3}}, p) = env.board
RLBase.state_space(env::TicTacToe, ::Observation{BitArray{3}}, p) =
    Space(fill(p, 3, 3, 3))

# Dummy implementations for Observation types we don't strictly need for MuZero but RLBase might query
RLBase.state(env::TicTacToe, ::Observation{Int}, p) = 0
RLBase.state_space(env::TicTacToe, ::Observation{Int}, p) = Base.OneTo(1)

RLBase.state_space(env::TicTacToe, ::Observation{String}, p) = WorldSpace{String}()

function RLBase.state(env::TicTacToe, ::Observation{String}, p)
    buff = IOBuffer()
    for i in 1:3
        for j in 1:3
            if env.board[i, j, 1]
                x = 'o'
            elseif env.board[i, j, 2]
                x = 'x'
            else
                x = '.'
            end
            print(buff, x)
        end
        print(buff, '\n')
    end
    String(take!(buff))
end

function RLBase.is_terminated(env::TicTacToe)
    # Terminated if P1 won, P2 won, or no empty spots left (plane 3)
    return is_win(env, 1) || is_win(env, 2) || !any(view(env.board, :, :, 3))
end

function RLBase.reward(env::TicTacToe, player)
    if is_win(env, player)
        return 1.0
    elseif is_win(env, player == 1 ? 2 : 1)
        return -1.0
    else
        return 0.0
    end
end

function is_win(env::TicTacToe, player)
    b = env.board
    p = player
    @inbounds begin
        # Rows
        (b[1, 1, p] & b[1, 2, p] & b[1, 3, p]) ||
        (b[2, 1, p] & b[2, 2, p] & b[2, 3, p]) ||
        (b[3, 1, p] & b[3, 2, p] & b[3, 3, p]) ||
        # Cols
        (b[1, 1, p] & b[2, 1, p] & b[3, 1, p]) ||
        (b[1, 2, p] & b[2, 2, p] & b[3, 2, p]) ||
        (b[1, 3, p] & b[2, 3, p] & b[3, 3, p]) ||
        # Diagonals
        (b[1, 1, p] & b[2, 2, p] & b[3, 3, p]) ||
        (b[1, 3, p] & b[2, 2, p] & b[3, 1, p])
    end
end

RLBase.NumAgentStyle(::TicTacToe) = MultiAgent(2)
RLBase.DynamicStyle(::TicTacToe) = SEQUENTIAL
RLBase.ActionStyle(::TicTacToe) = FULL_ACTION_SET
RLBase.InformationStyle(::TicTacToe) = PERFECT_INFORMATION
RLBase.StateStyle(::TicTacToe) =
    (Observation{String}(), Observation{Int}(), Observation{BitArray{3}}())
RLBase.RewardStyle(::TicTacToe) = TERMINAL_REWARD
RLBase.UtilityStyle(::TicTacToe) = ZERO_SUM
RLBase.ChanceStyle(::TicTacToe) = DETERMINISTIC

using Crayons

player_color(p) = p == 1 ? crayon"light_red" : crayon"light_blue"
player_name(p) = p == 1 ? "1" : "2"
player_mark(p) = p == 1 ? "o" : "x"

function render_game(env::AbstractEnv; with_position_names=true, botmargin=true)
    p = current_player(env)
    pname = player_name(p)
    pcol = player_color(p)
    print(pcol, pname, " plays:", crayon"reset", "\n\n")
    board_red = env.board[:, :, 1]
    board_blue = env.board[:, :, 2]
    indices = LinearIndices(board_blue)
    for x in 1:3
        for y in 1:3
            if board_red[x, y] == 0 && board_blue[x, y] == 0
                print(".")
            elseif board_red[x, y] == 1
                print(player_color(1), player_mark(1), crayon"reset")
            elseif board_blue[x, y] == 1
                print(player_color(2), player_mark(2), crayon"reset")
            end

            print(" ")
        end
        if with_position_names
            print(" | ")
            for y in 1:3
                print("$(indices[CartesianIndex(x,y)]) ")
            end
        end
        print("\n")
    end
    botmargin && print("\n")
end

function human_input()::Int
    print("Please take an action:\n")
    action_taken = parse(Int, readline())
    return action_taken
end
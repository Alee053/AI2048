#include "ExpectimaxSearcher.h"
#include "RandomUtil.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

ExpectimaxSearcher::ExpectimaxSearcher() = default;

float get_log_reward(int merge_score) {
    if (merge_score <= 0) return 0.0f;
    return std::log2(static_cast<float>(merge_score));
}

void ExpectimaxSearcher::gather_leaves(const Board& board, int depth, uint64_t board_hash,
                                       std::vector<Board>& leaves_queue,
                                       std::map<int, std::unordered_set<Board, BoardHash>>& visited) {
    if (depth == 0) {
        if (visited[depth].insert(board).second) {
            leaves_queue.push_back(board);
        }
        return;
    }

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);

        auto [ms, moved] = game_instance.move_simulated(move);
        if (!moved) continue;

        Board post_move_board = game_instance.get_board();

        std::vector<std::pair<int, int>> empty_cells;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if (post_move_board[r][c] == 0) empty_cells.push_back({r, c});
            }
        }

        if (empty_cells.empty()) {
            if (visited[depth].insert(post_move_board).second) {
                leaves_queue.push_back(post_move_board);
            }
            continue;
        }

        for (const auto& cell : empty_cells) {
            Board next_board_2 = post_move_board;
            next_board_2[cell.first][cell.second] = 1;
            uint64_t hash_2 = RandomUtil::compute_board_hash(next_board_2);
            gather_leaves(next_board_2, depth - 1, hash_2, leaves_queue, visited);

            Board next_board_4 = post_move_board;
            next_board_4[cell.first][cell.second] = 2;
            uint64_t hash_4 = RandomUtil::compute_board_hash(next_board_4);
            gather_leaves(next_board_4, depth - 1, hash_4, leaves_queue, visited);
        }
    }
}

float ExpectimaxSearcher::chance_node_substitute(const Board& board, int depth,
                                                 const std::unordered_map<Board, float, BoardHash>& leaf_cache,
                                                 float alpha, float beta) {
    std::vector<std::pair<int, int>> empty_cells;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) if (board[r][c] == 0) empty_cells.push_back({r, c});

    if (empty_cells.empty()) {
        return max_node_substitute(board, depth - 1, RandomUtil::compute_board_hash(board), leaf_cache, alpha, beta);
    }

    float total_value = 0;
    for (const auto& cell : empty_cells) {
        Board next_board2 = board;
        next_board2[cell.first][cell.second] = 1;
        total_value += 0.9f * max_node_substitute(next_board2, depth - 1, RandomUtil::compute_board_hash(next_board2), leaf_cache, alpha, beta);

        Board next_board4 = board;
        next_board4[cell.first][cell.second] = 2;
        total_value += 0.1f * max_node_substitute(next_board4, depth - 1, RandomUtil::compute_board_hash(next_board4), leaf_cache, alpha, beta);
    }
    return total_value / (2 * empty_cells.size());
}

float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                              const std::unordered_map<Board, float, BoardHash>& leaf_cache,
                                              float alpha, float beta) {
    if (depth == 0) {
        auto it = leaf_cache.find(board);
        return it != leaf_cache.end() ? it->second : -100.0f;
    }

    // TT lookup — uses board_hash only (depth not part of key)
    auto tp_it = transposition_table.find(board_hash);
    if (tp_it != transposition_table.end()) {
        return tp_it->second;
    }

    float max_value = -1e9;
    bool any_move_possible = false;

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [merge_score, moved] = game_instance.move_simulated(move);

        if (!moved) continue;
        any_move_possible = true;

        float immediate_reward = get_log_reward(merge_score);
        Board child_board = game_instance.get_board();
        uint64_t child_hash = RandomUtil::compute_board_hash(child_board);

        float future_value = chance_node_substitute(child_board, depth, leaf_cache, alpha, beta);

        float total_value = immediate_reward + future_value;
        if (total_value > max_value) {
            max_value = total_value;
        }

        // Alpha-beta pruning: update alpha, prune if >= beta
        if (max_value >= beta) {
            break;  // prune remaining branches
        }
        alpha = (max_value > alpha) ? max_value : alpha;
    }

    float result = any_move_possible ? max_value : 0.0f;
    transposition_table[board_hash] = result;
    return result;
}

int ExpectimaxSearcher::find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func) {
    transposition_table.clear();

    uint64_t root_hash = RandomUtil::compute_board_hash(board);

    // --- Move ordering: evaluate immediate post-move boards with CNN ---
    struct RootMove {
        int move_id;
        Board post_board;
        uint64_t post_hash;
        float immediate_reward;
        float post_board_eval;  // CNN evaluation for ordering
    };
    std::vector<RootMove> root_moves;
    root_moves.reserve(4);

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [merge_score, moved] = game_instance.move_simulated(move);
        if (!moved) continue;
        root_moves.push_back({move, game_instance.get_board(), RandomUtil::compute_board_hash(game_instance.get_board()), get_log_reward(merge_score), 0.0f});
    }

    if (root_moves.empty()) return 0;

    // --- Pre-eval root post-move boards for ordering ---
    if (root_moves.size() > 1) {
        std::vector<Board> boards_to_eval;
        for (size_t i = 0; i < root_moves.size(); ++i) {
            boards_to_eval.push_back(root_moves[i].post_board);
        }
        std::vector<float> evals = batch_eval_func(boards_to_eval);
        for (size_t i = 0; i < root_moves.size(); ++i) {
            root_moves[i].post_board_eval = evals[i];
        }
        // Sort by CNN evaluation descending (best first for alpha-beta pruning)
        std::sort(root_moves.begin(), root_moves.end(), [](const RootMove& a, const RootMove& b) {
            return a.post_board_eval > b.post_board_eval;
        });
    }

    // --- DFS with batched leaf evaluation ---
    float best_score = -1e9f;
    int best_move = root_moves[0].move_id;
    float global_alpha = -1e9f;

    for (const auto& rm : root_moves) {
        // For each root move, gather its leaves and evaluate in batches
        std::vector<Board> move_leaves;
        std::map<int, std::unordered_set<Board, BoardHash>> move_visited;

        // Gather leaves for this root move only
        gather_leaves(rm.post_board, depth, rm.post_hash, move_leaves, move_visited);

        // Batch evaluate
        std::unordered_map<Board, float, BoardHash> leaf_cache;
        for (size_t i = 0; i < move_leaves.size(); i += BATCH_SIZE) {
            size_t end = std::min(i + BATCH_SIZE, move_leaves.size());
            std::vector<Board> batch(move_leaves.begin() + i, move_leaves.begin() + end);
            std::vector<float> batch_evals = batch_eval_func(batch);
            for (size_t j = 0; j < batch.size(); ++j) {
                leaf_cache[batch[j]] = batch_evals[j];
            }
        }

        // Evaluate this root move using the populated leaf_cache
        float future_value = max_node_substitute(rm.post_board, depth, rm.post_hash, leaf_cache, global_alpha, 1e9f);
        float total_score = rm.immediate_reward + future_value;

        if (total_score > best_score) {
            best_score = total_score;
            best_move = rm.move_id;
        }
        global_alpha = std::max(global_alpha, total_score);
    }

    return best_move;
}

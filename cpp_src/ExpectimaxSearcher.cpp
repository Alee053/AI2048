#include "ExpectimaxSearcher.h"
#include "RandomUtil.h"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>

ExpectimaxSearcher::ExpectimaxSearcher() = default;

float get_log_reward(int merge_score) {
    if (merge_score <= 0) return 0.0f;
    return std::log2(static_cast<float>(merge_score));
}

void ExpectimaxSearcher::gather_leaves(const Board& board, int depth, uint64_t /*board_hash*/,
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

float ExpectimaxSearcher::chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                                 const std::unordered_map<Board, float, BoardHash>& leaf_cache,
                                                 float /*alpha*/, float /*beta*/) {
    // Persistent TT lookup for chance nodes
    tt_lookups++;
    float cached_score = 0.0f;
    if (transposition_table.probe(board_hash, static_cast<uint8_t>(depth), NodeType::CHANCE, cached_score)) {
        tt_hits++;
        return cached_score;
    }

    std::vector<std::pair<int, int>> empty_cells;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) if (board[r][c] == 0) empty_cells.push_back({r, c});

    if (empty_cells.empty()) {
        float val = max_node_substitute(board, depth - 1, board_hash, leaf_cache, -1e9f, 1e9f);
        transposition_table.store(board_hash, static_cast<uint8_t>(depth), NodeType::CHANCE, val);
        return val;
    }

    float total_value = 0;
    for (const auto& cell : empty_cells) {
        // Place a '2' tile (value == 1 in log2 space)
        Board next_board_2 = board;
        next_board_2[cell.first][cell.second] = 1;
        uint64_t hash_2 = RandomUtil::update_board_hash(board_hash, cell.first, cell.second, 0, 1);
        float val_2 = max_node_substitute(next_board_2, depth - 1, hash_2, leaf_cache, -1e9f, 1e9f);
        total_value += 0.9f * val_2;

        // Place a '4' tile (value == 2 in log2 space)
        Board next_board_4 = board;
        next_board_4[cell.first][cell.second] = 2;
        uint64_t hash_4 = RandomUtil::update_board_hash(board_hash, cell.first, cell.second, 0, 2);
        float val_4 = max_node_substitute(next_board_4, depth - 1, hash_4, leaf_cache, -1e9f, 1e9f);
        total_value += 0.1f * val_4;
    }

    float result = total_value / (2.0f * empty_cells.size());
    transposition_table.store(board_hash, static_cast<uint8_t>(depth), NodeType::CHANCE, result);
    return result;
}

float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                              const std::unordered_map<Board, float, BoardHash>& leaf_cache,
                                              float alpha, float beta) {
    if (depth == 0) {
        auto it = leaf_cache.find(board);
        return it != leaf_cache.end() ? it->second : -100.0f;
    }

    // Persistent TT lookup for max nodes
    tt_lookups++;
    float cached_score = 0.0f;
    if (transposition_table.probe(board_hash, static_cast<uint8_t>(depth), NodeType::MAX, cached_score)) {
        tt_hits++;
        return cached_score;
    }

    float max_value = -1e9f;
    bool any_move_possible = false;

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [merge_score, moved] = game_instance.move_simulated(move);

        if (!moved) continue;
        any_move_possible = true;

        float immediate_reward = get_log_reward(merge_score);
        Board child_board = game_instance.get_board();
        uint64_t child_hash = RandomUtil::compute_board_hash(child_board);

        float future_value = chance_node_substitute(child_board, depth, child_hash, leaf_cache, alpha, beta);

        float total_value = immediate_reward + future_value;
        if (total_value > max_value) {
            max_value = total_value;
        }

        // Alpha-beta pruning
        if (max_value >= beta) {
            break;
        }
        alpha = (max_value > alpha) ? max_value : alpha;
    }

    float result = any_move_possible ? max_value : 0.0f;
    transposition_table.store(board_hash, static_cast<uint8_t>(depth), NodeType::MAX, result);
    return result;
}

SearchStats ExpectimaxSearcher::find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func) {
    // NOTE: we intentionally do NOT clear the transposition table here.
    // The persistent TT allows cached evaluations from previous turns to
    // grant "free" search depth on subsequent moves.
    tt_lookups = 0;
    tt_hits = 0;
    batches_eval = 0;
    search_start = std::chrono::high_resolution_clock::now();

    uint64_t root_hash = RandomUtil::compute_board_hash(board);

    // --- Move ordering: evaluate immediate post-move boards with CNN ---
    struct RootMove {
        int move_id;
        Board post_board;
        uint64_t post_hash;
        float immediate_reward;
        float post_board_eval;
    };
    std::vector<RootMove> root_moves;
    root_moves.reserve(4);

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [ms, moved] = game_instance.move_simulated(move);
        if (!moved) continue;
        root_moves.push_back({move, game_instance.get_board(), RandomUtil::compute_board_hash(game_instance.get_board()), get_log_reward(ms), 0.0f});
    }

    if (root_moves.empty()) {
        SearchStats empty_stats{};
        empty_stats.best_move = 0;
        empty_stats.think_ms = 0;
        empty_stats.nodes_visited = 0;
        empty_stats.batches_eval = 0;
        for (int i = 0; i < 4; ++i) empty_stats.move_scores[i] = -1e9f;
        empty_stats.tt_size = transposition_table.occupancy();
        empty_stats.tt_lookups = 0;
        empty_stats.tt_hits = 0;
        return empty_stats;
    }

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
        std::sort(root_moves.begin(), root_moves.end(), [](const RootMove& a, const RootMove& b) {
            return a.post_board_eval > b.post_board_eval;
        });
    }

    // --- DFS with batched leaf evaluation ---
    float best_score = -1e9f;
    int best_move = root_moves[0].move_id;
    float global_alpha = -1e9f;

    float move_scores[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
    size_t total_nodes_visited = 0;

    for (const auto& rm : root_moves) {
        std::vector<Board> move_leaves;
        std::map<int, std::unordered_set<Board, BoardHash>> move_visited;

        gather_leaves(rm.post_board, depth, rm.post_hash, move_leaves, move_visited);
        total_nodes_visited += move_leaves.size();

        std::unordered_map<Board, float, BoardHash> leaf_cache;
        for (size_t i = 0; i < move_leaves.size(); i += BATCH_SIZE) {
            size_t end = std::min(i + BATCH_SIZE, move_leaves.size());
            std::vector<Board> batch(move_leaves.begin() + i, move_leaves.begin() + end);
            std::vector<float> batch_evals = batch_eval_func(batch);
            batches_eval++;
            for (size_t j = 0; j < batch.size(); ++j) {
                leaf_cache[batch[j]] = batch_evals[j];
            }
        }

        float future_value = max_node_substitute(rm.post_board, depth, rm.post_hash, leaf_cache, global_alpha, 1e9f);
        float total_score = rm.immediate_reward + future_value;
        move_scores[rm.move_id] = total_score;

        if (total_score > best_score) {
            best_score = total_score;
            best_move = rm.move_id;
        }
        global_alpha = std::max(global_alpha, total_score);
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    double think_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

    SearchStats stats;
    stats.best_move = best_move;
    stats.think_ms = think_ms;
    stats.nodes_visited = total_nodes_visited;
    stats.batches_eval = batches_eval;
    for (int i = 0; i < 4; ++i) stats.move_scores[i] = move_scores[i];
    stats.tt_size = transposition_table.occupancy();
    stats.tt_lookups = tt_lookups;
    stats.tt_hits = tt_hits;

    return stats;
}

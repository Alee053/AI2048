#include "ExpectimaxSearcher.h"
#include "RandomUtil.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

ExpectimaxSearcher::ExpectimaxSearcher(size_t target_batch_size)
    : target_batch_size_(target_batch_size) {
}

float get_log_reward(int merge_score) {
    if (merge_score <= 0) return 0.0f;
    return std::log2(static_cast<float>(merge_score));
}

float ExpectimaxSearcher::chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                                 std::vector<uint64_t>& batch_queue) {
    nodes_visited++;

    tt_lookups++;
    uint64_t canon = BoardEncoder::canonicalize(board);
    float cached_score = 0.0f;
    bool probe_result = transposition_table.probe(canon, static_cast<uint8_t>(depth), NodeType::CHANCE, cached_score);
    if (probe_result) {
        tt_hits++;
        return cached_score;
    }

    std::vector<std::pair<int, int>> empty_cells;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (board[r][c] == 0) empty_cells.push_back({r, c});
        }
    }

    if (empty_cells.empty()) {
        float val = max_node_substitute(board, depth - 1, 0, batch_queue, -1e9f, 1e9f);
        if (std::isinf(val) && val < 0) return UNRESOLVED;
        transposition_table.store(canon, static_cast<uint8_t>(depth), NodeType::CHANCE, val);
        return val;
    }

    float total_value = 0.0f;
    bool any_unresolved = false;
    for (const auto& cell : empty_cells) {
        if (batch_queue.size() >= target_batch_size_) {
            return UNRESOLVED;
        }

        Board next_board_2 = board;
        next_board_2[cell.first][cell.second] = 1;
        uint64_t canon_2 = BoardEncoder::canonicalize(next_board_2);
        float val_2 = max_node_substitute(next_board_2, depth - 1, canon_2, batch_queue, -1e9f, 1e9f);
        if (std::isinf(val_2) && val_2 < 0) {
            any_unresolved = true;
        } else {
            total_value += 0.9f * val_2;
        }

        if (batch_queue.size() >= target_batch_size_) {
            return UNRESOLVED;
        }

        Board next_board_4 = board;
        next_board_4[cell.first][cell.second] = 2;
        uint64_t canon_4 = BoardEncoder::canonicalize(next_board_4);
        float val_4 = max_node_substitute(next_board_4, depth - 1, canon_4, batch_queue, -1e9f, 1e9f);
        if (std::isinf(val_4) && val_4 < 0) {
            any_unresolved = true;
        } else {
            total_value += 0.1f * val_4;
        }
    }

    if (any_unresolved) return UNRESOLVED;

    float result = total_value / (2.0f * empty_cells.size());
    transposition_table.store(canon, static_cast<uint8_t>(depth), NodeType::CHANCE, result);
    return result;
}

float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, uint64_t /*board_hash*/,
                                              std::vector<uint64_t>& batch_queue,
                                              float alpha, float beta) {
    nodes_visited++;

    if (depth == 0) {
        uint64_t canon = BoardEncoder::canonicalize(board);
        tt_lookups++;
        float cached_score = 0.0f;
        bool probe_result = transposition_table.probe(canon, 0, NodeType::MAX, cached_score);
        if (probe_result) {
            tt_hits++;
            return cached_score;
        }
        batch_queue.push_back(canon);
        return UNRESOLVED;
    }

    tt_lookups++;
    uint64_t canon = BoardEncoder::canonicalize(board);
    float cached_score = 0.0f;
    if (transposition_table.probe(canon, static_cast<uint8_t>(depth), NodeType::MAX, cached_score)) {
        tt_hits++;
        return cached_score;
    }

    float max_value = -1e9f;
    bool any_move_possible = false;
    bool any_unresolved = false;

    for (int move = 0; move < 4; ++move) {
        if (batch_queue.size() >= target_batch_size_) {
            return UNRESOLVED;
        }
        game_instance.set_board(board);
        auto [merge_score, moved] = game_instance.move_simulated(move);

        if (!moved) continue;
        any_move_possible = true;

        float immediate_reward = get_log_reward(merge_score);
        Board child_board = game_instance.get_board();

        float future_value = chance_node_substitute(child_board, depth, 0, batch_queue);
        if (std::isinf(future_value) && future_value < 0) {
            any_unresolved = true;
        } else {
            float total_value = immediate_reward + future_value;
            if (total_value > max_value) {
                max_value = total_value;
            }
        }

        // Alpha-beta pruning: only active when no unresolved children
        if (!any_unresolved && max_value >= beta) {
            break;
        }
        alpha = std::max(alpha, max_value);
    }

    if (any_unresolved) return UNRESOLVED;

    float result = any_move_possible ? max_value : 0.0f;
    transposition_table.store(canon, static_cast<uint8_t>(depth), NodeType::MAX, result);
    return result;
}

SearchStats ExpectimaxSearcher::find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func) {
    tt_lookups = 0;
    tt_hits = 0;
    batches_eval = 0;
    nodes_visited = 0;
    search_start = std::chrono::high_resolution_clock::now();

    // --- Step 0: Generate root moves ---
    struct RootMove {
        int move_id;
        Board post_board;
        float immediate_reward;
        float post_board_eval;
    };
    std::vector<RootMove> root_moves;
    root_moves.reserve(4);

    for (int move = 0; move < 4; ++move) {
        game_instance.set_board(board);
        auto [ms, moved] = game_instance.move_simulated(move);
        if (!moved) continue;
        root_moves.push_back({move, game_instance.get_board(), get_log_reward(ms), 0.0f});
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

    // --- Step 1: Pre-eval root post-move boards for ordering ---
    if (root_moves.size() > 1) {
        std::vector<Board> boards_to_eval;
        for (size_t i = 0; i < root_moves.size(); ++i) {
            boards_to_eval.push_back(root_moves[i].post_board);
        }
        std::vector<float> evals = batch_eval_func(boards_to_eval);
        batches_eval++;
        for (size_t i = 0; i < root_moves.size(); ++i) {
            root_moves[i].post_board_eval = evals[i];
        }
        std::sort(root_moves.begin(), root_moves.end(), [](const RootMove& a, const RootMove& b) {
            return a.post_board_eval > b.post_board_eval;
        });
    }

    // --- Step 2: Per-move deferred batching ---
    float move_scores[4] = {UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED};
    int resolved_count = 0;

    for (const auto& rm : root_moves) {
        if (!std::isinf(move_scores[rm.move_id])) {
            resolved_count++;
            continue;
        }

        std::vector<uint64_t> batch_queue;
        batch_queue.reserve(target_batch_size_);

        while (std::isinf(move_scores[rm.move_id])) {
            batch_queue.clear();

            float future_value = chance_node_substitute(
                rm.post_board, depth,
                BoardEncoder::canonicalize(rm.post_board),
                batch_queue
            );

            if (!std::isinf(future_value)) {
                move_scores[rm.move_id] = rm.immediate_reward + future_value;
                resolved_count++;
                break;
            }

            if (batch_queue.empty()) {
                throw std::runtime_error(
                    "Empty batch_queue with unresolved move — possible infinite loop"
                );
            }

            std::sort(batch_queue.begin(), batch_queue.end());
            auto last = std::unique(batch_queue.begin(), batch_queue.end());
            batch_queue.erase(last, batch_queue.end());

            std::vector<Board> boards_for_python;
            boards_for_python.reserve(batch_queue.size());
            for (size_t bq_idx = 0; bq_idx < batch_queue.size(); ++bq_idx) {
                boards_for_python.push_back(BoardEncoder::unpack(batch_queue[bq_idx]));
            }

            std::vector<float> values = batch_eval_func(boards_for_python);
            batches_eval++;

            constexpr uint8_t LEAF_DEPTH = 0;
            for (size_t i = 0; i < batch_queue.size(); ++i) {
                transposition_table.store(batch_queue[i], LEAF_DEPTH, NodeType::MAX, values[i]);
                transposition_table.store(batch_queue[i], LEAF_DEPTH, NodeType::CHANCE, values[i]);
            }
        }
    }

    // --- Step 3: Extract best move ---
    float best_score = -1e9f;
    int best_move = root_moves[0].move_id;
    for (const auto& rm : root_moves) {
        if (move_scores[rm.move_id] > best_score) {
            best_score = move_scores[rm.move_id];
            best_move = rm.move_id;
        }
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    double think_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

    SearchStats stats;
    stats.best_move = best_move;
    stats.think_ms = think_ms;
    stats.nodes_visited = nodes_visited;
    stats.batches_eval = batches_eval;
    for (int i = 0; i < 4; ++i) stats.move_scores[i] = move_scores[i];
    stats.tt_size = transposition_table.occupancy();
    stats.tt_lookups = tt_lookups;
    stats.tt_hits = tt_hits;

    return stats;
}

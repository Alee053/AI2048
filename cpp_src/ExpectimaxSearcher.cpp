#include "ExpectimaxSearcher.h"
#include "RandomUtil.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

namespace {

void validate_board_exponents(const Board& board, const char* context) {
    for (const auto& row : board) {
        for (int exponent : row) {
            if (exponent < 0 || exponent > 15) {
                throw std::invalid_argument(
                    std::string(context) + " contains an exponent outside the supported range 0..15."
                );
            }
        }
    }
}

void validate_batch_values(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Batch evaluator returned a non-finite value.");
        }
    }
}

bool is_unresolved(float value) {
    return value == -std::numeric_limits<float>::infinity();
}

void validate_arithmetic_value(float value, const char* context) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(context) + " produced a non-finite value."
        );
    }
}

}  // namespace

ExpectimaxSearcher::ExpectimaxSearcher(size_t target_batch_size, bool use_transposition_table)
    : target_batch_size_(target_batch_size),
      use_transposition_table_(use_transposition_table) {
    if (target_batch_size_ == 0) {
        throw std::invalid_argument("target_batch_size must be positive.");
    }
    if (use_transposition_table_) {
        transposition_table = std::make_unique<TranspositionTable>();
    }
}

float get_log_reward(int merge_score) {
    if (merge_score <= 0) return 0.0f;
    return std::log2(static_cast<float>(merge_score));
}

float ExpectimaxSearcher::chance_node_substitute(const Board& board, int depth, uint64_t board_hash,
                                                  std::vector<uint64_t>& batch_queue,
                                                  const LeafCache& leaf_cache) {
    nodes_visited++;
    chance_nodes_evaluated_++;

    uint64_t canon = BoardEncoder::canonicalize(board);
    float cached_score = 0.0f;
    if (use_transposition_table_) {
        tt_lookups++;
        if (transposition_table->probe(canon, static_cast<uint8_t>(depth), NodeType::CHANCE, cached_score)) {
            tt_hits++;
            if (is_unresolved(cached_score)) return UNRESOLVED;
            validate_arithmetic_value(cached_score, "Cached chance value");
            return cached_score;
        }
    }

    std::vector<std::pair<int, int>> empty_cells;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (board[r][c] == 0) empty_cells.push_back({r, c});
        }
    }

    if (empty_cells.empty()) {
        float val = max_node_substitute(board, depth - 1, 0, batch_queue, leaf_cache);
        if (is_unresolved(val)) return UNRESOLVED;
        validate_arithmetic_value(val, "Chance node child value");
        if (use_transposition_table_) {
            transposition_table->store(canon, static_cast<uint8_t>(depth), NodeType::CHANCE, val);
        }
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
        float val_2 = max_node_substitute(next_board_2, depth - 1, canon_2, batch_queue, leaf_cache);
        if (is_unresolved(val_2)) {
            any_unresolved = true;
        } else {
            validate_arithmetic_value(val_2, "Chance node child value");
            float weighted_value = 0.9f * val_2;
            validate_arithmetic_value(weighted_value, "Chance node weighted value");
            total_value += weighted_value;
            validate_arithmetic_value(total_value, "Chance node accumulation");
        }

        if (batch_queue.size() >= target_batch_size_) {
            return UNRESOLVED;
        }

        Board next_board_4 = board;
        next_board_4[cell.first][cell.second] = 2;
        uint64_t canon_4 = BoardEncoder::canonicalize(next_board_4);
        float val_4 = max_node_substitute(next_board_4, depth - 1, canon_4, batch_queue, leaf_cache);
        if (is_unresolved(val_4)) {
            any_unresolved = true;
        } else {
            validate_arithmetic_value(val_4, "Chance node child value");
            float weighted_value = 0.1f * val_4;
            validate_arithmetic_value(weighted_value, "Chance node weighted value");
            total_value += weighted_value;
            validate_arithmetic_value(total_value, "Chance node accumulation");
        }
    }

    if (any_unresolved) return UNRESOLVED;

    // Expectimax chance node: E[V] = sum_c ( (1/N) * (0.9*V(c,2) + 0.1*V(c,4)) )
    //                        = (1/N) * sum_c (0.9*V(c,2) + 0.1*V(c,4))
    // so divide by N (the number of empty cells), not 2N.
    // The earlier (2.0f * empty_cells.size()) divisor half-scaled the chance
    // value and biased the search.
    float result = total_value / static_cast<float>(empty_cells.size());
    validate_arithmetic_value(result, "Chance node result");
    chance_value_sum_ += result;
    chance_value_count_ += 1;
    if (use_transposition_table_) {
        transposition_table->store(canon, static_cast<uint8_t>(depth), NodeType::CHANCE, result);
    }
    return result;
}

float ExpectimaxSearcher::max_node_substitute(const Board& board, int depth, uint64_t /*board_hash*/,
                                               std::vector<uint64_t>& batch_queue,
                                               const LeafCache& leaf_cache) {
    nodes_visited++;
    max_nodes_evaluated_++;

    if (depth == 0) {
        uint64_t canon = BoardEncoder::canonicalize(board);
        if (const auto cached_leaf = leaf_cache.find(canon); cached_leaf != leaf_cache.end()) {
            if (is_unresolved(cached_leaf->second)) return UNRESOLVED;
            validate_arithmetic_value(cached_leaf->second, "Cached leaf value");
            return cached_leaf->second;
        }
        if (use_transposition_table_) {
            tt_lookups++;
            float cached_score = 0.0f;
            if (transposition_table->probe(canon, 0, NodeType::MAX, cached_score)) {
                tt_hits++;
                if (is_unresolved(cached_score)) return UNRESOLVED;
                validate_arithmetic_value(cached_score, "Cached leaf value");
                return cached_score;
            }
        } else {
            std::vector<Board> boards{BoardEncoder::unpack(canon)};
            std::vector<float> values = (*batch_eval_func_)(boards);
            batches_eval++;
            if (values.size() != boards.size()) {
                throw std::invalid_argument(
                    "Batch evaluator returned an invalid number of values: expected 1, got " +
                    std::to_string(values.size())
                );
            }
            validate_batch_values(values);
            return values[0];
        }
        batch_queue.push_back(canon);
        return UNRESOLVED;
    }

    uint64_t canon = BoardEncoder::canonicalize(board);
    if (use_transposition_table_) {
        tt_lookups++;
        float cached_score = 0.0f;
        if (transposition_table->probe(canon, static_cast<uint8_t>(depth), NodeType::MAX, cached_score)) {
            tt_hits++;
            if (is_unresolved(cached_score)) return UNRESOLVED;
            validate_arithmetic_value(cached_score, "Cached max value");
            return cached_score;
        }
    }

    float max_value = -std::numeric_limits<float>::infinity();
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
        validate_board_exponents(child_board, "simulated move");

        float future_value = chance_node_substitute(child_board, depth, 0, batch_queue, leaf_cache);
        if (is_unresolved(future_value)) {
            any_unresolved = true;
        } else {
            validate_arithmetic_value(future_value, "Max node child value");
            float total_value = immediate_reward + future_value;
            validate_arithmetic_value(total_value, "Max node score");
            if (total_value > max_value) {
                max_value = total_value;
            }
        }

    }

    if (any_unresolved) return UNRESOLVED;

    float result = any_move_possible ? max_value : 0.0f;
    if (use_transposition_table_) {
        transposition_table->store(canon, static_cast<uint8_t>(depth), NodeType::MAX, result);
    }
    return result;
}

SearchStats ExpectimaxSearcher::find_best_move(const Board& board, int depth, const BatchEvalFunc& batch_eval_func) {
    if (depth < 1) {
        throw std::invalid_argument(
            "Expectimax search depth must be at least 1; use the raw PPO policy path for depth 0."
        );
    }
    if (depth > std::numeric_limits<uint8_t>::max()) {
        throw std::invalid_argument(
            "Expectimax search depth must be at most 255 because TT depths are uint8_t."
        );
    }
    validate_board_exponents(board, "search input");

    tt_lookups = 0;
    tt_hits = 0;
    batches_eval = 0;
    nodes_visited = 0;
    chance_nodes_evaluated_ = 0;
    max_nodes_evaluated_ = 0;
    chance_value_sum_ = 0.0;
    chance_value_count_ = 0;
    if (use_transposition_table_) {
        transposition_table->reset_counters();
        transposition_table->begin_new_search();   // age the TT for cross-search eviction
    }
    int moves_resolved_this_call = 0;
    int moves_unresolved_this_call = 0;
    int cap_hits_this_call = 0;
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
        Board post_board = game_instance.get_board();
        validate_board_exponents(post_board, "simulated move");
        root_moves.push_back({move, post_board, get_log_reward(ms), 0.0f});
    }

    if (root_moves.empty()) {
        SearchStats empty_stats{};
        empty_stats.best_move = -1;
        empty_stats.has_legal_move = false;
        empty_stats.search_complete = true;
        empty_stats.think_ms = 0;
        empty_stats.nodes_visited = 0;
        empty_stats.batches_eval = 0;
        for (int i = 0; i < 4; ++i) empty_stats.move_scores[i] = UNRESOLVED;
        empty_stats.tt_size = transposition_table ? transposition_table->occupancy() : 0;
        empty_stats.tt_lookups = 0;
        empty_stats.tt_hits = 0;
        empty_stats.tt_collisions = 0;
        empty_stats.tt_same_key_overwrites = 0;
        empty_stats.moves_resolved = 0;
        empty_stats.moves_unresolved = 0;
        empty_stats.cap_hits = 0;
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
        if (evals.size() != boards_to_eval.size()) {
            throw std::invalid_argument(
                "Batch evaluator returned an invalid number of values: expected " +
                std::to_string(boards_to_eval.size()) + ", got " + std::to_string(evals.size())
            );
        }
        validate_batch_values(evals);
        for (size_t i = 0; i < root_moves.size(); ++i) {
            root_moves[i].post_board_eval = evals[i];
        }
        std::sort(root_moves.begin(), root_moves.end(), [](const RootMove& a, const RootMove& b) {
            if (a.post_board_eval != b.post_board_eval) {
                return a.post_board_eval > b.post_board_eval;
            }
            return a.move_id < b.move_id;
        });
    }

    // --- Step 2: Per-move deferred batching ---
    batch_eval_func_ = &batch_eval_func;
    float move_scores[4] = {UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED};
    int resolved_count = 0;
    constexpr int MAX_ITERATIONS_PER_MOVE = 100;

    for (const auto& rm : root_moves) {
        if (!is_unresolved(move_scores[rm.move_id])) {
            resolved_count++;
            continue;
        }

        std::vector<uint64_t> batch_queue;
        batch_queue.reserve(target_batch_size_);
        LeafCache leaf_cache;
        leaf_cache.reserve(target_batch_size_);

        int iter = 0;
        while (is_unresolved(move_scores[rm.move_id])) {
            if (++iter > MAX_ITERATIONS_PER_MOVE) {
                cap_hits_this_call++;
                std::cerr << "[WARNING] Move " << rm.move_id
                          << " reached MAX_ITERATIONS_PER_MOVE="
                          << MAX_ITERATIONS_PER_MOVE
                          << " for depth=" << depth << "\n";
                break;
            }
            batch_queue.clear();

            float future_value = chance_node_substitute(
                rm.post_board, depth,
                BoardEncoder::canonicalize(rm.post_board),
                batch_queue, leaf_cache
            );

            if (!is_unresolved(future_value)) {
                validate_arithmetic_value(future_value, "Root move value");
                float total_value = rm.immediate_reward + future_value;
                validate_arithmetic_value(total_value, "Root move score");
                move_scores[rm.move_id] = total_value;
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
            if (values.size() != boards_for_python.size()) {
                throw std::invalid_argument(
                    "Batch evaluator returned an invalid number of values: expected " +
                    std::to_string(boards_for_python.size()) + ", got " + std::to_string(values.size())
                );
            }
            validate_batch_values(values);

            for (size_t i = 0; i < batch_queue.size(); ++i) {
                leaf_cache.emplace(batch_queue[i], values[i]);
                if (use_transposition_table_) {
                    constexpr uint8_t LEAF_DEPTH = 0;
                    transposition_table->store(batch_queue[i], LEAF_DEPTH, NodeType::MAX, values[i]);
                }
            }
        }
    }

    // --- Step 3: Extract best move ---
    for (const auto& rm : root_moves) {
        if (is_unresolved(move_scores[rm.move_id])) {
            moves_unresolved_this_call++;
        } else {
            moves_resolved_this_call++;
        }
    }

    const bool search_complete = cap_hits_this_call == 0 && moves_unresolved_this_call == 0;
    int best_move = -1;
    if (search_complete) {
        float best_score = UNRESOLVED;
        for (const auto& rm : root_moves) {
            if (move_scores[rm.move_id] > best_score ||
                (move_scores[rm.move_id] == best_score && rm.move_id < best_move)) {
                best_score = move_scores[rm.move_id];
                best_move = rm.move_id;
            }
        }
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    double think_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

    SearchStats stats;
    stats.best_move = best_move;
    stats.has_legal_move = true;
    stats.search_complete = search_complete;
    stats.think_ms = think_ms;
    stats.nodes_visited = nodes_visited;
    stats.batches_eval = batches_eval;
    for (int i = 0; i < 4; ++i) stats.move_scores[i] = move_scores[i];
    stats.tt_size = transposition_table ? transposition_table->occupancy() : 0;
    stats.tt_lookups = tt_lookups;
    stats.tt_hits = tt_hits;
    stats.tt_collisions = transposition_table ? transposition_table->collision_count() : 0;
    stats.tt_same_key_overwrites = transposition_table ? transposition_table->same_key_overwrite_count() : 0;
    stats.moves_resolved = moves_resolved_this_call;
    stats.moves_unresolved = moves_unresolved_this_call;
    stats.cap_hits = cap_hits_this_call;
    stats.alpha_beta_cuts = 0;
    stats.chance_nodes_evaluated = chance_nodes_evaluated_;
    stats.max_nodes_evaluated = max_nodes_evaluated_;
    stats.chance_value_sum = chance_value_sum_;
    stats.chance_value_count = chance_value_count_;

    return stats;
}

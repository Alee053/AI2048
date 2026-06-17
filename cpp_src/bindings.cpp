#include <pybind11/pybind11.h>
#include "Fast2048.h"
#include "ExpectimaxSearcher.h"
#include "BoardEncoder.h"
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(searcher, m) {
    m.doc() = "High-performance 2048 game logic and AI searcher in C++";

    py::class_<Fast2048>(m, "Fast2048")
        .def(py::init<>())
        .def("reset", &Fast2048::reset, "Resets the game board.")
        .def("move", &Fast2048::move, "Performs a move in a given direction.")
        .def("is_move_valid", &Fast2048::is_move_valid, "Checks if a move is valid.")
        .def("get_board", &Fast2048::get_board, "Returns the current board as a list of lists.")
        .def("set_board", &Fast2048::set_board, "Sets the board to a specific state.")
        .def_property_readonly("board", &Fast2048::get_board)
        .def_property_readonly("score", &Fast2048::get_score)
        .def_property_readonly("max_tile", &Fast2048::get_max_tile);
    py::class_<SearchStats>(m, "SearchStats")
        .def_readonly("best_move", &SearchStats::best_move)
        .def_readonly("think_ms", &SearchStats::think_ms)
        .def_readonly("nodes_visited", &SearchStats::nodes_visited)
        .def_readonly("batches_eval", &SearchStats::batches_eval)
        .def_readonly("tt_size", &SearchStats::tt_size)
        .def_readonly("tt_lookups", &SearchStats::tt_lookups)
        .def_readonly("tt_hits", &SearchStats::tt_hits)
        .def_readonly("tt_collisions", &SearchStats::tt_collisions)
        .def_readonly("tt_same_key_overwrites", &SearchStats::tt_same_key_overwrites)
        .def_readonly("moves_resolved", &SearchStats::moves_resolved)
        .def_readonly("moves_unresolved", &SearchStats::moves_unresolved)
        .def_readonly("cap_hits", &SearchStats::cap_hits)
        .def_readonly("alpha_beta_cuts", &SearchStats::alpha_beta_cuts)
        .def_readonly("chance_nodes_evaluated", &SearchStats::chance_nodes_evaluated)
        .def_readonly("max_nodes_evaluated", &SearchStats::max_nodes_evaluated)
        .def_readonly("chance_value_sum", &SearchStats::chance_value_sum)
        .def_readonly("chance_value_count", &SearchStats::chance_value_count)
        .def_readonly("unique_leaves_evaluated", &SearchStats::unique_leaves_evaluated)
        .def_property_readonly("move_scores", [](const SearchStats& s) {
            return std::array<float, 4>{s.move_scores[0], s.move_scores[1], s.move_scores[2], s.move_scores[3]};
        });

    py::class_<BoardEncoder>(m, "BoardEncoder")
        .def_static("pack", &BoardEncoder::pack)
        .def_static("unpack", &BoardEncoder::unpack)
        .def_static("canonicalize_board", static_cast<uint64_t(*)(const Board&)>(&BoardEncoder::canonicalize))
        .def_static("canonicalize_packed", static_cast<uint64_t(*)(uint64_t)>(&BoardEncoder::canonicalize));

    py::class_<ExpectimaxSearcher>(m, "ExpectimaxSearcher")
        .def(py::init<size_t>(), py::arg("target_batch_size") = 32768)
        .def("find_best_move", &ExpectimaxSearcher::find_best_move,
             "Returns SearchStats with best move and search statistics.",
             py::arg("board"), py::arg("depth"), py::arg("batch_eval_func"))
        .def("clear_tt", &ExpectimaxSearcher::clear_tt,
             "Explicitly clears the persistent transposition table.")
        .def("dump_leaves", &ExpectimaxSearcher::dump_leaves,
             "Diagnostic: dump (canonical_key, value) for all unique leaves "
             "the last find_best_move evaluated. Format: hex_key value, one per line.")
        .def("last_unique_leaves", &ExpectimaxSearcher::last_unique_leaves,
             "Diagnostic: number of unique leaves the last find_best_move evaluated.");
}
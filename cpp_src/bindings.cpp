#include <pybind11/pybind11.h>
#include "Fast2048.h"
#include "ExpectimaxSearcher.h"
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
        .def_property_readonly("move_scores", [](const SearchStats& s) {
            return std::array<float, 4>{s.move_scores[0], s.move_scores[1], s.move_scores[2], s.move_scores[3]};
        });

    py::class_<ExpectimaxSearcher>(m, "ExpectimaxSearcher")
        .def(py::init<>())
        .def("find_best_move", &ExpectimaxSearcher::find_best_move,
             "Returns SearchStats with best move and search statistics.",
             py::arg("board"), py::arg("depth"), py::arg("batch_eval_func"));
}
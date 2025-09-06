#include <pybind11/pybind11.h>
#include "Fast2048.h"
#include "ExpectimaxSearcher.h"
#include <pybind11/stl.h>           // enable conversions for STL containers
#include <pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(fast2048_cpp, m) {
    m.doc() = "Fast 2048 game logic implemented in C++";

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
    py::class_<ExpectimaxSearcher>(m, "ExpectimaxSearcher")
        .def(py::init<>())
        .def("find_best_move", &ExpectimaxSearcher::find_best_move_with_eval,
             py::arg("board"), py::arg("depth"), py::arg("eval_func"));
}
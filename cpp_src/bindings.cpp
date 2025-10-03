/**
 * @file bindings.cpp
 * @brief Creates Python bindings for the C++ 2048 game logic and searcher.
 *
 * This file uses the pybind11 library to expose the high-performance C++ classes
 * `Fast2048` and `ExpectimaxSearcher` to Python. This allows the core game
 * simulation and AI search to run as fast compiled code while maintaining the
 * flexibility of a Python-based training and visualization environment.
 */

﻿#include <pybind11/pybind11.h>
#include <pybind11/stl.h>           // Enable conversions for STL containers
#include <pybind11/functional.h>   // Enable conversions for std::function

#include "Fast2048.h"
#include "ExpectimaxSearcher.h"

namespace py = pybind11;

/**
 * @brief Defines the Python module 'searcher'.
 *
 * This macro is the entry point for pybind11. It defines the module name
 * and the content that will be exposed to Python when the module is imported.
 */
PYBIND11_MODULE(searcher, m) {
    m.doc() = "High-performance C++ module for 2048 game logic and AI search.";

    // --- Bind the Fast2048 game logic class ---
    py::class_<Fast2048>(m, "Fast2048", "A highly optimized C++ implementation of the 2048 game engine.")
        .def(py::init<>(), "Initializes a new game instance.")
        .def("reset", &Fast2048::reset, "Resets the game to a starting state with two random tiles.")
        .def("move", &Fast2048::move,
             "Performs a move in a given direction (0:Up, 1:Right, 2:Down, 3:Left).\n\n"
             "Returns:\n"
             "    A tuple (merge_score, is_done, was_moved).",
             py::arg("direction"))
        .def("is_move_valid", &Fast2048::is_move_valid, "Checks if a move is valid in the current state.", py::arg("action"))
        .def("set_board", &Fast2048::set_board, "Manually sets the board to a specific state.", py::arg("new_board"))
        .def_property_readonly("board", &Fast2048::get_board, "A 4x4 list of lists representing the current board (log2 values).")
        .def_property_readonly("score", &Fast2048::get_score, "The current game score.")
        .def_property_readonly("max_tile", &Fast2048::get_max_tile, "The log2 value of the highest tile on the board.");

    // --- Bind the ExpectimaxSearcher AI class ---
    py::class_<ExpectimaxSearcher>(m, "ExpectimaxSearcher", "An AI searcher that uses a batched Expectimax algorithm.")
        .def(py::init<>())
        .def("find_best_move", &ExpectimaxSearcher::find_best_move,
             "Finds the best move using a batched Expectimax search guided by a value function.\n\n"
             "Args:\n"
             "    board (list[list[int]]): The current board state.\n"
             "    depth (int): The search depth.\n"
             "    batch_eval_func (callable): A Python function that takes a list of boards and returns a list of their values.",
             py::arg("board"), py::arg("depth"), py::arg("batch_eval_func"));
}
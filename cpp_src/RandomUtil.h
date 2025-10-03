/**
 * @file RandomUtil.h
 * @brief Defines a singleton utility class for random number generation.
 */

﻿#pragma once
#include <random>

/**
 * @class RandomUtil
 * @brief A singleton class for providing thread-safe random number generation.
 *
 * This class uses the Mersenne Twister 19937 engine (`std::mt19937`) for
 * high-quality random numbers. The singleton pattern ensures that a single,
 * properly seeded random engine is used throughout the application.
 */
class RandomUtil {
public:
    /**
     * @brief Gets the single instance of the RandomUtil class.
     * @return A reference to the singleton RandomUtil instance.
     */
    static RandomUtil& get();

    /**
     * @brief Generates a random number within a specified range.
     *
     * This template function can generate random numbers of either integral or
     * floating-point types.
     *
     * @tparam T The data type of the random number (e.g., int, float, double).
     * @param min The minimum value of the desired range (inclusive).
     * @param max The maximum value of the desired range (inclusive).
     * @return A random number of type T within the specified range.
     */
    template<typename T>
    T getRandom(T min, T max);

private:
    /**
     * @brief Private constructor to enforce the singleton pattern.
     * Seeds the random number engine using `std::random_device`.
     */
    RandomUtil();

    // Deleted copy constructor and assignment operator to prevent copying.
    RandomUtil(const RandomUtil&) = delete;
    void operator=(const RandomUtil&) = delete;

    /// @brief The core Mersenne Twister random number engine.
    std::mt19937 m_engine;
};


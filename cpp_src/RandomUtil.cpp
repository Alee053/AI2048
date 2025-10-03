/**
 * @file RandomUtil.cpp
 * @brief Implements the RandomUtil singleton class for random number generation.
 */
﻿#include "RandomUtil.h"
#include <type_traits> // For std::is_integral and std::is_floating_point

/**
 * @brief Provides access to the singleton instance of the RandomUtil class.
 *
 * This function ensures that only one instance of RandomUtil is created and used
 * throughout the application, making random number generation thread-safe
 * and consistently seeded.
 *
 * @return A static reference to the RandomUtil instance.
 */
RandomUtil& RandomUtil::get() {
    static RandomUtil instance;
    return instance;
}

/**
 * @brief Private constructor that seeds the random number engine.
 *
 * It uses `std::random_device` to obtain a non-deterministic random number
 * to seed the Mersenne Twister engine, ensuring different sequences of random
 * numbers on each program run.
 */
RandomUtil::RandomUtil() {
    std::random_device rd;
    m_engine.seed(rd());
}

/**
 * @brief Generic random number generator.
 *
 * This is a template function definition. It uses `if constexpr` to select the
 * correct uniform distribution (either `uniform_int_distribution` for integral types
 * or `uniform_real_distribution` for floating-point types) at compile time.
 *
 * @tparam T The data type of the number to generate.
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return A random number of type T between min and max.
 */
template<typename T>
T RandomUtil::getRandom(T min, T max) {
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(m_engine);
    }
    else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(m_engine);
    }
}

// --- Explicit Template Instantiation ---
/**
 * @brief Explicitly instantiates the getRandom template for int, double, and float.
 * This ensures that the linker can find these specific implementations, as the
 * template definition is not in the header file.
 */
template int RandomUtil::getRandom<int>(int min, int max);
template double RandomUtil::getRandom<double>(double min, double max);
template float RandomUtil::getRandom<float>(float min, float max);

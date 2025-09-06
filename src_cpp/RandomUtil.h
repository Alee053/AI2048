#pragma once
#include <random>

class RandomUtil {
public:
    static RandomUtil& get();

    template<typename T>
    T getRandom(T min, T max);

private:
    RandomUtil();

    RandomUtil(const RandomUtil&) = delete;
    void operator=(const RandomUtil&) = delete;

    std::mt19937 m_engine;
};


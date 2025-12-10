#include "RandomUtil.h"
#include <type_traits>

RandomUtil& RandomUtil::get() {
    static RandomUtil instance;
    return instance;
}

RandomUtil::RandomUtil() {
    std::random_device rd;
    m_engine.seed(rd());
}

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

// Template instantiations
template int RandomUtil::getRandom<int>(int min, int max);
template double RandomUtil::getRandom<double>(double min, double max);
template float RandomUtil::getRandom<float>(float min, float max);

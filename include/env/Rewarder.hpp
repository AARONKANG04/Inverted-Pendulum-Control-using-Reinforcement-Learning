#pragma once

#include "../Constants.hpp"

class Rewarder {
public:
    Rewarder() = default;

    float computeReward(const float* state, const float action);

    bool isTerminated(const float* state);
};
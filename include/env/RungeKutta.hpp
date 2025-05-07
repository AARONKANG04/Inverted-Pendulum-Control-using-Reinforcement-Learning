#pragma once

#include "../Constants.hpp"

class RungeKuttaIntegrator {
public:
    RungeKuttaIntegrator() = default;

    void step(float* state, float F);

private:
    inline void dynamics(const float* state, float F, float* ds);
};
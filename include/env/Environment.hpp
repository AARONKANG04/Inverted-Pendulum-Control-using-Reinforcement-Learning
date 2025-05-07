#pragma once

class RolloutBuffer;
#include "RolloutBuffer.hpp"
#include "RungeKutta.hpp"
#include "Rewarder.hpp"
#include "../rl/Networks.hpp"
#include "../Constants.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

class PolicyNetwork;
class ValueNetwork;

inline float normal_log_prob(float x, float mu, float std) {
    static const float LOG_SQRT_2PI = 0.5f * std::log(2.f * M_PI);
    return -LOG_SQRT_2PI - std::log(std) - 0.5f * ((x - mu) * (x - mu)) / (std * std);
}

class DoublePendulumEnvironment {
public:
    DoublePendulumEnvironment();
    ~DoublePendulumEnvironment();

    void reset(float* initial_state);
    void step(float action, float* next_state, float& reward, bool& done);
    void collectRollout(PolicyNetwork* policy, ValueNetwork* value);
    static void convertEnvToNetworkState(float* env_state, float* network_state);

    RolloutBuffer* getBuffer() { return buffer; }
    float* getState() { return state; }
    int getT() { return current_timestep; }

private:
    int current_timestep;
    float state[ENV_STATE_DIM];
    RungeKuttaIntegrator* integrator;
    Rewarder* rewarder;
    RolloutBuffer* buffer;
};
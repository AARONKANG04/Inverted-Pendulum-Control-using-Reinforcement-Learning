#include <env/Rewarder.hpp>

#include <cmath>

float Rewarder::computeReward(const float* state, float action) {
    const float x = state[0];
    const float theta = state[1];
    const float xdot = state[2];
    const float omega = state[3];

    const float healthy_reward = 0.5f;

    float x_tip = x + LENGTH * std::sin(theta);
    float y_tip = LENGTH * std::cos(theta);

    float dx = x_tip - 0.0f;
    float dy = y_tip - 1.0f;
    float distance_penalty = dx * dx + dy * dy;

    float velocity_penalty = omega * omega;

    float raw_reward = healthy_reward - distance_penalty - velocity_penalty;
    return raw_reward * REWARD_SCALE;
}


bool Rewarder::isTerminated(const float* state) {
    float theta = state[1];
    float y_tip = LENGTH * std::cos(theta);
    return y_tip <= 0.8f;
}
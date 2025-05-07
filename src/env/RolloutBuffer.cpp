#include <env/RolloutBuffer.hpp>

#include <iostream>

RolloutBuffer::RolloutBuffer() : ptr(0) {
    transitions = new Transition*[ROLLOUT_CAPACITY];
    for (int i = 0; i < ROLLOUT_CAPACITY; i++) {
        transitions[i] = new Transition();
    }
}

RolloutBuffer::~RolloutBuffer() {
    for (int i = 0; i < ROLLOUT_CAPACITY; i++) {
        delete transitions[i];
    }
    delete[] transitions;
}


void RolloutBuffer::store(float* state, float action, float reward, bool done, float value, float logprob) {
    if (ptr >= ROLLOUT_CAPACITY) std::cout << "RolloutBuffer is full" << "\n";
    float network_state[NETWORK_STATE_DIM];
    DoublePendulumEnvironment::convertEnvToNetworkState(state, network_state);
    transitions[ptr]->copy_state(network_state);
    transitions[ptr]->action = action;
    transitions[ptr]->reward = reward;
    transitions[ptr]->done = done;
    transitions[ptr]->value = value;
    transitions[ptr]->logprob = logprob;
    ptr++;
}

void RolloutBuffer::computeAdvantages() {
    if (ptr == 0) return;

    float gae = 0.0f;
    float next_value = 0.0f;

    for (int t = ptr - 1; t >= 0; t--) {
        const float reward = transitions[t]->reward;
        const float value  = transitions[t]->value;
        const bool done    = transitions[t]->done;
        float mask = done ? 0.0f : 1.0f;

        next_value = (t < ptr - 1) ? transitions[t + 1]->value : 0.0f;

        float delta = reward + GAMMA * mask * next_value - value;
        gae = delta + GAMMA * LAMBDA * mask * gae;

        transitions[t]->advantage     = gae;
        transitions[t]->return_to_go  = (gae + value);
    }

    float adv_mean = 0.0f, ret_mean = 0.0f;
    for (int i = 0; i < ptr; i++) {
        adv_mean += transitions[i]->advantage;
        ret_mean += transitions[i]->return_to_go;
    }
    adv_mean /= ptr;
    ret_mean /= ptr;

    float adv_var = 0.0f, ret_var = 0.0f;
    for (int i = 0; i < ptr; i++) {
        float da = transitions[i]->advantage - adv_mean;
        float dr = transitions[i]->return_to_go - ret_mean;
        adv_var += da * da;
        ret_var += dr * dr;
    }
    float adv_std = std::sqrt(adv_var / ptr) + 1e-8f;
    float ret_std = std::sqrt(ret_var / ptr) + 1e-8f;

    for (int i = 0; i < ptr; i++) {
        transitions[i]->advantage    = (transitions[i]->advantage    - adv_mean) / adv_std * 4.0f;
        transitions[i]->return_to_go = (transitions[i]->return_to_go - ret_mean) / ret_std * 4.0f;
    }
}


void RolloutBuffer::clear() {
    ptr = 0;
}
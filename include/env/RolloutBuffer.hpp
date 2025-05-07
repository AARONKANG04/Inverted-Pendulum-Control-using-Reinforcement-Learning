#pragma once

#include "../Constants.hpp"
#include "Environment.hpp"

#include <cstring>



struct Transition {
    float* state; // The observation before taking action
    float action; // The action taken
    float reward; // The scalar reward received
    bool done; // Whether the episode terminated
    float value; // Estimated value of state (from the valueNet)
    float logprob; // log probability of the action 
    float advantage; // GAE computed advantage 
    float return_to_go; // for value network training -> A_GAE + V(s_t)

    Transition() {
        state = new float[NETWORK_STATE_DIM]; // 8
    }

    ~Transition() {
        delete[] state;
    }

    void copy_state(const float* src) {
        std::memcpy(state, src, NETWORK_STATE_DIM * sizeof(float));
    }
};

class RolloutBuffer {
public:
    RolloutBuffer();
    ~RolloutBuffer();

    void store(float* state, float action, float reward, bool done, float value, float logprob);
    void computeAdvantages();
    void clear();
    
    Transition* getTransition(int idx) { return transitions[idx]; }
    Transition** getTransitions() { return transitions; }
    int getSize() const { return ptr; }

private:
    int ptr;
    Transition** transitions;
};
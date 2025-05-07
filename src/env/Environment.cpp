#include <env/Environment.hpp>

#include <random>
#include <iostream>
#include <cassert>

DoublePendulumEnvironment::DoublePendulumEnvironment() : integrator(new RungeKuttaIntegrator()), rewarder(new Rewarder()), buffer(new RolloutBuffer()), current_timestep(0) {
    reset(state);
}

DoublePendulumEnvironment::~DoublePendulumEnvironment() {
    delete integrator;
    delete rewarder;
    delete buffer;
}

void DoublePendulumEnvironment::reset(float* initial_state) {

    const float scale = 0.03f;
    std::mt19937 rng(std::random_device{}());

    std::uniform_real_distribution<float> uni(-scale, scale);
    state[0] = uni(rng); // x
    state[1] = uni(rng); // theta

    std::normal_distribution<float> normal(0.0f, scale);
    state[2] = normal(rng); // xdot
    state[3] = normal(rng); // thetadot

    std::copy(state, state + ENV_STATE_DIM, initial_state);

    current_timestep = 0;
}

void DoublePendulumEnvironment::step(float action, float* next_state, float& reward, bool& done) {
    const float clipped = std::clamp(action, -MAX_FORCE, MAX_FORCE);

    std::copy(state, state + ENV_STATE_DIM, next_state);
    integrator->step(next_state, clipped);

    reward = rewarder->computeReward(next_state, clipped);
    done = rewarder->isTerminated(next_state);

    if (done)    
        reward -= 10.0f * REWARD_SCALE;

    std::copy(next_state, next_state + ENV_STATE_DIM, state);
    current_timestep++;
}



void DoublePendulumEnvironment::collectRollout(PolicyNetwork* policy, ValueNetwork* value) {
    std::random_device rd; 
    std::mt19937 gen(rd());

    float local_state[ENV_STATE_DIM];
    reset(local_state);

    int total_ep_lens = 0, ep_count = 0, ep_len = 0;

    while (buffer->getSize() < ROLLOUT_CAPACITY) {
        float policy_out[2];
        float value_out[1];

        float network_state[NETWORK_STATE_DIM];
        convertEnvToNetworkState(state, network_state);
        network_state[0]  = state[0] / 2.4f;   
        network_state[3]  = state[2] / 10.f;        
        network_state[4]  = state[3] / 10.f;        

        policy->forward(network_state, policy_out);
        value->forward(network_state, value_out);
        float mu = policy_out[0];
        float log_std = policy_out[1];
        float sigma = std::exp(log_std);
        float v_pred = value_out[0];

        std::normal_distribution<float> dist(mu, sigma);
        float sampled_action = dist(gen);
        float logprob = normal_log_prob(sampled_action, mu, sigma);

        float next_state[ENV_STATE_DIM];
        float reward;
        bool done;

        std::cout << "action: " << sampled_action << "\n";
        step(sampled_action, next_state, reward, done);

        buffer->store(local_state, sampled_action, reward, done, v_pred, logprob);

        if (done || current_timestep > MAX_TIMESTEPS) {
            reset(local_state);
            ep_count++;
            total_ep_lens += ep_len;
            ep_len = 0;
        } else {
            std::copy(next_state, next_state + ENV_STATE_DIM, local_state);
            ep_len++;
        }
    }
    buffer->computeAdvantages();

    std::cout << (float)total_ep_lens / ep_count << "\n";
}

void DoublePendulumEnvironment::convertEnvToNetworkState(float* env_state, float* network_state) {
    network_state[0] = env_state[0];
    network_state[1] = std::sin(env_state[1]);
    network_state[2] = std::cos(env_state[1]);
    network_state[3] = env_state[2];
    network_state[4] = env_state[3];
}
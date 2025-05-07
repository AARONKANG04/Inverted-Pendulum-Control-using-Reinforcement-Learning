#include <iostream>
#include <random>
#include <new>
#include <numeric>
#include <vector>
#include <env/Environment.hpp>
#include <rl/Networks.hpp>
#include <window/Window.hpp>

#include <SFML/Graphics.hpp>

#include "../include/Constants.hpp"

int main() {
    
    DoublePendulumEnvironment env;
    PolicyNetwork policy;
    ValueNetwork value;
    RolloutBuffer* buf = env.getBuffer();

    std::random_device rd; 
    std::mt19937 gen(rd());

    float local_state[ENV_STATE_DIM];
    env.reset(local_state);

    auto window = sf::RenderWindow(sf::VideoMode({1920u, 1080u}), "Inverted Pendulum on Cart");
    window.setFramerateLimit(FRAME_RATE * 1.5);

    const float scale = 200.f; // 1 m -> 100 px
    sf::Vector2u ws = window.getSize();
    sf::Vector2f center(ws.x / 2.f, ws.y / 2.f);

    sf::RectangleShape cart({100, 40});
    cart.setFillColor(sf::Color::Green);
    cart.setOrigin(cart.getSize() / 2.f);

    sf::VertexArray rods(sf::PrimitiveType::Lines, 2);
    rods[0].color = rods[1].color = sf::Color::Black;

    sf::CircleShape bob(10.f);
    bob.setOrigin(sf::Vector2f(10.f, 10.f));
    bob.setFillColor(sf::Color::Red);

    int t = 0;
    int update = 0;

    while (window.isOpen())
    {
        checkForExit(window);

        if (t <= 250 && VISUALIZE) {
            float policy_out[2];
            float value_out[1];

            float network_state[NETWORK_STATE_DIM];
            env.convertEnvToNetworkState(env.getState(), network_state);

            policy.forward(network_state, policy_out);
            value.forward(network_state, value_out);
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
            env.step(sampled_action, next_state, reward, done);

            if (done || env.getT() > MAX_TIMESTEPS) {
                env.reset(local_state);
            } else {
                std::copy(next_state, next_state + ENV_STATE_DIM, local_state);
            }
    
        } else {
            t = 0;

            env.collectRollout(&policy, &value);
            const int N = buf->getSize();
            
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                std::vector<int> indices(N);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), gen);

                float total_policy_loss = 0.0f;
                float total_value_loss = 0.0f;

                int num_minibatches = N / BATCH_SIZE;
                for (int mb = 0; mb < num_minibatches; mb++) {
                    std::vector<int> minibatch_indices(
                        indices.begin() + mb * BATCH_SIZE,
                        indices.begin() + (mb + 1) * BATCH_SIZE
                    );

                    float policy_loss = 0.0f;
                    float value_loss = 0.0f;

                    policy.backward(buf->getTransitions(), minibatch_indices, &policy_loss);
                    value.backward(buf->getTransitions(), minibatch_indices, &value_loss);

                    total_policy_loss += policy_loss;
                    total_value_loss += value_loss;
                }
                
                std::cout << "Update " << update << " | Epoch " << epoch 
                << " | Avg Policy Loss: " << (total_policy_loss / num_minibatches)
                << " | Avg Value Loss: " << (total_value_loss / num_minibatches) << "\n";
            }
            buf->clear();
            update++;

        }

        float x = local_state[0];
        float theta = local_state[1];

        sf::Vector2f pivot = center + sf::Vector2f(x * scale, 0.f);

        sf::Vector2f bob1{
            pivot.x - std::sin(theta)*LENGTH*scale,
            pivot.y - std::cos(theta)*LENGTH*scale
        };
        
        window.clear(sf::Color(135, 206, 235));
        // track
        sf::RectangleShape track({static_cast<float>(ws.x), 5.f});
        track.setFillColor(sf::Color::White);
        track.setOrigin(sf::Vector2f(ws.x / 2.f, 2.5f));
        track.setPosition(center);
        window.draw(track);
        // cart
        cart.setPosition(sf::Vector2f(pivot.x, pivot.y - 22.5f));
        window.draw(cart);
        // rods & bobs
        rods[0].position = pivot;
        rods[1].position = bob1;

        window.draw(rods);
        bob.setPosition(bob1);
        window.draw(bob);

        window.display();  
        t++;  
    }

    return 0;
}

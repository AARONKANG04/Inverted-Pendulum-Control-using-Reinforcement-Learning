#pragma once

constexpr int FRAME_RATE = 30;
constexpr float DELTA_TIME = 1.f / FRAME_RATE;

constexpr int BATCH_SIZE = 8;
constexpr int NETWORK_STATE_DIM = 5; // x, sintheta, costheta, xdot, thetadot
constexpr int H1_DIM = 16;
constexpr int H2_DIM = 16;
constexpr int ROLLOUT_CAPACITY = 512;
constexpr int MAX_TIMESTEPS = 500;
constexpr float MAX_FORCE = 10.0f;
constexpr float POLICY_LR = 1e-3f;
constexpr float VALUE_LR = 3e-3f;
constexpr float EPOCHS = 5;
constexpr float EPSILON_CLIP = 0.2;
constexpr float ENTROPY_COEF = 0.05f;
constexpr float REWARD_SCALE = 1.0f;

constexpr int ENV_STATE_DIM = 4; // x, theta, xdot, thetadot;
constexpr float GAMMA = 0.99f;
constexpr float LAMBDA = 0.95f;

constexpr float G = 4.0f;
constexpr float CART_MASS = 0.1f;
constexpr float BOB_MASS = 0.5f;
constexpr float LENGTH = 1.0f;

constexpr bool VISUALIZE = true;
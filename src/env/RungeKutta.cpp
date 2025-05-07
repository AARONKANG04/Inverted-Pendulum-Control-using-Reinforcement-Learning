#include <env/RungeKutta.hpp>

#include <cmath>

void RungeKuttaIntegrator::step(float* state, float F) {
    float k1[4], k2[4], k3[4], k4[4], s[4];

    dynamics(state, F, k1);

    for(int i = 0; i < 4; ++i) s[i] = state[i] + 0.5f * DELTA_TIME * k1[i];
    dynamics(s, F, k2);
    
    for(int i = 0; i < 4; ++i) s[i] = state[i] + 0.5f * DELTA_TIME * k2[i];
    dynamics(s, F, k3);

    for(int i = 0; i < 4; ++i) s[i] = state[i] + DELTA_TIME * k3[i];
    dynamics(s, F, k4);
    
    for(int i=0; i<4; ++i) {
        state[i] += (DELTA_TIME / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }
}

void RungeKuttaIntegrator::dynamics(const float* state, float F, float* ds) {
    float x = state[0];
    float theta = state[1];
    float xdot = state[2];
    float thetadot = state[3];

    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);

    float D = CART_MASS + BOB_MASS * sin_theta * sin_theta;

    float xdd = (F + BOB_MASS * LENGTH * thetadot * thetadot * sin_theta
                - BOB_MASS * G * sin_theta * cos_theta) / D;

    float thetadd = (-F * cos_theta - BOB_MASS * LENGTH * thetadot * thetadot * sin_theta * cos_theta
                    + (CART_MASS + BOB_MASS) * G * sin_theta) / (LENGTH * D);

    ds[0] = xdot;
    ds[1] = thetadot;
    ds[2] = xdd;
    ds[3] = thetadd;
}

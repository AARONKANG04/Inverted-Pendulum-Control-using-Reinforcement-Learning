#include <rl/Networks.hpp>

#include <cassert>
#include <random>
#include <cmath>
#include <iostream>

void neonMV(float* weightsT, float* bias, float* input, float* output, int in_features, int out_features) {
    #pragma omp parallel for
    for (int i = 0; i < out_features; i++) {
        const float* weights_row = weightsT + i * in_features;
        int d = 0;
        float32x4_t acc_vec = vdupq_n_f32(0.0f);
        for (; d <= in_features - 4; d += 4) {
            float32x4_t weights_vec = vld1q_f32(weights_row + d);
            float32x4_t input_vec = vld1q_f32(input + d);
            acc_vec = vfmaq_f32(acc_vec, weights_vec, input_vec);
        }
        float sum = vaddvq_f32(acc_vec);
        for (; d < in_features; d++) {
            sum += weights_row[d] * input[d];
        }
        output[i] = sum + bias[i];
    }
} 

void neonRelu(float* input, int in_features)
{
    float32x4_t zero_vec = vdupq_n_f32(0.0f);

    int d = 0;
    for (; d + 4 <= in_features; d += 4) {  
        float32x4_t v = vld1q_f32(input + d);
        v = vmaxq_f32(v, zero_vec);
        vst1q_f32(input + d, v);
    }

    for (; d < in_features; ++d) {
        input[d] = std::max(0.0f, input[d]);
    }
}


PolicyNetwork::PolicyNetwork() {
    layer1_weightsT       = (float*)safe_aligned_alloc(sizeof(float) * NETWORK_STATE_DIM * H1_DIM);
    layer1_biases         = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM);
    layer2_weightsT       = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM * H2_DIM);
    layer2_biases         = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM);
    mu_layer_weightsT     = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM * 1);
    mu_layer_bias         = (float*)safe_aligned_alloc(sizeof(float) * 1);
    logstd_layer_weightsT = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM * 1);
    logstd_layer_bias     = (float*)safe_aligned_alloc(sizeof(float) * 1);

    layer1_dweightsT       = (float*)safe_aligned_alloc(sizeof(float) * NETWORK_STATE_DIM * H1_DIM);
    layer1_dbiases         = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM);
    layer2_dweightsT       = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM * H2_DIM);
    layer2_dbiases         = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM);
    mu_layer_dweightsT     = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM * 1);
    mu_layer_dbias         = (float*)safe_aligned_alloc(sizeof(float) * 1);
    logstd_layer_dweightsT = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM * 1);
    logstd_layer_dbias     = (float*)safe_aligned_alloc(sizeof(float) * 1);

    layer1_out = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM);
    layer2_out = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM);
    mu_out = (float*)safe_aligned_alloc(sizeof(float) * 1);
    logstd_out = (float*)safe_aligned_alloc(sizeof(float) * 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist1(0.0f, 1.0f / NETWORK_STATE_DIM);
    for (int i = 0; i < NETWORK_STATE_DIM * H1_DIM; i++) layer1_weightsT[i] = dist1(gen);
    std::fill(layer1_biases, layer1_biases + H1_DIM, 0.0f);
    std::normal_distribution<float> dist2(0.0f, 1.0f / H1_DIM);
    for (int i = 0; i < H1_DIM * H2_DIM; i++) layer2_weightsT[i] = dist2(gen);
    std::fill(layer2_biases, layer2_biases + H2_DIM, 0.0f);
    std::normal_distribution<float> dist3(0.0f, 1.0f / H2_DIM);
    for (int i = 0; i < H2_DIM * 1; i++) mu_layer_weightsT[i] = dist3(gen);
    *mu_layer_bias = 0.0f;
    for (int i = 0; i < H2_DIM * 1; i++) logstd_layer_weightsT[i] = dist3(gen);
    *logstd_layer_bias = 0.0f;

    std::fill(layer1_dweightsT, layer1_dweightsT + NETWORK_STATE_DIM * H1_DIM, 0.0f);
    std::fill(layer1_dbiases, layer1_dbiases + H1_DIM, 0.0f);
    std::fill(layer2_dweightsT, layer2_dweightsT + H1_DIM * H2_DIM, 0.0f);
    std::fill(layer2_dbiases, layer2_dbiases + H2_DIM, 0.0f);
    std::fill(mu_layer_dweightsT, mu_layer_dweightsT + H2_DIM * 1, 0.0f);
    std::fill(mu_layer_dbias, mu_layer_dbias + 1, 0.0f);
    std::fill(logstd_layer_dweightsT, logstd_layer_dweightsT + H2_DIM * 1, 0.0f);
    std::fill(logstd_layer_dbias, logstd_layer_dbias + 1, 0.0f);
}

void PolicyNetwork::forward(float* state, float* out) {
    neonMV(layer1_weightsT, layer1_biases, state, layer1_out, NETWORK_STATE_DIM, H1_DIM);
    neonRelu(layer1_out, H1_DIM);
    neonMV(layer2_weightsT, layer2_biases, layer1_out, layer2_out, H1_DIM, H2_DIM);
    neonRelu(layer2_out, H2_DIM);
    neonMV(mu_layer_weightsT, mu_layer_bias, layer2_out, mu_out, H2_DIM, 1);
    neonMV(logstd_layer_weightsT, logstd_layer_bias, layer2_out, logstd_out, H2_DIM, 1);
    out[0] = std::tanh(mu_out[0]) * MAX_FORCE;
    logstd_out[0] = std::clamp(logstd_out[0], -1.5f, 1.0f);
    out[1] = logstd_out[0];
}


void PolicyNetwork::backward(Transition** data, std::vector<int> indices, float* loss_value) {
    float* states[BATCH_SIZE];
    float actions[BATCH_SIZE];
    float old_logprobs[BATCH_SIZE];
    float advantages[BATCH_SIZE];

    float policy_out[2];
    float minibatch_loss = 0.0f;

    float mus[BATCH_SIZE];
    float stds[BATCH_SIZE];
    bool isClipped[BATCH_SIZE];

    std::fill(layer1_dweightsT,       layer1_dweightsT + NETWORK_STATE_DIM * H1_DIM, 0.0f);
    std::fill(layer1_dbiases,         layer1_dbiases + H1_DIM,                        0.0f);
    std::fill(layer2_dweightsT,       layer2_dweightsT + H1_DIM * H2_DIM,             0.0f);
    std::fill(layer2_dbiases,         layer2_dbiases + H2_DIM,                        0.0f);
    std::fill(mu_layer_dweightsT,     mu_layer_dweightsT + H2_DIM,                    0.0f);
    *mu_layer_dbias = 0.0f;
    std::fill(logstd_layer_dweightsT, logstd_layer_dweightsT + H2_DIM,                0.0f);
    *logstd_layer_dbias = 0.0f;

    for (int k = 0; k < indices.size(); k++) {
        Transition* t = data[indices[k]];
        states[k] = t->state;
        actions[k] = t->action;
        old_logprobs[k] = t->logprob;
        advantages[k] = t->advantage;

        forward(states[k], policy_out);
        mus[k] = policy_out[0];
        float logstd = policy_out[1];
        stds[k] = std::max(1e-6f, std::exp(logstd));

        std::memcpy(layer1_cache[k], layer1_out, sizeof(float)*H1_DIM);
        std::memcpy(layer2_cache[k], layer2_out, sizeof(float)*H2_DIM);

        const float entropy = 0.5f * (1.837877f + 2.0f * logstd);

        float new_logprob = normal_log_prob(actions[k], mus[k], stds[k]);
        float r = std::exp(new_logprob - old_logprobs[k]);
        float unclipped = r * advantages[k];
        float clipped = std::clamp(r, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advantages[k];

        if (clipped >= unclipped) {
            minibatch_loss -= unclipped;
            isClipped[k] = false;
        } else {
            minibatch_loss -= clipped;
            isClipped[k] = true;
        }
        minibatch_loss -= ENTROPY_COEF * entropy;

        float delta = (actions[k] - mus[k]) / (stds[k] * stds[k]);

        float tanh_mu = std::tanh(mus[k] / MAX_FORCE);
        float dtanh = 1.0f - tanh_mu * tanh_mu;

        float dL_dmu = isClipped[k] ? 0.0f : -advantages[k] * r * delta * dtanh * MAX_FORCE;
        float dL_dlogstd = isClipped[k] ? -ENTROPY_COEF : -advantages[k] * r * (-1 + delta * (actions[k] - mus[k])) - ENTROPY_COEF;

        for (int j = 0; j < H2_DIM; j++) {
            mu_layer_dweightsT[j] += dL_dmu * layer2_cache[k][j];
            logstd_layer_dweightsT[j] += dL_dlogstd * layer2_cache[k][j];
        }
        *mu_layer_dbias += dL_dmu;
        *logstd_layer_dbias += dL_dlogstd;

        float dL_dlayer2[H2_DIM];
        for (int j = 0; j < H2_DIM; j++) {
            dL_dlayer2[j] = dL_dmu * mu_layer_weightsT[j] + dL_dlogstd * logstd_layer_weightsT[j];
        }

        for (int j = 0; j < H2_DIM; j++) {
            if (layer2_cache[k][j] <= 0.0f) dL_dlayer2[j] = 0.0f;
        }

        for (int j = 0; j < H2_DIM; j++) {
            for (int i = 0; i < H1_DIM; i++) {
                layer2_dweightsT[j * H1_DIM + i] += dL_dlayer2[j] * layer1_cache[k][i];
            }
            layer2_dbiases[j] += dL_dlayer2[j];
        }

        float dL_dlayer1[H1_DIM] = {0};
        for (int i = 0; i < H1_DIM; i++) {
            for (int j = 0; j < H2_DIM; j++) {
                dL_dlayer1[i] += dL_dlayer2[j] * layer2_weightsT[j * H1_DIM + i];
            }
            if (layer1_cache[k][i] <= 0.0f) dL_dlayer1[i] = 0.0f;
        }

        for (int i = 0; i < H1_DIM; i++) {
            for (int p = 0; p < NETWORK_STATE_DIM; p++) {
                layer1_dweightsT[i * NETWORK_STATE_DIM + p] += dL_dlayer1[i] * states[k][p];
            }
            layer1_dbiases[i] += dL_dlayer1[i];
        }
    }

    *loss_value = minibatch_loss / BATCH_SIZE;

    float scale = -POLICY_LR / BATCH_SIZE;
    for (int i = 0; i < NETWORK_STATE_DIM * H1_DIM; i++) {
        layer1_weightsT[i] += scale * layer1_dweightsT[i];
    }

    float mean = 0.0f;
    for (int i = 0; i < NETWORK_STATE_DIM * H1_DIM; i++) {
        mean += scale * layer1_dweightsT[i];
    }

    for (int i = 0; i < H1_DIM; i++) {
        layer1_biases[i] += scale * layer1_dbiases[i];
    }
    for (int i = 0; i < H1_DIM * H2_DIM; i++) {
        layer2_weightsT[i] += scale * layer2_dweightsT[i];
    }
    for (int i = 0; i < H2_DIM; i++) {
        layer2_biases[i] += scale * layer2_dbiases[i];
    }
    for (int i = 0; i < H2_DIM * 1; i++) {
        mu_layer_weightsT[i] += scale * mu_layer_dweightsT[i];
    }
    *mu_layer_bias += scale * (*mu_layer_dbias);
    for (int i = 0; i < H2_DIM * 1; i++) {
        logstd_layer_weightsT[i] += scale * logstd_layer_dweightsT[i];
    }
    *logstd_layer_bias += scale * (*logstd_layer_dbias);
}



ValueNetwork::ValueNetwork() {
    layer1_weightsT       = (float*)safe_aligned_alloc(sizeof(float) * NETWORK_STATE_DIM * H1_DIM);
    layer1_biases         = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM);
    layer2_weightsT       = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM * H2_DIM);
    layer2_biases         = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM);
    value_layer_weightsT     = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM * 1);
    value_layer_bias         = (float*)safe_aligned_alloc(sizeof(float) * 1);

    layer1_dweightsT       = (float*)safe_aligned_alloc(sizeof(float) * NETWORK_STATE_DIM * H1_DIM);
    layer1_dbiases         = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM);
    layer2_dweightsT       = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM * H2_DIM);
    layer2_dbiases         = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM);
    value_layer_dweightsT     = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM * 1);
    value_layer_dbias         = (float*)safe_aligned_alloc(sizeof(float) * 1);

    layer1_out = (float*)safe_aligned_alloc(sizeof(float) * H1_DIM);
    layer2_out = (float*)safe_aligned_alloc(sizeof(float) * H2_DIM);
    value_out = (float*)safe_aligned_alloc(sizeof(float) * 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist1(0.0f, 1.0f / NETWORK_STATE_DIM);
    for (int i = 0; i < NETWORK_STATE_DIM * H1_DIM; i++) layer1_weightsT[i] = dist1(gen);
    std::fill(layer1_biases, layer1_biases + H1_DIM, 0.0f);
    std::normal_distribution<float> dist2(0.0f, 1.0f / H1_DIM);
    for (int i = 0; i < H1_DIM * H2_DIM; i++) layer2_weightsT[i] = dist2(gen);
    std::fill(layer2_biases, layer2_biases + H2_DIM, 0.0f);
    std::normal_distribution<float> dist3(0.0f, 1.0f / H2_DIM);
    for (int i = 0; i < H2_DIM * 1; i++) value_layer_weightsT[i] = dist3(gen);
    *value_layer_bias = 0.0f;

    std::fill(layer1_dweightsT, layer1_dweightsT + NETWORK_STATE_DIM * H1_DIM, 0.0f);
    std::fill(layer1_dbiases, layer1_dbiases + H1_DIM, 0.0f);
    std::fill(layer2_dweightsT, layer2_dweightsT + H1_DIM * H2_DIM, 0.0f);
    std::fill(layer2_dbiases, layer2_dbiases + H2_DIM, 0.0f);
    std::fill(value_layer_dweightsT, value_layer_dweightsT + H2_DIM * 1, 0.0f);
    std::fill(value_layer_dbias, value_layer_dbias + 1, 0.0f);
}



void ValueNetwork::forward(float* state, float* out) {
    neonMV(layer1_weightsT, layer1_biases, state, layer1_out, NETWORK_STATE_DIM, H1_DIM);
    neonRelu(layer1_out, H1_DIM);
    neonMV(layer2_weightsT, layer2_biases, layer1_out, layer2_out, H1_DIM, H2_DIM);
    neonRelu(layer2_out, H2_DIM);
    neonMV(value_layer_weightsT, value_layer_bias, layer2_out, value_out, H2_DIM, 1);
    out[0] = value_out[0];
}


void ValueNetwork::backward(Transition** data, std::vector<int> indices, float* loss_value) {
    float minibatch_loss = 0.0f;

    std::fill(layer1_dweightsT, layer1_dweightsT + NETWORK_STATE_DIM * H1_DIM, 0.0f);
    std::fill(layer1_dbiases,   layer1_dbiases + H1_DIM,                        0.0f);
    std::fill(layer2_dweightsT, layer2_dweightsT + H1_DIM * H2_DIM,            0.0f);
    std::fill(layer2_dbiases,   layer2_dbiases + H2_DIM,                        0.0f);
    std::fill(value_layer_dweightsT, value_layer_dweightsT + H2_DIM,           0.0f);
    *value_layer_dbias = 0.0f;

    for (int k = 0; k < indices.size(); k++) {
        Transition* t = data[indices[k]];
        float* state = t->state;
        float target = t->return_to_go;

        forward(state, value_out);  
        std::memcpy(layer1_cache[k], layer1_out, sizeof(float)*H1_DIM);
        std::memcpy(layer2_cache[k], layer2_out, sizeof(float)*H2_DIM);
        float prediction = value_out[0];

        float loss = 0.5f * (prediction - target) * (prediction - target);
        minibatch_loss += loss;

        if (!std::isfinite(prediction)) {
            std::cerr << "Prediction is NaN or Inf: " << prediction << "\n";
        }
        if (!std::isfinite(target)) {
            std::cerr << "Target is NaN or Inf: " << target << "\n";
        }

        float dL_dvalue = prediction - target;

        assert(std::isfinite(dL_dvalue));

        for (int i = 0; i < H2_DIM; i++) {
            value_layer_dweightsT[i] += dL_dvalue * layer2_cache[k][i];
        }
        *value_layer_dbias += dL_dvalue;

        float dL_dlayer2[H2_DIM];
        for (int i = 0; i < H2_DIM; i++) {
            dL_dlayer2[i] = dL_dvalue * value_layer_weightsT[i];
            if (layer2_cache[k][i] <= 0.0f) dL_dlayer2[i] = 0.0f;
        }

        for (int j = 0; j < H2_DIM; j++) {
            for (int i = 0; i < H1_DIM; i++) {
                layer2_dweightsT[j * H1_DIM + i] += dL_dlayer2[j] * layer1_cache[k][i];
            }
            layer2_dbiases[j] += dL_dlayer2[j];
        }

        float dL_dlayer1[H1_DIM] = {0};
        for (int i = 0; i < H1_DIM; i++) {
            for (int j = 0; j < H2_DIM; j++) {
                dL_dlayer1[i] += dL_dlayer2[j] * layer2_weightsT[j * H1_DIM + i];
            }
            if (layer1_cache[k][i] <= 0.0f) dL_dlayer1[i] = 0.0f;
        }

        for (int i = 0; i < H1_DIM; i++) {
            for (int p = 0; p < NETWORK_STATE_DIM; p++) {
                layer1_dweightsT[i * NETWORK_STATE_DIM + p] += dL_dlayer1[i] * state[p];
            }
            layer1_dbiases[i] += dL_dlayer1[i];
        }
    }
    
    assert(std::isfinite(minibatch_loss));

    *loss_value = minibatch_loss;

    float scale = -VALUE_LR / BATCH_SIZE;

    for (int i = 0; i < NETWORK_STATE_DIM * H1_DIM; i++) {
        layer1_weightsT[i] += scale * layer1_dweightsT[i];
    }
    for (int i = 0; i < H1_DIM; i++) {
        layer1_biases[i] += scale * layer1_dbiases[i];
    }
    for (int i = 0; i < H1_DIM * H2_DIM; i++) {
        layer2_weightsT[i] += scale * layer2_dweightsT[i];
    }
    for (int i = 0; i < H2_DIM; i++) {
        layer2_biases[i] += scale * layer2_dbiases[i];
    }
    for (int i = 0; i < H2_DIM; i++) {
        value_layer_weightsT[i] += scale * value_layer_dweightsT[i];
    }
    *value_layer_bias += scale * (*value_layer_dbias);
}
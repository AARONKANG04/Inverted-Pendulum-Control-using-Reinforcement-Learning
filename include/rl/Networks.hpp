#pragma once

#include "../Constants.hpp"
#include <env/RolloutBuffer.hpp>
#include <env/Environment.hpp>

#include <arm_neon.h>
#include <omp.h>
#include <new>
#include <vector>

struct Transition;

inline void* safe_aligned_alloc(size_t size, size_t alignment = 16) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0 || ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void neonMV(float* weightsT, float* bias, float* input, float* output, int in_features, int out_features);

void neonRelu(float* input, int in_features);

class PolicyNetwork {
public:
    PolicyNetwork();
    // ~PolicyNetwork();

    void forward(float* state, float* out);
    void backward(Transition** data, std::vector<int> indices, float* loss_value);

private:
    // Weights and Biases
    float* layer1_weightsT = nullptr;
    float* layer1_biases = nullptr;
    float* layer2_weightsT = nullptr;
    float* layer2_biases = nullptr;
    float* mu_layer_weightsT = nullptr;
    float* mu_layer_bias = nullptr;
    float* logstd_layer_weightsT = nullptr;
    float* logstd_layer_bias = nullptr;

    // Gradients
    float* layer1_dweightsT = nullptr;
    float* layer1_dbiases = nullptr;
    float* layer2_dweightsT = nullptr;
    float* layer2_dbiases = nullptr;
    float* mu_layer_dweightsT = nullptr;
    float* mu_layer_dbias = nullptr;
    float* logstd_layer_dweightsT = nullptr;
    float* logstd_layer_dbias = nullptr;

    // Buffers
    float* layer1_out = nullptr;
    float* layer2_out = nullptr;
    float* mu_out = nullptr;
    float* logstd_out = nullptr;

    float layer1_cache[BATCH_SIZE][H1_DIM];
    float layer2_cache[BATCH_SIZE][H2_DIM];
};


class ValueNetwork {
public:
    ValueNetwork();

    void forward(float* state, float* out);
    void backward(Transition** data, std::vector<int> indices, float* loss_value);

private:
    // Weights and Biases
    float* layer1_weightsT = nullptr;
    float* layer1_biases = nullptr;
    float* layer2_weightsT = nullptr;
    float* layer2_biases = nullptr;
    float* value_layer_weightsT = nullptr;
    float* value_layer_bias = nullptr;

    // Gradients
    float* layer1_dweightsT = nullptr;
    float* layer1_dbiases = nullptr;
    float* layer2_dweightsT = nullptr;
    float* layer2_dbiases = nullptr;
    float* value_layer_dweightsT = nullptr;
    float* value_layer_dbias = nullptr;

    // Buffers
    float* layer1_out = nullptr;
    float* layer2_out = nullptr;
    float* value_out = nullptr;

    float layer1_cache[BATCH_SIZE][H1_DIM];
    float layer2_cache[BATCH_SIZE][H2_DIM];
};
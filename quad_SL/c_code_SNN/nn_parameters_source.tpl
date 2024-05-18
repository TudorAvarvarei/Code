#include "nn_parameters.h"

// NN network parameters -- atribute values
const float weights_in[NUM_NODES][NUM_STATES] = ${weights_in};

const float weights_out[NUM_CONTROLS][NUM_NODES] = ${weights_out};

const float in_norm_min[NUM_STATES] = ${in_norm_min};

const float in_norm_max[NUM_STATES] = ${in_norm_max};

const float out_scale_min = ${out_scale_min};

const float out_scale_max = ${out_scale_max};

const float weights_hid[NUM_HIDDEN_LAYERS-1][NUM_NODES][NUM_NODES] = ${weights_hidden};

const float leak_i[NUM_HIDDEN_LAYERS][NUM_NODES] = ${leak_i};

const float leak_i[NUM_HIDDEN_LAYERS][NUM_NODES] = ${leak_v};

const float thresh[NUM_HIDDEN_LAYERS][NUM_NODES] = ${thresh};
#include "nn_parameters.h"

// NN network parameters -- atribute values

const float in_norm_min[NUM_STATES] = ${in_norm_min};

const float in_norm_max[NUM_STATES] = ${in_norm_max};

const float out_scale_min = ${out_scale_min};

const float out_scale_max = ${out_scale_max};

float leak_i_layer1[NUM_NODES] = ${leak_i_layer1};

float leak_i_layer2[NUM_NODES] = ${leak_i_layer2};

float leak_i_layer3[NUM_NODES] = ${leak_i_layer3};

float leak_v_layer1[NUM_NODES] = ${leak_v_layer1};

float leak_v_layer2[NUM_NODES] = ${leak_v_layer2};

float leak_v_layer3[NUM_NODES] = ${leak_v_layer3};

float thresh_layer1[NUM_NODES] = ${thresh_layer1};

float thresh_layer2[NUM_NODES] = ${thresh_layer2};

float thresh_layer3[NUM_NODES] = ${thresh_layer3};

float weights_in[NUM_NODES*NUM_STATES] = ${weights_in};

float weights_out[NUM_CONTROLS*NUM_NODES] = ${weights_out};

float weights_hid_layers1_2[NUM_NODES*NUM_NODES] = ${weights_hid_layers1_2};

float weights_hid_layers2_3[NUM_NODES*NUM_NODES] = ${weights_hid_layers2_3};

#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#define NUM_STATES 19
#define NUM_CONTROLS 4
#define NUM_LAYERS 5
#define NUM_HIDDEN_LAYERS 3
#define NUM_NODES 500


// NN network parameters -- define variables
extern const float weights_in[NUM_NODES][NUM_STATES];

extern const float weights_out[NUM_CONTROLS][NUM_NODES];

extern const float in_norm_min[NUM_STATES];

extern const float in_norm_max[NUM_STATES];

extern const float out_scale_min;

extern const float out_scale_max;

extern const float weights_hid[NUM_HIDDEN_LAYERS-1][NUM_NODES][NUM_NODES];

extern const float leak_i[NUM_HIDDEN_LAYERS][NUM_NODES];

extern const float leak_v[NUM_HIDDEN_LAYERS][NUM_NODES];

extern const float thresh[NUM_HIDDEN_LAYERS][NUM_NODES];

#endif

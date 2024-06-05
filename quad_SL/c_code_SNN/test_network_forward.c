#include "test_network_forward.h"

Network net;

// Test network forward functions
int main() {
  // Build network
  // type, decoding_scale, centers, in_size, in_enc_size, hid_size, out_size,
  // inhid, hid, hidout, out
  // post, pre, w
  ConnectionConf const conf_inhid = {NUM_NODES, NUM_STATES, weights_in};

  // post, pre, w
  // const float *ptr_weights_out = &weights_out[0][0];
  ConnectionConf const conf_hidout = {NUM_CONTROLS, NUM_NODES, weights_out};

  // post, pre, w
  // const float *ptr_weights_hid_layers1_2 = &weights_hid_layers1_2[0][0];
  ConnectionConf const conf_hid_1 = {NUM_NODES, NUM_NODES, weights_hid_layers1_2};

  // post, pre, w
  // const float *ptr_weights_hid_layers2_3 = &weights_hid_layers2_3[0][0];
  ConnectionConf const conf_hid_2 = {NUM_NODES, NUM_NODES, weights_hid_layers2_3};

  // size, leak_i, leak_v, thresh, v_rest
  NeuronConf const conf_layer1 = {NUM_NODES, leak_i_layer1, leak_v_layer1, 
                                  thresh_layer1, 0.0f};

  // size, leak_i, leak_v, thresh, v_rest
  NeuronConf const conf_layer2 = {NUM_NODES, leak_i_layer2, leak_v_layer2, 
                                  thresh_layer2, 0.0f};

  // size, leak_i, leak_v, thresh, v_rest
  NeuronConf const conf_layer3 = {NUM_NODES, leak_i_layer3, leak_v_layer3, 
                                  thresh_layer3, 0.0f};

  const float out_scale_min = 3.000000000e+03;

  const float out_scale_max = 1.200000000e+04;

  NetworkConf const qz_conf = {in_norm_min, in_norm_max, out_scale_min, out_scale_max, 
                          NUM_STATES, NUM_HIDDEN_LAYERS, NUM_NODES, NUM_CONTROLS,
                          &conf_inhid, &conf_hid_1, &conf_hid_2, &conf_hidout,
                          &conf_layer1, &conf_layer2, &conf_layer3};
  net = build_network(qz_conf.in_size, qz_conf.hid_layer_size, 
                      qz_conf.hid_neuron_size, qz_conf.out_size);
  // Init network
  init_network(&net);

  // printf("Afterinit:\n");
  // print_array_1d(net.in_size, net.in_norm_max);

  // Load network parameters from header file
  load_network_from_header(&net, &qz_conf);
  // printf("After load network:\n");
  // print_array_1d(net.hid_neuron_size, net.layer1->leak_i);
  reset_network(&net);

  net.input[0] = 0.0f;
  net.input[1] = 1.0f;
  net.input[2] = 2.0f;
  net.input[3] = 3.0f;
  net.input[4] = 4.0f;
  net.input[5] = 5.0f;
  net.input[6] = 6.0f;
  net.input[7] = 7.0f;
  net.input[8] = 8.0f;
  net.input[9] = 9.0f;
  net.input[10] = 10.0f;
  net.input[11] = 11.0f;
  net.input[12] = 12.0f;
  net.input[13] = 13.0f;
  net.input[14] = 14.0f;
  net.input[15] = 15.0f;
  net.input[16] = 16.0f;
  net.input[17] = 17.0f;
  net.input[18] = 18.0f;

  forward_network(&net);

  // Print network state
  printf("\nHeader loading\n\n");
  // Print network inputs
  printf("Network inputs:\n");
  print_array_1d(net.in_size, net.input);
  // Print encoded network inputs
  printf("Encoded network inputs:\n");
  print_array_1d(net.in_size, net.input_norm);
  // Print network output
  printf("Network output:\n");
  print_array_1d(net.out_size, net.output);
  // Print decoded network output
  printf("Decoded network output:\n");
  print_array_1d(net.out_size, net.output_decoded);

  // printf("After reset:\n");
  // print_array_1d(net.in_size, net.input);

  // Set input to network
  for (int i=0; i<net.in_size; i++){
    net.input[i] = 0.0f;
  }
  net.input[0] = 4.0f;
  net.input[12] = 7500.0f;
  net.input[13] = 7500.0f;
  net.input[14] = 7500.0f;
  net.input[15] = 7500.0f;

  forward_network(&net);

  // Print network state
  printf("\nHeader loading\n\n");
  // Print network inputs
  printf("Network inputs:\n");
  print_array_1d(net.in_size, net.input);
  // Print encoded network inputs
  printf("Encoded network inputs:\n");
  print_array_1d(net.in_size, net.input_norm);
  // Print network output
  printf("Network output:\n");
  print_array_1d(net.out_size, net.output);
  // Print decoded network output
  printf("Decoded network output:\n");
  print_array_1d(net.out_size, net.output_decoded);

  // printf("Network took %f seconds to execute \n", time/100000.0f); 

  // Free network memory again
  free_network(&net);

  return 0;
}
#include "test_network_forward.h"

// Test network forward functions
int main() {
  // Build network
  Network net = build_network(conf.in_size, conf.hid_layer_size, 
                              conf.hid_neuron_size, conf.out_size);
  // Init network
  init_network(&net);
  // Set input to network
  for (int i=0; i<net.in_size; i++){
    net.input[i] = 0.0f;
  }
  net.input[0] = 4.0f;
  net.input[12] = 7500.0f;
  net.input[13] = 7500.0f;
  net.input[14] = 7500.0f;
  net.input[15] = 7500.0f;

  // Load network parameters from header file
  load_network_from_header(&net, &conf);
  reset_network(&net);

  // Forward network
  forward_network(&net);

  // Print network state
  printf("\nHeader loading\n\n");
  // Print network inputs
  printf("Network inputs:\n");
  print_array_1d(net.in_size, net.input);
  // Print encoded network inputs
  printf("Encoded network inputs:\n");
  print_array_1d(net.in_size, net.input_norm);
  // Print input -> hidden weights
  // printf("Input -> hidden weights:\n");
  // print_array_2d(net.hid_size, net.in_enc_size, net.inhid->w);
  // Print hidden layer inputs
  // printf("Hidden layer inputs:\n");
  // print_array_1d(net.hid_neuron_size, net.hid->x);
  // Print hidden layer voltages
  // printf("Hidden layer voltages:\n");
  // print_array_1d(net.hid_neuron_size, net.hid->v);
  // Print hidden layer thresholds
  // printf("Hidden layer thresholds:\n");
  // print_array_1d(net.hid_neuron_size, net.hid->th);
  // Print hidden layer spikes
  // printf("Hidden layer spikes:\n");
  // print_array_1d_bool(net.hid_neuron_size, net.hid->s);
  // Print hidden layer spike count
  // printf("Hidden layer spike count: %d\n\n", net.hid->s_count);
  // Print hidden layer trace
  // printf("Hidden layer trace:\n");
  // print_array_1d(net.hid_neuron_size, net.hid->t);
  // Print hidden -> output weights
  // printf("Hidden -> output weights:\n");
  // print_array_2d(net.out_size, net.hid_size, net.hidout->w);
  // Print output layer inputs
  // printf("Output layer inputs:\n");
  // print_array_1d(net.out_size, net.out->x);
  // Print output layer voltages
  // printf("Output layer voltages:\n");
  // print_array_1d(net.out_size, net.out->v);
  // Print output layer thresholds
  // printf("Output layer thresholds:\n");
  // print_array_1d(net.out_size, net.out->th);
  // Print output layer spikes
  // printf("Output layer spikes:\n");
  // print_array_1d_bool(net.out_size, net.out->s);
  // Print output layer spike count
  // printf("Output layer spike count: %d\n\n", net.out->s_count);
  // Print output layer trace
  // printf("Output layer trace:\n");
  // print_array_1d(net.out_size, net.out->t);
  // Print network output
  printf("Network output:\n");
  print_array_1d(net.out_size, net.output);
  // Print decoded network output
  printf("Decoded network output:\n");
  print_array_1d(net.out_size, net.output_decoded);

  // Free network memory again
  free_network(&net);

  return 0;
}
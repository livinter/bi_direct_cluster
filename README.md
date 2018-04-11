# Bi-directional Clustering 
MNIST bi directional clustering as prove of concept. 
Exploring alternatives to backpropagation.


How it works:
------------

   * Learning is just works by adjusting weighs to activations.
   * Only the neuron that most matches is adjusted.
   * "matches" is defined by highest activation from the previous layer AND the next layer.



Introduction
-----------------
The evolution of ANN has developed on picture recognizing, mainly MNIST. 
In contrast, the evolution of biological NN has grown based on streams, experiences what is more like videos. 
No child is trained on batches of images, instead it plays with an object for some time.
It is not about making a 1:1 copy of biology, but I argue that thinking of streams will bring 
another kind neuronal networks of, that has also different advantages and may be even more natural.
A network that receive a constant stream of data can relay on being already semi- initialized. 
Traditional ANN start totally in blank and have to do a serial forward path, followed by a backward path for learning their weights.
The described model can also be processed in random order, because every neuron can expect to have already semi- initialized neurons around it.
This knowledge from previous frames is used to learn and recognize the actual frame by design. The model does not use back propagation, 
instead, neuron areas behave more like in k-mean, but not only from the input, instead  a symmetrical way taking the output also into account.



Concept
=======



Activation from both sides
--------------------------------------
A neuron is not only activated from its input but also from its output!!!
Based in the intuition that we see also depends on what we expect. 
During processing a video for example, there is a prior frame with similar information. That way knowledge from 
prior frames help to understand the actual moment and even better.

A neuron needs to know two things:
 - How it gets activated 
 - How it gets better activated in the future.
 
Normally the activation is the sum of its input processed by an activation function, and the better activation in the future is calculated from the expectation of its outputs.

In contrast, at this model the activation comes from the sum of its activation of its inputs AND its outputs.
As the neurons in an area compete who match the actual situation most, only the most matching neuron is taken and its connections are 
modified to match in the future even more. Similar to clustering with the k-mean algorithm, where the centroid positions is adjusted.

There are two forces that shape the connections 
 - Expectations
 - Clustering

This model is nearly symmetric, as the clustering works also in both directions. A neuron may not only represent a common input pattern, but also represent a state expected by several following layers.

No special order in the execution is required, even works great in random order.



Divide layers into areas
----------------------------------
Layers are divided into areas. There is one winner neuron in each area. An activation function similar to softmax is applied to the neuron area/group.

Imagine representing a 3-digit-number like 421, now if you want to represent all possible 3 digit numbers you would need 1000 neurons. 
But if you divide the layer into 3 areas with 10 neurons each, you can represent the three digit numbers with only 30 neurons. 
In each area one neuron is selected that matches the number most.
Speaking in general: The amount of areas represents the amount of aspects that the layer should represent. 
The size of each area the amount of options per aspect. This can also be thought of a specific form of
sparse activation. 
When you add more layers into a network two things can happen:
  1. best case: different layers take different level of abstraction
  2. worst case: additional layer tend to behave like pass through, maybe they refine the data, but also tend to lose information
Forcing the layer into different areas is a way to enforce the second option, as the area is small and can only cover a partial aspect.
This is nothing new: CNNs are actually local y connected less independent areas/groups of neurons.
Some people initialize the weights in ANN similar using a Gauss distribution focused to a special point.



Modifying just one neuron per layer-area
-----------------------------------------------------------
Only the weights of the most active neuron are changed. 
Based in the intuition: If you see a frog, do not modify/destroy the knowledge about cars. 
Just the frog will be recognized in future better as a frog. This aspect in ANN is somehow 
covered by the ReLU function that is off when it does not reach a certain threshold. In this case
the idea is applied a bit stronger. First I thought that I should modify several most active neurons.
Experimenting turns out that the best results came up only modifying the best matching neuron in each area.
This also results in an easier implementation. 



Train on variations instead of batches
------------------------------------------------------
When a baby grab a new toy, it likes to rotate the toy to see it from different angels.
In this model learning is done that way: Objects need to be presented in different variations.
This way the changes in the first layers get connected with abstractions already done in further layers.
I made small animations from (fashion) MNIST of several frames.





Make it work
==========

On MNIST with a single layer I get 94%. Adding a fixed single simplified CNN layer gets it to 97%.
So at this scale the concept seem to work quite well. The challenge is to scale it into a deep network.
I got the network into a state where it does not collapse, but I feel that with all the stabilization in position
flexibility has been removed that affect performance. 

Normalization 
-------------------- 
To calculate the activation the percentage that comes from input and output need to be normalized.
A relation of 1:1 from the weights of the input side and the outputs side is enforced.

    `sum(activation_*weights_in)/sum(weights_in) + sum(activation_*weights_out)/sum(weights_out)`

Bigger areas tend to have less probability of a neuron to be active then bigger once. For this reason
each side is multiplied by their source area side.



Ensure that all neurons are used
-----------------------------------------------
There is a counter how offer each neuron has been selected/activated. The more a neuron has been selected,
the more difficult it gets for it to be selected in the future. This is enforced that all neurons represent something.



Activation function
---------------------------
A ReLU function is applied to the activation of all neurons. Its scaling parameters are taken 
from the that area the neuron is located. The most active neuron is one. Other neurons that have high values 
between 0 and 1 and everything under a certain threshold is set to 0. 

As there are no negative connections and no negative activation the function can not react to the case, where all 
values are input values a zero. So this special case is caught and the output need to be ignored.



Learning
-------------
I take all MNIST pictures and find for each image the most closely related (pixel difference)...
Then all images are lines up. That way several animations are generated. This are the animations used
for training the network.
For CIFAR variations are generated by transforming like stretching/rotating/zooming/..



All neurons are placed in a single 1d array 
------------------------------------------------------------

Each neuron comes with an array of indexes to its source neurons and destination neurons,
including another index to the corresponding weights.



Simple CNN
-----------------
Normally a whole layer share a single kernel, but we use a kernel for neuron. Then the max pooling
is applied and the indexes are moved, with a pre-calculated move-table. 
Having individual kernels result in a little slower training at the beginning, but is more flexible,
when different shapes are typical in different areas.



Specific connections vs fully connected
--------------------------------------------------------
The connections are bidirectional. So both neurons point to the same weight. This is more overhead at the beginning. 
But during the training most weights tend to go to zero and can be pruned, resulting in a more
efficient network at the end. The connections are sorted slowly with bubble sort. After some time when the 
weights have been established 



Calculating weights
----------------------------

Weights recalculation is only applied to the winner neuron that got most activation in its area.
This reduces weight recalculation 1/(area size).




Progress
========


Things I wonder
-----------------------


 - The network has to avoid vanishing. If the network gets into a state where all neurons are activated 50% all the time it can not have a (clearly) represent information.
 - The network has to avoid too much concentration. If activation gets to concentrated on few neurons, most of the network will neither represent information.

Both problems exist in the value of a neuron, it its distribution at any given moment, and during time.
To force a neuron activation into a range, an activation function is used.
To force a special distribution a function like softmax is applied to an area.
Avoiding that the same neurons are active/deactive during all the time, a timer is used. The more time has passed the easier the neuron gets activated again.

I feel that there is too much external forcing in the network instead of natural balance. 
But forcing the network (too much) destroys information.
Neurons are forced to represent something, but in the worst case this is noise.

The activation influences the weights(distribution). The weights influence the activation (distribution)


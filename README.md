# Bi-directional Clustering - Numpy Only
MNIST bi directional clustering as prove of concept. 

Exploring alternatives to backpropagation.
... > 95% on MINST with only one layer and no image augmentation/preparation/variation and no conv.

Code focus on simplicity. main calculation in few lines NumPy.
Main goal is to prove that a neuron does not only need to be defined on how it should behave (backpropagation) but also on how it could behave by just clustering the input. This clustering can be applyed forward and backward.


How it works:
------------
  - Learning is just works by adjusting weighs to activations.
  - Only the neuron that most matches is adjusted.
  - "matches" is defined by highest activation from the previous layer AND the next layer.


Concept Features
----------------
 - apply logic symmetrically
 - only learn the neuron with most focus, for faster learning
 - only small part to adjust result in much faster trainging.
 - BackPropFree, RandomFree, BatchTrainFree, CrazyMathFree - only healthy and natural ingredient's.
 - backprop is allays changing everything (or a lot) resulting bad at Plastizitäts-Stabilitäts-Dilemma this is
   tried to fix with batch-training what require shuffled data.
 - backprop is telling what should be. instead we make a (adjustable) balance of what should be and what could be.
 - k-mean like for segmentation (in both directions!) AND `what should be.`
 - crazy-math-free: only ADD, SUB, MUL, MAX, MIN and DIV not required in inner loops


thanks & credits..
----------------
 - Biopsychology: John Pinel, Manfred Spitzer, Paul Flechsig
 - Progamming: demoscene, particle system, k-mean
 - Networks: ANN, CNN. ReLU, Conway's Game of Life
 - Tools: NumPy, Python, MNIST Dataset


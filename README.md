# Code for "Towards a theory of model distillation" 

See https://arxiv.org/abs/2403.09053 for description of the distillation algorithm implemented here. We distill neural networks that implicitly represent decision trees, into the explicitly-represented decision trees.

1. Run first: `train_decision_trees.ipynb` trains ReLU ResNet neural networks to learn several random decision trees, and saves the networks in the `saved_models` folder.

2. Run second: `distill_decision_tree.ipynb` distills the trained neural networks to recover the original decision tree, and saves the reconstructions, plus temporary files, in the `saved_reconstructions` folder. 

* ⚠️ **Warning**: this is academic code intended to demonstrate a proof of concept, and was not optimized to run fast. The goal is to show that in many cases the networks linearly-represent the decision tree's intermediate computations, and that under this condition we have a provably correct and efficient algorithm that extracts those decision trees using linear probes on the trained neural network.

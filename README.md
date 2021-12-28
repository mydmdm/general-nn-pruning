# Model Pruner Interface

## Motivation

- Give users a clear picture on the model pruning algorithms and architecture
- Facilitate on-demand customization of pruners (One Interface, Many Utils)
- Explore the feasibility of hardware aware model compression (e.g., inference latency as the objective)

## Introduction
Generally, a model pruning algorithm is composed by the following key aspects:

- **Target** to answer the question, "*what (scope and granularity) to be pruned*". The pruning granularity could be part of a tensor (e.g., blocks), or a computing concept comprised by multiple tensors (e.g., channels or heads). The finer grained pruning usually performs better (with less accuracy drop) but rarely gain large speedup ratio.
- **Metric** to answer the question, "*which of targets selected to be pruned*". Frequently used metrics include *weights magnitudes* (e.g., $L_1$ or $L_2$ norm), *gradients* and *activations magnitudes*. There are also learnable metrics during fine tuning (*movement*).
- **Allocation** to answer the question, "*how to allocate the sparsity ratio among multiple targets*". The basic allocation method is uniformly distributed, where all targets share the same sparsity ratio. Non-uniform sparsity allocators often apply searching and optimization (e.g., simulated annealing) to find a good solution.
- **Scheduling** to answer the question, *whether the pruning is completed by one time or iteratively*. Frequently used schedulers include one-pass, linear, and AGP.

The above aspects are orthogonal and users could combine them arbitrarily to realize existing algorithms (built-in in NNI) and explore new pruning algorithms. For hardware-aware neural network pruning, for example, pruning model to satisfy the inference latency on mobile device, requires another key aspect. 

- **Objective** to answer the question, "*how to translate the objective (e.g., latency) to sparsity ratio*".

The control flow is a nested loop, the outer of which is scheduler loop, and the inner loop is allocator loop.

```python
spar = ... # initial sparsity attributes
while not scheduler.terminate(): # scheduling loop
    step_objective = scheduler.get_objective() # generate objective for current step
    while not is_achieved(step_objective, model, spar):
        new_spar_specs = allocator.try_more()
        pruner.prune(new_spar_specs)
    speed_up(model, spar) # speeding up if necessary
    fine_tune(model)
```

## One Interface, Many Utils



### Target selector

- *Tensor target*. This category mainly focuses on the important part inside a tensor (weights or activations). The typical granularity in this level is *block*, which is structured part of a tensor. The block is described by the *block sizes* and *axes*. The block size could be special values, such as single-element block (*fine grained*), *row* (or *column*) block, and even covering one or more dimensions. 
- *Operator target*. Some pruning granularity could be conceptual and larger than a single tensor. For example, the *head* in Multi-Head Attention (MHA) module, which is composed 4 weights tensors ($W_{Q,K,V,O}$). So does the *convolution channel*, which consists of weights and bias. Pruners dealing with such granularity need to understand the operator computation. 



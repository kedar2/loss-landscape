# loss-landscape
Experiments are located in the directory ```experiments```.

## Requirements
```
torch
numpy
matplotlib
pandas
tqdm
```

## Evolution of Jacobian rank during training

This experiment trains a two-layer network, and tracks the rank of the Jacobian matrix as it evolves during training. To run the two-layer network experiment, open the directory ```experiments``` and run ```python twolayer.py```.

## Sampling activation regions

This experiment numerically approximates the probability that a randomly initialized two-layer network is of full rank for various scalings of $n$, $d_0$, and $d_1$. To run this experiment, open the directory ```experiments``` and run ```python full_rank_regions.py```. A heatmap will be generated showing the probability of the Jacobian being of full rank for various choices of $n$ and $d_1$. The choice of $d_0$ depends on the experiment settings. The setting ```--experiment varying_input_dim``` will scale $d_0$ as a constant multiple of $n$. For example,

```python full_rank_regions.py --experiment varying_input_dim --ratio 0.5```

will generate a heatmap with $d_0 = 0.5n$ over all entries. The setting ```--experiment fixed_input_dim``` will set $d_0$ to a constant value over the entire heatmap. So

```python full_rank_regions.py --experiment fixed_input_dim --input_dim 1```

will set $d_0 = 1$ over all entries.
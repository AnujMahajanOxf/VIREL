# VIREL
This repo contains the code for VIREL: A Variational Inference Framework for Reinforcement Learning, accepted for NeurIPS 2019. The paper can be found at https://arxiv.org/abs/1811.01132.
The code is based on Vitchyr Pong's rlkit implementation available at https://github.com/vitchyr/rlkit/tree/master/rlkit.
Clone that repo first, our code for virel and beta can be run by adding the files virel.py and beta.py in rlkit/rlkit/torch/sac/. We also provide experiment files virel_exp.py, beta_exp.py to run gym-mujoco tasks on our algorithms, the arguments for which are:

 - 1. env name
 - 2. epochs
 - 3. reward_scale
 - 4. logger name
 - 5. seed
 - 6. beta_scale

Example usage : python beta_exp.py Walker2d-v2 3000 3.0 "-" 1 0.004

A working dockerfile for the environment is also provided for running the experiments on a server.

## Citation

Please use the following bibtex entry for citation:
```
@incollection{NIPS2019_8934,
title = {VIREL: A Variational Inference Framework for Reinforcement Learning},
author = {Fellows, Matthew and Mahajan, Anuj and Rudner, Tim G. J. and Whiteson, Shimon},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {7120--7134},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8934-virel-a-variational-inference-framework-for-reinforcement-learning.pdf}
}

```

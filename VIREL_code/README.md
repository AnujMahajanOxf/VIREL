# VIREL
This repo contains the code for VIREL: A Variational Inference Framework for Reinforcement Learning, accepted for NeurIPS 2019. The paper can be founds at https://arxiv.org/abs/1811.01132.
The code is based on Vitchyr Pong's rlkit implemtation available at https://github.com/vitchyr/rlkit/tree/master/rlkit
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
@article{fellows2018virel,
  title={VIREL: A Variational Inference Framework for Reinforcement Learning},
  author={Fellows, Matthew and Mahajan, Anuj and Rudner, Tim GJ and Whiteson, Shimon},
  journal={arXiv preprint arXiv:1811.01132},
  year={2018}
}

```

# Code for "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"

This repository contains our implementation of PPO and TRPO, with manual toggles
for the code-level optimizations described in our paper. We assume that the user
has a machine with MuJoCo and mujoco_py properly set up and installed, i.e.
you should be able to run the following command on your system without errors:

```python
import gym
gym.make_env("Humanoid-v2")
```

The code itself is quite simple to use. To run the ablation case study discussed
in our paper, you can run the following list of commands:

1. ``cd src/case_study_experiment/``
2. ``python setup_agents.py``
3. ``cd ../``
4. Edit the ``NUM_THREADS`` variables in the ``run_agents.py`` file according to your local machine.
5. Train the agents: ``python run_agents.py case_study_experiment/agent_configs``
6. Open the ``case_study_experiment/Plot_Results.ipynb`` notebook to browse the
results of the experiment as a ``pandas`` DataFrame.

You can easily make a new ``X_experiment`` folder to train new agents. For
example, to run a grid search for ``PPO-M`` hyperparameters, one could create a
new ``X_experiment`` folder, and can edit the ``PARAMS`` variable in
``setup_agents.py`` as follows:

```python
PARAMS = {
    "game": ["Humanoid-v2"], # Can be any MuJoCo environment
    "mode": ["ppo"],
    "out_dir": ["X_experiment/agents"],
    "norm_rewards": ["none"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "ppo_lr_adam": iwt(1e-5, 2.9e-4, 7e-5, 5),
    "val_lr": [1e-4, 2e-4, 3e-4],
    "cpu": [True], # If false, try to use GPU 
    "advanced_logging": [False] # Whether to log trust region stats (KL, ratio),
}
```
See the ``MuJoCo.json`` file for a full list of adjustable parameters.

We have provided the following folders which contain setup and
analysis files for all the experiments in our paper:
- Ablation study: ``case_study_experiment``
- Grid searches for PPO/PPO-M/PPO-NoClip/TRPO/TRPO+: ``get_rewards_experiment``
- KL and Ratio plots: ``trust_region_experiment``

Any of these folders can be run using steps (1-6) above to recreate the results
of our work (though the ``get_rewards_experiment`` does one algorithm at a time,
so one would have to change the ``PARAMS`` attribute a few times to capture all
the algorithms)


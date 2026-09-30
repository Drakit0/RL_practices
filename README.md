# RL practices

Lab work for the Reinforcement Learning course of the Industrial Engineering and Mathematics degree at Universidad Pontificia Comillas ICAI, autumn 2025. The three labs were done as a group by Pablo Tuñón Laguna, Alberto Velasco Rodríguez and Lydia Ruiz Martínez. Reports and READMEs are written in Spanish.

## Lab 1: Bellman equations, policy iteration and value iteration

Folder `Lab1`. Policy iteration and value iteration on a small grid world (`env.py`, 7x5 grid, actions up-right and down-right), in a deterministic version and a stochastic version where the intended move succeeds with probability 0.8. `main.py` runs either algorithm with `--env_behavior` and `--algorithm`. The plots in `fonts` show the suboptimality gap and the value updates per state. `prev-work` holds the pre-lab derivation of the Bellman equations for three states as a function of the step reward. The report is in `analysis`.

The report states that policy iteration reaches the optimal policy in 3 iterations in both versions, and that value iteration reaches the same policy.

## Lab 2: Model-free tabular RL

Folder `Lab2`. SARSA (`sarsa.py`) and Q-learning (`q_learning.py`) with epsilon-greedy exploration on the same grid world, with the start state chosen at random among four states. The scripts plot Q-value convergence for states 13 and 22, rewards per episode and steps per episode, and compare the learned values with the ones from policy and value iteration. The report is in `analysis`.

The report gives, for the four state-action pairs it tracks, absolute errors of 0.00 to 0.06 for SARSA and 0.00 to 0.63 for Q-learning.

## Lab 3: Navigation and warehouse agents

Folder `Lab3`, two sessions, each with a short README.

`sesion1`: a SARSA(0) agent with tile coding (`tiles3.py`, `representacion.py`, `agente.py`) for a continuous 2D navigation environment with obstacles and a target area. Two agents are trained with `entrenar_agente_a.py` (10,000 episodes) and `entrenar_agente_b.py` (50,000 episodes). The saved agents are the `.pkl` files. `evaluar_agentes.py` evaluates them and `visualizar_agente.py` renders an episode.

`sesion2`: a DQN agent (`agente_dqn.py`) with experience replay, a target network and Double DQN, for a warehouse environment with three variants: pick objects at fixed positions, pick and deliver at fixed positions, and pick and deliver with random positions. `representacion_almacen.py` builds the state features, `entrenar_entorno1.py` to `entrenar_entorno3.py` train one agent per variant (each starting from the previous one), and `evaluar_entornos.py` evaluates them. Trained weights are the `.pth` files and the training curves are the `.png` files.

## Running

Python 3 with gymnasium, numpy and matplotlib. Lab 3 session 2 also needs PyTorch. Scripts expect to be run from their own folder, for example:

```
cd Lab1
python main.py --env_behavior stochastic --algorithm value_iteration
```

## Authors

Pablo Tuñón Laguna, Alberto Velasco Rodríguez, Lydia Ruiz Martínez.

## Licence

The code, reports and plots written by the authors are under the MIT licence (see `LICENSE`). The following files were provided by the course and are not covered by it:

- `Lab1/env.py` and `Lab2/env.py` (grid-world environment; the Lab 2 copy has the start-state change)
- the scaffolding in `Lab1/main.py` and in the function signatures of the Lab 1 and Lab 2 scripts
- `Lab1/fonts/Logo ICAI.png` and `Lab2/fonts/Logo ICAI.png`
- `Lab3/sesion1/entorno_navegacion.py`
- `Lab3/sesion2/almacen_alu_v1.py`
- `Lab3/sesion1/tiles3.py` (tile coding by Rich Sutton)

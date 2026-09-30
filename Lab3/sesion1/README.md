# Lab 3, session 1: SARSA with tile coding

A SARSA(0) agent for a continuous 2D navigation environment with obstacles and a target area (`entorno_navegacion.py`, provided by the course).

- `tiles3.py`: tile coding library (Rich Sutton).
- `representacion.py`: turns the agent position into active tiles.
- `agente.py`: the SARSA agent with an epsilon-greedy policy.
- `entrenar_agente_a.py` and `entrenar_agente_b.py`: train agent A (10,000 episodes) and agent B (50,000 episodes) and save them as `agente_grupo_xx_a.pkl` and `agente_grupo_xx_b.pkl`.
- `evaluar_agentes.py`: evaluates the saved agents.
- `visualizar_agente.py`: renders episodes, `python visualizar_agente.py agente_grupo_xx_a.pkl 3`.

Run the scripts from this folder. The `.pkl` files are pickles: only load files you trust.

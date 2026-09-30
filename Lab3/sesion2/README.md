# Lab 3, session 2: DQN in a warehouse

A DQN agent for a warehouse environment (`almacen_alu_v1.py`, provided by the course) with three variants: pick objects at fixed positions, pick and deliver at fixed positions, and pick and deliver with random positions.

- `agente_dqn.py`: network, replay buffer, target network and the agent.
- `representacion_almacen.py`: state features, including distance to obstacles.
- `entrenar_entorno1.py`, `entrenar_entorno2.py`, `entrenar_entorno3.py`: train one agent per variant. The second starts from the weights of the first and the third from the second. They save `entornoN_agente.pth` and `entornoN_training_progress.png`.
- `evaluar_entornos.py`: evaluates the three agents and writes `comparacion_agentes_sesion2.png`. With `--visualize N` it renders environment N.
- `visualizar_agente.py`: renders episodes, `python visualizar_agente.py 1 3`.

Needs PyTorch, numpy and matplotlib. Run the scripts from this folder. The `.pth` files are pickled PyTorch weights: only load files you trust.

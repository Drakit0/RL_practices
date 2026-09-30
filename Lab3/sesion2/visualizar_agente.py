"""
Visualiza un agente DQN entrenado en el entorno de almacén.
Requiere haber ejecutado antes entrenar_entornoX.py

Uso: python3 visualizar_agente.py [1|2|3] [num_episodios]
"""
import sys
from almacen_alu_v1 import WarehouseEnv
from representacion_almacen import WarehouseFeedback  # Usar representación mejorada
from agente_dqn import DQNAgent

# Configuración
entorno = int(sys.argv[1]) if len(sys.argv) > 1 else 1
num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

configs = {
    1: (True, False, 'entorno1_agente.pth'),
    2: (False, False, 'entorno2_agente.pth'),
    3: (False, True, 'entorno3_agente.pth')
}
just_pick, random_obj, archivo = configs[entorno]

# Cargar agente con la representación mejorada (23 features)
print(f"Cargando {archivo}...")
env = WarehouseEnv(just_pick=just_pick, random_objects=random_obj)
feedback = WarehouseFeedback()  # 23 features con proximidad a obstáculos
agent = DQNAgent(feedback.get_feature_size(), env.action_space.n, feedback, hidden_sizes=[128, 64])
agent.load(archivo)

# Visualizar
env = WarehouseEnv(just_pick=just_pick, random_objects=random_obj, render_mode='human')
for ep in range(num_eps):
    print(f"\n--- Episodio {ep+1} ---")
    state, _ = env.reset()
    steps = 0
    done = False
    while steps < 200 and not done:
        action = agent.get_action(state, epsilon=0.01)
        state, _, term, trunc, _ = env.step(action)
        env.render()
        steps += 1
        done = term or trunc
        if done:
            if env.delivery or (just_pick and env.agent_has_object):
                print(f"[ok] Éxito en {steps} pasos")
            elif env.collision:
                print(f"[fail] Colisión en {steps} pasos")
            else:
                print(f"Timeout en {steps} pasos")
    if not done:
        print(f"Timeout en {steps} pasos")
env.close()

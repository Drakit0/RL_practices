"""
Visualiza un agente entrenado en el entorno de navegación.
Requiere haber ejecutado antes entrenar_agente_a.py o entrenar_agente_b.py

Uso: python3 visualizar_agente.py [archivo.pkl] [num_episodios]
"""
import sys
import pickle
from entorno_navegacion import Navegacion

# Cargar agente
archivo = sys.argv[1] if len(sys.argv) > 1 else 'agente_grupo_xx_a.pkl'
num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

print(f"Cargando {archivo}...")
with open(archivo, 'rb') as f:
    agent = pickle.load(f)

# Visualizar
env = Navegacion(render_mode='human')
for ep in range(num_eps):
    print(f"\n--- Episodio {ep+1} ---")
    state, _ = env.reset()
    steps = 0
    while True:
        action = agent.get_action(state, epsilon=0.01)
        state, reward, term, trunc, _ = env.step(action)
        env.render()
        steps += 1
        if term or trunc:
            resultado = "✅ Éxito" if env.target else "❌ Colisión"
            print(f"{resultado} en {steps} pasos")
            break
env.close()

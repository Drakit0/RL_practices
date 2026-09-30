"""
Script para entrenar el Agente B con más episodios según el enunciado de la Sesión 1
"""
import numpy as np
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction
from agente import SarsaAgent
import pickle

# Configuración del entorno
env = Navegacion()
warehouse_width = 10.0
warehouse_height = 10.0
target_area = (2.5, 8, 1.0, 2.0)

# Configuración de tile coding (misma que agente A)
n_tiles_width = 10
n_tiles_height = 10
n_tilings = 8

# Crear representación
feedback = FeedbackConstruction((warehouse_width, warehouse_height), 
                             (n_tiles_width, n_tiles_height), 
                             n_tilings, target_area)

# Crear agente con hiperparámetros optimizados
agent = SarsaAgent(env, feedback, 
                  learning_rate=0.1, 
                  discount_factor=0.99, 
                  epsilon=0.5)

print("=" * 60)
print("ENTRENAMIENTO AGENTE B - 50,000 EPISODIOS")
print("=" * 60)
print(f"Configuración:")
print(f"  - Tiles: {n_tiles_width}x{n_tiles_height}")
print(f"  - Tilings: {n_tilings}")
print(f"  - Learning rate: {agent.learning_rate}")
print(f"  - Discount factor: {agent.discount_factor}")
print(f"  - Epsilon inicial: {agent.epsilon}")
print("=" * 60)

# Entrenar el agente con más episodios (50,000 para mejor convergencia)
agent.train(num_episodes=50000)

# Guardar el agente
filename = 'agente_grupo_xx_b.pkl'
with open(filename, 'wb') as f:
    pickle.dump(agent, f)
print(f"\nAgente B guardado en: {filename}")

# Evaluar el agente con algunos episodios
print("\n" + "=" * 60)
print("EVALUACIÓN PRELIMINAR DEL AGENTE B")
print("=" * 60)
avg_return = agent.evaluate(num_episodes=10)
print(f"Retorno promedio en 10 episodios: {avg_return:.2f}")
print("=" * 60)

"""
Entrenamiento del Agente para ENTORNO 3: Objetos aleatorios, recogida y entrega
Tarea: Recoger un objeto Y entregarlo (objetos en posiciones aleatorias cada episodio)
Este es el entorno más difícil - requiere generalización
"""
import numpy as np
import torch
from almacen_alu_v1 import WarehouseEnv
from representacion_almacen import WarehouseFeedback  # Usar representación mejorada
from agente_dqn import DQNAgent
import matplotlib.pyplot as plt

print("=" * 70)
print("ENTRENAMIENTO ENTORNO 3: OBJETOS ALEATORIOS, RECOGIDA Y ENTREGA")
print("=" * 70)

# Crear entorno
env = WarehouseEnv(just_pick=False, random_objects=True, render_mode=None)
print(f"[ok] Entorno creado")
print(f"  - Tarea: Recoger un objeto Y entregarlo")
print(f"  - Objetos: ALEATORIOS (cambian cada episodio)")
print(f"  - Acciones: {env.action_space.n} (4 movimiento + 1 coger + 1 soltar)")
print(f"  - Desafío: El agente debe GENERALIZAR")

# Crear representación (MEJORADA con proximidad a obstáculos)
feedback = WarehouseFeedback()
state_size = feedback.get_feature_size()
action_size = env.action_space.n
print(f"[ok] Representación: {state_size} features (con proximidad a obstáculos)")

# Crear agente DQN (configuración robusta para generalización)
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    feedback=feedback,
    learning_rate=0.0003,      # LR más bajo para estabilidad con transfer
    gamma=0.99,
    epsilon=0.3,               # Empezar con epsilon bajo (transfer learning)
    epsilon_min=0.05,          # Epsilon mínimo más alto para exploración
    epsilon_decay=0.998,       # Decay moderado
    buffer_size=30000,         # Buffer grande para diversidad
    batch_size=64,
    target_update_freq=10,
    hidden_sizes=[128, 64]     # Misma arquitectura que entornos 1 y 2
)
print(f"[ok] Agente DQN creado")
print(f"  - Learning rate: 0.0003")
print(f"  - Gamma: 0.99")
print(f"  - Epsilon: 0.3 → 0.05 (decay: 0.998)")
print(f"  - Buffer size: 30,000")
print(f"  - Batch size: 64")
print(f"  - Arquitectura: [128, 64]")

# Transfer Learning desde Entorno 2
print("\nTRANSFER LEARNING:")
try:
    # Cargar pesos del Entorno 2 (mismo número de acciones: 6)
    checkpoint = torch.load('entorno2_agente.pth', map_location=agent.device, weights_only=False)
    # El checkpoint es un diccionario con las claves del estado
    if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        print("[ok] Pesos cargados desde entorno2_agente.pth (formato checkpoint)")
    else:
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
        print("[ok] Pesos cargados desde entorno2_agente.pth (formato directo)")
    print("  - Misma arquitectura y acciones - transferencia completa")
except Exception as e:
    print(f"[warn] No se pudo cargar entorno2_agente.pth: {e}")
    print("  - Entrenando desde cero")

print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO (con early stopping)")
print("=" * 70)
print("NOTA: Este entorno requiere MÁS episodios para generalizar")
print("=" * 70)

# Entrenar con early stopping cuando alcance 90% de éxito (margen para variabilidad)
num_episodes = 8000  # Máximo, pero para antes si alcanza objetivo
agent.train(env, num_episodes=num_episodes, verbose=True,
            early_stopping=True, target_success_rate=0.90, patience=1500)

print("\n" + "=" * 70)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 70)

# Guardar agente
agent.save('entorno3_agente.pth')

# Evaluación robusta (3 rondas x 500 episodios según requisitos)
print("\nEVALUACIÓN ROBUSTA:")
results = agent.evaluate_robust(env, num_episodes=500, num_rounds=3)

# Graficar progreso
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Recompensas
axes[0].plot(agent.training_rewards, alpha=0.3, color='blue')
window = 200  # Ventana más grande para suavizar más
moving_avg = np.convolve(agent.training_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(range(window-1, len(agent.training_rewards)), moving_avg, color='red', linewidth=2)
axes[0].set_xlabel('Episodio')
axes[0].set_ylabel('Recompensa Total')
axes[0].set_title('Progreso de Entrenamiento - Entorno 3 (Generalización)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(['Recompensa', f'Media móvil ({window} ep)'])

# Pérdida
if agent.losses:
    axes[1].plot(agent.losses, alpha=0.5, color='green')
    window_loss = min(2000, len(agent.losses) // 10)
    if len(agent.losses) > window_loss:
        moving_avg_loss = np.convolve(agent.losses, np.ones(window_loss)/window_loss, mode='valid')
        axes[1].plot(range(window_loss-1, len(agent.losses)), moving_avg_loss, color='darkgreen', linewidth=2)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Pérdida Durante el Entrenamiento')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('entorno3_training_progress.png', dpi=150, bbox_inches='tight')
print(f"\n[ok] Gráficas guardadas en: entorno3_training_progress.png")

print("\n" + "=" * 70)
print("RESUMEN ENTORNO 3")
print("=" * 70)
print(f"Episodios entrenados: {num_episodes}")
print(f"Tasa de éxito final: {results['success_rate']:.2f}%")
print(f"Recompensa promedio final: {results['avg_reward']:.2f}")
print(f"Agente guardado: entorno3_agente.pth")
print("=" * 70)

env.close()

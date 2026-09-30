"""
Entrenamiento del Agente para ENTORNO 1: Objetos fijos, solo recogida
Tarea: Aproximarse a cualquier objeto y recogerlo exitosamente
"""
import numpy as np
from almacen_alu_v1 import WarehouseEnv
from representacion_almacen import WarehouseFeedback
from agente_dqn import DQNAgent
import matplotlib.pyplot as plt

print("=" * 70)
print("ENTRENAMIENTO ENTORNO 1: OBJETOS FIJOS, SOLO RECOGIDA")
print("=" * 70)

# Crear entorno
env = WarehouseEnv(just_pick=True, random_objects=False, render_mode=None)
print(f"[ok] Entorno creado")
print(f"  - Tarea: Recoger un objeto")
print(f"  - Objetos: Fijos en posiciones predefinidas")
print(f"  - Acciones: {env.action_space.n} (4 movimiento + 1 coger)")

# Usar representación RICA con features engineered
feedback = WarehouseFeedback()
state_size = feedback.get_feature_size()
action_size = env.action_space.n
print(f"[ok] Representación RICA: {state_size} features")
print(f"  - Incluye: posición, distancias, direcciones a objetos")

# Crear agente DQN con arquitectura optimizada
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    feedback=feedback,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    target_update_freq=10,
    hidden_sizes=[128, 64]      # Red más grande para más features
)
print(f"[ok] Agente DQN (Double DQN) creado")
print(f"  - Learning rate: 0.001")
print(f"  - Gamma: 0.99")
print(f"  - Epsilon: 1.0 → 0.01 (decay: 0.995)")
print(f"  - Buffer size: 10,000")
print(f"  - Batch size: 64")
print(f"  - Arquitectura: [128, 64]")
print(f"  - Target update: Hard, cada 10 episodios")
print(f"  - Checkpointing: Mejor modelo guardado")

print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO (con early stopping)")
print("=" * 70)

# Entrenar con early stopping cuando alcance 95% de éxito
num_episodes = 5000  # Máximo, pero para antes si alcanza objetivo
agent.train(env, num_episodes=num_episodes, verbose=True,
            early_stopping=True, target_success_rate=0.96, patience=600)

print("\n" + "=" * 70)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 70)

# Guardar agente
agent.save('entorno1_agente.pth')

# Evaluación robusta (3 rondas x 500 episodios según requisitos)
print("\nEVALUACIÓN ROBUSTA:")
results = agent.evaluate_robust(env, num_episodes=500, num_rounds=3)

# Graficar progreso
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Recompensas
axes[0].plot(agent.training_rewards, alpha=0.3, color='blue')
window = 100
moving_avg = np.convolve(agent.training_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(range(window-1, len(agent.training_rewards)), moving_avg, color='red', linewidth=2)
axes[0].set_xlabel('Episodio')
axes[0].set_ylabel('Recompensa Total')
axes[0].set_title('Progreso de Entrenamiento - Entorno 1')
axes[0].grid(True, alpha=0.3)
axes[0].legend(['Recompensa', f'Media móvil ({window} ep)'])

# Pérdida
if agent.losses:
    axes[1].plot(agent.losses, alpha=0.5, color='green')
    window_loss = min(1000, len(agent.losses) // 10)
    if len(agent.losses) > window_loss:
        moving_avg_loss = np.convolve(agent.losses, np.ones(window_loss)/window_loss, mode='valid')
        axes[1].plot(range(window_loss-1, len(agent.losses)), moving_avg_loss, color='darkgreen', linewidth=2)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Pérdida Durante el Entrenamiento')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('entorno1_training_progress.png', dpi=150, bbox_inches='tight')
print(f"\n[ok] Gráficas guardadas en: entorno1_training_progress.png")

print("\n" + "=" * 70)
print("RESUMEN ENTORNO 1")
print("=" * 70)
print(f"Episodios entrenados: {num_episodes}")
print(f"Tasa de éxito final: {results['success_rate']:.2f}%")
print(f"Recompensa promedio final: {results['avg_reward']:.2f}")
print(f"Agente guardado: entorno1_agente.pth")
print("=" * 70)

env.close()

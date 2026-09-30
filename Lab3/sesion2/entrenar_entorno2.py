"""
Entrenamiento del Agente para ENTORNO 2: Objetos fijos, recogida y entrega
Tarea: Recoger un objeto Y entregarlo en la zona de entrega
Usa transfer learning desde el agente del entorno 1
"""
import numpy as np
import torch
from almacen_alu_v1 import WarehouseEnv
from representacion_almacen import WarehouseFeedback
from agente_dqn import DQNAgent
import matplotlib.pyplot as plt

print("=" * 70)
print("ENTRENAMIENTO ENTORNO 2: OBJETOS FIJOS, RECOGIDA Y ENTREGA")
print("=" * 70)

# Crear entorno
env = WarehouseEnv(just_pick=False, random_objects=False, render_mode=None)
print(f"[ok] Entorno creado")
print(f"  - Tarea: Recoger un objeto Y entregarlo")
print(f"  - Objetos: Fijos en posiciones predefinidas")
print(f"  - Acciones: {env.action_space.n} (4 movimiento + 1 coger + 1 soltar)")

# Usar representación RICA (igual que entorno 1)
feedback = WarehouseFeedback()
state_size = feedback.get_feature_size()
action_size = env.action_space.n
print(f"[ok] Representación RICA: {state_size} features")

# Crear agente DQN (misma arquitectura que entorno 1 pero con 6 acciones)
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    feedback=feedback,
    learning_rate=0.0005,      # LR moderado
    gamma=0.99,
    epsilon=0.5,               # Más exploración inicial para encontrar entregas
    epsilon_min=0.02,          # Epsilon bajo para explotar bien en objetos fijos
    epsilon_decay=0.996,       # Decay más rápido
    buffer_size=10000,         # Buffer pequeño: objetos fijos = menos diversidad necesaria
    batch_size=64,
    target_update_freq=10,
    hidden_sizes=[128, 64]     # Misma arquitectura que entorno 1
)

# --- TRANSFER LEARNING: Cargar pesos del entorno 1 ---
print("\n--- TRANSFER LEARNING ---")
try:
    checkpoint = torch.load('entorno1_agente.pth', weights_only=False)
    
    pretrained_state = checkpoint['policy_net_state_dict']
    current_state = agent.policy_net.state_dict()
    
    # Copiar capas que coinciden en tamaño
    for name, param in pretrained_state.items():
        if name in current_state and param.shape == current_state[name].shape:
            current_state[name] = param
            print(f"  [ok] Cargada capa: {name}")
        elif name in current_state:
            # La última capa tiene diferente tamaño (5 vs 6 acciones)
            if 'weight' in name and param.shape[0] == 5:
                current_state[name][:5, :] = param
                print(f"  [ok] Cargada parcialmente: {name} (5 de 6 acciones)")
            elif 'bias' in name and param.shape[0] == 5:
                current_state[name][:5] = param
                print(f"  [ok] Cargada parcialmente: {name} (5 de 6 acciones)")
    
    agent.policy_net.load_state_dict(current_state)
    agent.target_net.load_state_dict(current_state)
    print("[ok] Transfer learning completado desde entorno1_agente.pth")
except FileNotFoundError:
    print("[warn] No se encontró entorno1_agente.pth - entrenando desde cero")

print(f"\n[ok] Agente DQN (Double DQN + Transfer Learning + Prioritized Replay) creado")
print(f"  - Learning rate: 0.0005")
print(f"  - Gamma: 0.99")
print(f"  - Epsilon: 0.3 → 0.01 (decay: 0.998)")
print(f"  - Buffer size: 20,000 (con priorización)")
print(f"  - Batch size: 64")
print(f"  - Arquitectura: [128, 64]")

print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO (con early stopping)")
print("=" * 70)

# Entrenar con early stopping cuando alcance 92% de éxito
num_episodes = 10000  # Máximo, pero para antes si alcanza objetivo
agent.train(env, num_episodes=num_episodes, verbose=True,
            early_stopping=True, target_success_rate=0.92, patience=1500, eval_freq=200)

print("\n" + "=" * 70)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 70)

# Guardar agente
agent.save('entorno2_agente.pth')

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
axes[0].set_title('Progreso de Entrenamiento - Entorno 2')
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
plt.savefig('entorno2_training_progress.png', dpi=150, bbox_inches='tight')
print(f"\n[ok] Gráficas guardadas en: entorno2_training_progress.png")

print("\n" + "=" * 70)
print("RESUMEN ENTORNO 2")
print("=" * 70)
print(f"Episodios entrenados: {num_episodes}")
print(f"Tasa de éxito final: {results['success_rate']:.2f}%")
print(f"Recompensa promedio final: {results['avg_reward']:.2f}")
print(f"Agente guardado: entorno2_agente.pth")
print("=" * 70)

env.close()

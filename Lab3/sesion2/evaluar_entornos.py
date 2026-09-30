"""
Evaluación completa de los 3 agentes entrenados para los entornos de almacén
"""
import numpy as np
import torch
from almacen_alu_v1 import WarehouseEnv
from representacion_almacen import WarehouseFeedback  # Usar representación mejorada
from agente_dqn import DQNAgent
import matplotlib.pyplot as plt

def evaluate_agent(agent_path, env, env_name, num_episodes=500, num_rounds=3):
    """
    Evalúa un agente guardado con evaluación robusta (múltiples rondas)
    """
    print(f"\n{'='*70}")
    print(f"EVALUANDO: {env_name}")
    print(f"{'='*70}")
    
    # Cargar agente - usar representación mejorada y arquitectura correcta
    feedback = WarehouseFeedback()  # 23 features con proximidad a obstáculos
    state_size = feedback.get_feature_size()
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        feedback=feedback,
        hidden_sizes=[128, 64]  # Arquitectura usada en entrenamiento
    )
    
    try:
        agent.load(agent_path)
    except FileNotFoundError:
        print(f"[fail] Error: No se encontró {agent_path}")
        print(f"   Por favor, entrena primero el agente correspondiente.")
        return None
    
    # Evaluación robusta (múltiples rondas)
    results = agent.evaluate_robust(env, num_episodes=num_episodes, num_rounds=num_rounds)
    
    return results


def compare_agents():
    """
    Compara el rendimiento de los 3 agentes
    """
    print("=" * 70)
    print("EVALUACIÓN COMPLETA DE AGENTES - SESIÓN 2")
    print("=" * 70)
    
    # Configuración de entornos
    configs = [
        {
            'name': 'Entorno 1: Objetos fijos, solo recogida',
            'path': 'entorno1_agente.pth',
            'env': WarehouseEnv(just_pick=True, random_objects=False, render_mode=None)
        },
        {
            'name': 'Entorno 2: Objetos fijos, recogida y entrega',
            'path': 'entorno2_agente.pth',
            'env': WarehouseEnv(just_pick=False, random_objects=False, render_mode=None)
        },
        {
            'name': 'Entorno 3: Objetos aleatorios, recogida y entrega',
            'path': 'entorno3_agente.pth',
            'env': WarehouseEnv(just_pick=False, random_objects=True, render_mode=None)
        }
    ]
    
    results = []
    
    for config in configs:
        result = evaluate_agent(
            config['path'],
            config['env'],
            config['name'],
            num_episodes=500,  # 500 episodios según requisitos
            num_rounds=3       # 3 rondas para robustez
        )
        if result:
            results.append({
                'name': config['name'],
                'result': result
            })
        config['env'].close()
    
    # Resumen comparativo
    if results:
        print("\n" + "=" * 70)
        print("COMPARACIÓN DE RESULTADOS")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparación de Agentes - Sesión 2', fontsize=16, fontweight='bold')
        
        names = [r['name'].split(':')[0] for r in results]
        success_rates = [r['result']['success_rate'] for r in results]
        collision_rates = [r['result']['collision_rate'] for r in results]
        avg_rewards = [r['result']['avg_reward'] for r in results]
        std_rewards = [r['result']['std_reward'] for r in results]
        
        # Gráfico 1: Tasa de éxito
        axes[0, 0].bar(names, success_rates, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        axes[0, 0].set_ylabel('Tasa de Éxito (%)', fontsize=12)
        axes[0, 0].set_title('Tasa de Éxito por Entorno', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim([0, 105])
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Gráfico 2: Tasa de colisión
        axes[0, 1].bar(names, collision_rates, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        axes[0, 1].set_ylabel('Tasa de Colisión (%)', fontsize=12)
        axes[0, 1].set_title('Tasa de Colisión por Entorno', fontsize=12, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(collision_rates):
            axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Gráfico 3: Recompensa promedio
        axes[1, 0].bar(names, avg_rewards, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        axes[1, 0].errorbar(names, avg_rewards, yerr=std_rewards, fmt='none', ecolor='black', capsize=5)
        axes[1, 0].set_ylabel('Recompensa Promedio', fontsize=12)
        axes[1, 0].set_title('Recompensa Promedio ± Desviación Estándar', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        for i, (v, std) in enumerate(zip(avg_rewards, std_rewards)):
            axes[1, 0].text(i, v + std + 5, f'{v:.1f}', ha='center', fontweight='bold')
        
        # Gráfico 4: Tabla de resumen
        axes[1, 1].axis('off')
        table_data = []
        for r in results:
            table_data.append([
                r['name'].split(':')[0],
                f"{r['result']['success_rate']:.1f}%",
                f"{r['result']['collision_rate']:.1f}%",
                f"{r['result']['avg_reward']:.1f} ± {r['result']['std_reward']:.1f}"
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Entorno', 'Éxito', 'Colisión', 'Recompensa'],
            cellLoc='center',
            loc='center',
            colWidths=[0.35, 0.15, 0.15, 0.35]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estilo de la tabla
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        plt.savefig('comparacion_agentes_sesion2.png', dpi=150, bbox_inches='tight')
        print("\n[ok] Gráfica comparativa guardada en: comparacion_agentes_sesion2.png")
        
        # Tabla en consola
        print("\n" + "=" * 70)
        print(f"{'Entorno':<40} {'Éxito':<12} {'Colisión':<12} {'Recompensa':<20}")
        print("=" * 70)
        for r in results:
            print(f"{r['name']:<40} "
                  f"{r['result']['success_rate']:>6.1f}%     "
                  f"{r['result']['collision_rate']:>6.1f}%     "
                  f"{r['result']['avg_reward']:>6.1f} ± {r['result']['std_reward']:.1f}")
        print("=" * 70)
    
    print("\n[ok] Evaluación completada")


def visualize_agent(agent_path, env, env_name, num_episodes=3):
    """
    Visualiza algunos episodios de un agente
    """
    print(f"\n{'='*70}")
    print(f"VISUALIZANDO: {env_name}")
    print(f"{'='*70}")
    
    # Cargar agente - usar representación mejorada y arquitectura correcta
    feedback = WarehouseFeedback()  # 23 features con proximidad a obstáculos
    state_size = feedback.get_feature_size()
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        feedback=feedback,
        hidden_sizes=[128, 64]  # Arquitectura usada en entrenamiento
    )
    
    try:
        agent.load(agent_path)
    except FileNotFoundError:
        print(f"[fail] Error: No se encontró {agent_path}")
        return
    
    # Crear entorno con render
    env_render = type(env)(
        just_pick=env.just_pick,
        random_objects=env.random_objects,
        render_mode='human'
    )
    
    for episode in range(num_episodes):
        print(f"\nEpisodio {episode + 1}/{num_episodes}")
        state, _ = env_render.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = agent.get_action(state, epsilon=0.0)  # Greedy
            next_state, reward, terminated, truncated, _ = env_render.step(action)
            done = terminated or truncated
            
            env_render.render()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        print(f"  Recompensa: {total_reward:.1f}, Pasos: {steps}")
        if env_render.delivery:
            print("  [ok] Entrega exitosa")
        elif env_render.collision:
            print("  [fail] Colisión")
    
    env_render.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--visualize':
        # Modo visualización
        entorno_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        
        configs = [
            ('entorno1_agente.pth', WarehouseEnv(just_pick=True, random_objects=False), 'Entorno 1'),
            ('entorno2_agente.pth', WarehouseEnv(just_pick=False, random_objects=False), 'Entorno 2'),
            ('entorno3_agente.pth', WarehouseEnv(just_pick=False, random_objects=True), 'Entorno 3')
        ]
        
        if 1 <= entorno_num <= 3:
            visualize_agent(*configs[entorno_num - 1])
        else:
            print("Uso: python3 evaluar_entornos.py --visualize [1|2|3]")
    else:
        # Modo evaluación completa
        compare_agents()

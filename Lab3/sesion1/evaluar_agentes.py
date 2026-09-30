"""
Script para evaluar los agentes entrenados con 1000 episodios según el enunciado de la Sesión 1
Usa posiciones iniciales pseudoaleatorias reproducibles
"""
import numpy as np
import pickle
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction

def evaluar_agente(agente_filename, num_episodes=1000, seed=42):
    """
    Evalúa un agente guardado durante num_episodes episodios
    
    Args:
        agente_filename: Nombre del archivo .pkl con el agente
        num_episodes: Número de episodios de evaluación
        seed: Semilla para reproducibilidad
    """
    # Cargar agente
    print(f"\nCargando agente desde: {agente_filename}")
    with open(agente_filename, 'rb') as f:
        agent = pickle.load(f)
    
    # Configurar entorno con semilla para reproducibilidad
    env = agent.env
    np.random.seed(seed)
    
    # Métricas de evaluación
    total_returns = []
    exitos = 0
    colisiones = 0
    longitudes = []
    
    print(f"Evaluando durante {num_episodes} episodios...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_return = 0
        steps = 0
        terminated = False
        
        while not terminated:
            # Política greedy (epsilon muy bajo)
            action = agent.get_action(state, epsilon=0.01)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_return += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Registrar métricas
        total_returns.append(total_return)
        longitudes.append(steps)
        
        # Verificar si llegó al objetivo o colisionó
        if env.target:
            exitos += 1
        if env.collision:
            colisiones += 1
        
        # Mostrar progreso cada 100 episodios
        if (episode + 1) % 100 == 0:
            print(f"Episodios completados: {episode + 1}/{num_episodes}")
    
    # Calcular estadísticas
    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    avg_steps = np.mean(longitudes)
    tasa_exito = (exitos / num_episodes) * 100
    tasa_colision = (colisiones / num_episodes) * 100
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print(f"RESULTADOS DE EVALUACIÓN - {agente_filename}")
    print("=" * 60)
    print(f"Episodios evaluados: {num_episodes}")
    print(f"Retorno promedio: {avg_return:.2f} ± {std_return:.2f}")
    print(f"Pasos promedio: {avg_steps:.2f}")
    print(f"Tasa de éxito: {tasa_exito:.2f}%")
    print(f"Tasa de colisión: {tasa_colision:.2f}%")
    print("=" * 60)
    
    return {
        'returns': total_returns,
        'avg_return': avg_return,
        'std_return': std_return,
        'avg_steps': avg_steps,
        'tasa_exito': tasa_exito,
        'tasa_colision': tasa_colision
    }

if __name__ == "__main__":
    print("=" * 60)
    print("EVALUACIÓN DE AGENTES - SESIÓN 1")
    print("=" * 60)
    
    # Evaluar Agente A (10,000 episodios de entrenamiento)
    try:
        resultados_a = evaluar_agente('agente_grupo_xx_a.pkl', num_episodes=1000, seed=42)
    except FileNotFoundError:
        print("\nERROR: No se encontró agente_grupo_xx_a.pkl")
        print("Por favor, ejecuta primero: python entrenar_agente_a.py")
    
    print("\n")
    
    # Evaluar Agente B (más episodios de entrenamiento)
    try:
        resultados_b = evaluar_agente('agente_grupo_xx_b.pkl', num_episodes=1000, seed=42)
    except FileNotFoundError:
        print("\nERROR: No se encontró agente_grupo_xx_b.pkl")
        print("Por favor, ejecuta primero: python entrenar_agente_b.py")
    
    print("\n" + "=" * 60)
    print("EVALUACIÓN COMPLETADA")
    print("=" * 60)

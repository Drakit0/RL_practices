"""
Agente DQN (Deep Q-Network) para resolver los entornos de almacén
Implementación más potente que SARSA lineal, adecuada para espacios de estado complejos
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle

class DQNNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q
    """
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Capas ocultas
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Buffer de experiencia para Experience Replay con priorización simple
    SOLO experiencias con recompensas POSITIVAS tienen mayor probabilidad de ser sampleadas
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        # Prioridad: boost SOLO para éxitos, no para fracasos
        if reward >= 100:  # Éxito grande (recoger o entregar)
            priority = 10.0
        elif reward > 0:
            priority = 3.0
        else:
            priority = 1.0  # Experiencias negativas tienen prioridad base
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        n = len(self.buffer)
        if n < batch_size:
            batch_size = n
        
        # Convertir prioridades a probabilidades
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array / priorities_array.sum()
        
        # Samplear con probabilidades ponderadas
        replace = n < batch_size * 2
        indices = np.random.choice(n, batch_size, p=probabilities, replace=replace)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN con Experience Replay y Target Network
    """
    def __init__(self, state_size, action_size, feedback=None,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=10, hidden_sizes=[128, 128, 64],
                 tau=0.005):  # Soft update parameter
        
        self.state_size = state_size
        self.action_size = action_size
        self.feedback = feedback
        
        # Hiperparámetros
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau  # Para soft update
        
        # Redes Q (policy y target)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador y función de pérdida
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss - más estable que MSE
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Métricas
        self.training_rewards = []
        self.training_lengths = []
        self.losses = []
        self.update_count = 0
        
    def get_action(self, state, epsilon=None):
        """
        Selecciona acción usando política epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        
        # Procesar estado si hay feedback
        if self.feedback is not None:
            state = self.feedback.process_observation(state)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        Almacena experiencia en el replay buffer
        """
        if self.feedback is not None:
            state = self.feedback.process_observation(state)
            next_state = self.feedback.process_observation(next_state)
        
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Entrena la red usando un batch del replay buffer
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Samplear batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Escalar recompensas para estabilidad (sin normalizar)
        rewards = rewards / 100.0
        
        # Q-values actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-values objetivo usando DOUBLE DQN
        # La policy net selecciona la mejor acción, la target net evalúa
        with torch.no_grad():
            # Policy net selecciona las mejores acciones
            best_actions = self.policy_net(next_states).argmax(1)
            # Target net evalúa esas acciones
            next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular pérdida
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Hard update de la target network (copia completa)
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env, num_episodes=1000, verbose=True, early_stopping=False, 
              target_success_rate=None, patience=500, eval_freq=100):
        """
        Entrena el agente
        
        Args:
            env: Entorno de gymnasium
            num_episodes: Número máximo de episodios
            verbose: Mostrar progreso
            early_stopping: Si True, para cuando se estabiliza
            target_success_rate: Tasa de éxito objetivo (ej: 0.90 para 90%)
            patience: Episodios sin mejora antes de parar
            eval_freq: Frecuencia de evaluación para early stopping
        """
        best_avg_reward = float('-inf')
        best_success_rate = 0.0
        episodes_without_improvement = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Seleccionar acción
                action = self.get_action(state)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Almacenar experiencia
                self.remember(state, action, reward, next_state, done)
                
                # Entrenar
                loss = self.replay()
                if loss is not None:
                    self.losses.append(loss)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Actualizar target network periódicamente
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            # Decrecer epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Guardar métricas
            self.training_rewards.append(total_reward)
            self.training_lengths.append(steps)
            
            # Mostrar progreso y guardar mejor modelo
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if len(self.training_rewards) >= 100 else np.mean(self.training_rewards)
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else (np.mean(self.losses) if self.losses else 0)
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.memory)}")
                
                # Guardar modelo si supera umbral Y es mejor que el anterior
                if avg_reward > best_avg_reward and episode > 100:
                    best_avg_reward = avg_reward
                    episodes_without_improvement = 0
                    self._save_best_weights()
                    if verbose:
                        print(f"  ✓ Nuevo mejor modelo guardado (reward: {avg_reward:.2f})")
                else:
                    episodes_without_improvement += 100
            
            # Early stopping check
            if early_stopping and episode > 0 and episode % eval_freq == 0 and episode >= 500:
                # Evaluación rápida
                quick_eval = self._quick_evaluate(env, num_episodes=100)
                success_rate = quick_eval['success_rate'] / 100.0
                
                if verbose:
                    print(f"  → Eval rápida: {quick_eval['success_rate']:.1f}% éxito")
                
                # Verificar si alcanzamos objetivo
                if target_success_rate and success_rate >= target_success_rate:
                    if verbose:
                        print(f"\n✓ EARLY STOPPING: Objetivo alcanzado ({success_rate*100:.1f}% >= {target_success_rate*100:.1f}%)")
                    break
                
                # Verificar estabilización
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    episodes_without_improvement = 0
                
                if episodes_without_improvement >= patience:
                    if verbose:
                        print(f"\n✓ EARLY STOPPING: Estabilizado ({patience} eps sin mejora)")
                    break
    
    def _quick_evaluate(self, env, num_episodes=100):
        """Evaluación rápida para early stopping"""
        old_epsilon = self.epsilon
        success_count = 0
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.get_action(state, epsilon=0.01)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            if env.delivery or (env.just_pick and env.agent_has_object):
                success_count += 1
        
        self.epsilon = old_epsilon
        return {'success_rate': (success_count / num_episodes) * 100}
    
    def evaluate(self, env, num_episodes=100, render=False):
        """
        Evalúa el agente
        """
        total_rewards = []
        success_count = 0
        collision_count = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state, epsilon=0.01)  # Casi greedy en evaluación
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if render:
                    env.render()
                
                state = next_state
                total_reward += reward
                
                if done:
                    # Para just_pick=True, éxito es coger objeto (agent_has_object=True)
                    # Para just_pick=False, éxito es entregar (delivery=True)
                    if env.delivery or (env.just_pick and env.agent_has_object):
                        success_count += 1
                    elif env.collision:
                        collision_count += 1
                    break
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        success_rate = (success_count / num_episodes) * 100
        collision_rate = (collision_count / num_episodes) * 100
        
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN: {num_episodes} episodios")
        print(f"{'='*60}")
        print(f"Recompensa promedio: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Tasa de éxito: {success_rate:.2f}%")
        print(f"Tasa de colisión: {collision_rate:.2f}%")
        print(f"{'='*60}\n")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'collision_rate': collision_rate
        }
    
    def evaluate_robust(self, env, num_episodes=500, num_rounds=3):
        """
        Evaluación robusta: ejecuta múltiples rondas y calcula estadísticas.
        
        Args:
            env: Entorno
            num_episodes: Episodios por ronda
            num_rounds: Número de rondas de evaluación
        
        Returns:
            dict con media y desviación de las métricas
        """
        all_success_rates = []
        all_collision_rates = []
        all_rewards = []
        
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN ROBUSTA: {num_rounds} rondas x {num_episodes} episodios")
        print(f"{'='*60}")
        
        for round_num in range(num_rounds):
            success_count = 0
            collision_count = 0
            round_rewards = []
            
            for _ in range(num_episodes):
                state, _ = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action = self.get_action(state, epsilon=0.01)
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                
                round_rewards.append(total_reward)
                if env.delivery or (env.just_pick and env.agent_has_object):
                    success_count += 1
                elif env.collision:
                    collision_count += 1
            
            success_rate = (success_count / num_episodes) * 100
            collision_rate = (collision_count / num_episodes) * 100
            avg_reward = np.mean(round_rewards)
            
            all_success_rates.append(success_rate)
            all_collision_rates.append(collision_rate)
            all_rewards.append(avg_reward)
            
            print(f"  Ronda {round_num + 1}: {success_rate:.1f}% éxito, {collision_rate:.1f}% colisión, reward: {avg_reward:.1f}")
        
        # Calcular estadísticas finales
        final_success = np.mean(all_success_rates)
        final_success_std = np.std(all_success_rates)
        final_collision = np.mean(all_collision_rates)
        final_reward = np.mean(all_rewards)
        final_reward_std = np.std(all_rewards)
        
        print(f"{'='*60}")
        print(f"RESULTADO FINAL (media de {num_rounds} rondas):")
        print(f"  Tasa de éxito: {final_success:.2f}% ± {final_success_std:.2f}%")
        print(f"  Tasa de colisión: {final_collision:.2f}%")
        print(f"  Recompensa: {final_reward:.2f} ± {final_reward_std:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'success_rate': final_success,
            'success_rate_std': final_success_std,
            'collision_rate': final_collision,
            'avg_reward': final_reward,
            'std_reward': final_reward_std,
            'all_success_rates': all_success_rates
        }
    
    def _save_best_weights(self):
        """Guarda los mejores pesos internamente"""
        self._best_weights = {
            'policy_net': {k: v.clone() for k, v in self.policy_net.state_dict().items()},
            'target_net': {k: v.clone() for k, v in self.target_net.state_dict().items()}
        }
    
    def _load_best_weights(self):
        """Carga los mejores pesos guardados"""
        if hasattr(self, '_best_weights'):
            self.policy_net.load_state_dict(self._best_weights['policy_net'])
            self.target_net.load_state_dict(self._best_weights['target_net'])
            print("✓ Cargados los mejores pesos del entrenamiento")
    
    def save(self, filepath, use_best=False):
        """
        Guarda el agente
        """
        if use_best and hasattr(self, '_best_weights'):
            self._load_best_weights()
            print("✓ Usando los mejores pesos del entrenamiento")
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_lengths': self.training_lengths,
            'losses': self.losses
        }, filepath)
        print(f"Agente guardado en: {filepath}")
    
    def load(self, filepath):
        """
        Carga el agente
        """
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_rewards = checkpoint['training_rewards']
        self.training_lengths = checkpoint['training_lengths']
        self.losses = checkpoint['losses']
        print(f"Agente cargado desde: {filepath}")


if __name__ == "__main__":
    print("Módulo DQN Agent cargado correctamente")
    print(f"Device disponible: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

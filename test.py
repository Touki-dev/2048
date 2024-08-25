import numpy as np
import tensorflow as tf
from game_env import GameEnv  # Assurez-vous que cette classe est correctement définie
from DQN_agent import DQNAgent  # Si vous avez besoin d'utiliser d'autres fonctionnalités de l'agent
import time

# Charger le modèle sauvegardé
model = tf.keras.models.load_model('models/2048-dqn-0.weights.h5')

# Initialiser l'environnement
env = GameEnv(grid_size=4)
state_size = env.grid_size * env.grid_size

# Réinitialiser l'environnement pour commencer un nouvel épisode
state = env.reset()
state = np.reshape(state, [1, state_size])

done = False
total_score = 0

while not done:
    env.render()  # Affiche l'état actuel du jeu

    # L'agent prédit la meilleure action
    action = np.argmax(model.predict(state))
    
    # Appliquez l'action et récupérez l'état suivant, la récompense, et si le jeu est terminé
    next_state, reward, done = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])

    state = next_state  # Mettre à jour l'état actuel
    total_score += reward  # Accumulez le score

    if done:
        print(f"Game Over! Final score: {total_score}")
        break

    # Vous pouvez ajouter un temps d'attente pour observer chaque étape
    time.sleep(0.1)

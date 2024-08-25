import numpy as np
import logic

class GameEnv:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.matrix = logic.new_game(self.grid_size)
        self.score = 0
        return np.array(self.matrix)

    def step(self, action):
        if action == 0:  # Haut
            self.matrix, done = logic.up(self.matrix)
        elif action == 1:  # Bas
            self.matrix, done = logic.down(self.matrix)
        elif action == 2:  # Gauche
            self.matrix, done = logic.left(self.matrix)
        elif action == 3:  # Droite
            self.matrix, done = logic.right(self.matrix)
        
        reward = np.sum(self.matrix) - self.score  # Récompense basée sur l'augmentation du score
        self.score = np.sum(self.matrix)
        
        if done:
            self.matrix = logic.add_two(self.matrix)
        
        state = np.array(self.matrix)
        game_status = logic.game_state(self.matrix)
        done = game_status == 'lose' or game_status == 'win'
        
        return state, reward, done

    def render(self):
        for row in self.matrix:
            print("\t".join(map(str, row)))
        print()

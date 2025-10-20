from SeaWorldEnv import SeaWorldEnv
from Q_learning import train_q_learning, visualize_q_table

# User Settings:
train = True                   # Toggle training
visualize_results = True       # Toggle visualization

# Q-learning Hyperparameters:
learning_rate = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
no_episodes = 1000

# Grid & Environment Configuration:
grid_size = 5
goal_coordinates = (4, 4)
hell_state_coordinates = [(0, 3), (2, 1), (2, 3)]

# Create environment:


def create_env():
    env = SeaWorldEnv(grid_size=grid_size)

    # Add danger states (obstacles):
    for danger in hell_state_coordinates:
        env.add_danger(danger)

    return env


# Main Execution:
if __name__ == "__main__":

    if train:
        env = create_env()
        train_q_learning(env=env,
                         no_episodes=no_episodes,
                         epsilon=epsilon,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         alpha=learning_rate,
                         gamma=gamma,
                         q_table_save_path="q_table.npy")

    if visualize_results:
        visualize_q_table(q_values_path="q_table.npy",
                          hell_state_coordinates=hell_state_coordinates,
                          goal_coordinates=goal_coordinates,
                          actions=["Up", "Down", "Right", "Left"])

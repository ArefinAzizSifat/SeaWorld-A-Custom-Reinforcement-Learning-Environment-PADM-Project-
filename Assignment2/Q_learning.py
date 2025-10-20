import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):
    """
    Trains a Q-learning agent on the given environment.
    Saves the Q-table to a .npy file.
    """

    # Initialize the Q-table:
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        # Take steps until episode ends
        while True:
            # Exploration vs. Exploitation:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward

            # Q-value update:
            q_table[state][action] += alpha * (
                reward + gamma *
                np.max(q_table[next_state]) - q_table[state][action]
            )

            state = next_state
            env.render()

            if done:
                break

        # Decay epsilon:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(
            f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")

    # Close the environment:
    env.close()
    print("Training finished.\n")

    # Save the Q-table:
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
def visualize_q_table(
        q_values_path="q_table.npy",
        hell_state_coordinates=[(2, 1), (0, 3), (2, 3)],
        goal_coordinates=(4, 4),
        actions=["Up", "Down", "Right", "Left"]):

    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Create mask:
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for danger in hell_state_coordinates:
                mask[danger] = True

            # Draw heatmap:
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            # Mark goal and hell:
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            for h in hell_state_coordinates:
                ax.text(h[1] + 0.5, h[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f"Action: {action}")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

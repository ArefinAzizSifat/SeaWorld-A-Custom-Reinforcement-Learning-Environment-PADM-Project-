# 🌊 SeaWorld - Custom Reinforcement Learning Environment

> 🎓 Developed as part of the **M.Eng. AI Engineering of Autonomous Systems** coursework at Technische Hochschule Ingolstadt.  
> 🧠 Designed for experimenting with **Q-learning** and **Deep Q-Networks (DQN)** in a simple but visually interpretable underwater grid world.

---

## 📖 Overview

**SeaWorld** is a custom reinforcement learning environment built with [Gymnasium](https://gymnasium.farama.org/) and **Pygame**, where an intelligent agent (a big fish 🐟) learns to navigate a 5x5 underwater grid. The agent aims to reach a goal while avoiding danger zones, collecting rewards, and improving its policy through learning.

---

## 🎯 Features

- ✅ 5x5 grid world with goal state and danger zones  
- 🎮 Pygame-based real-time rendering  
- 🔁 `reset()`, `step()`, `render()` methods following OpenAI Gymnasium standards  
- 🧠 Compatible with Q-learning & DQN algorithms  
- 📊 Episode-wise training output with performance metrics  

---

## 🛠️ Environment Setup

To get started, set up a Python environment and install the required libraries:

### 1. 📦 Create a Conda Environment
```bash
conda create -n <env_name> python=3.11
````
When prompted, type y and press Enter.

### 2. ▶️ Activate the Environment
```bash
conda activate seaworld_env
````
### 3. 📚 Install Gymnasium and Optional Extras
```bash
pip install gymnasium
pip install gymnasium[other]
pip install gymnasium[classic-control]
````
### 4. 🎮 Install Pygame for Rendering
```bash
pip install pygame
````
🔗 Learn more about Gymnasium:
https://gymnasium.farama.org/

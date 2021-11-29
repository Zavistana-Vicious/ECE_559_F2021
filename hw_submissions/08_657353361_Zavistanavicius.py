#%%
import numpy as np


class Enviroment:
    def __init__(self, n, max_gold):
        self.n = n
        self.max_gold = max_gold
        self.x_pos = 0
        self.y_pos = 0
        self.gold = 0
        self.action_list = []

    def step(self, action):
        self.action_list.append(action)
        reward = 0
        x = int(self.x_pos)
        y = int(self.y_pos)
        # default movement
        if action == 0:
            x -= 1
        if action == 1:
            y -= 1
        if action == 2:
            x += 1
        if action == 3:
            y += 1

        # prevent out of bounds
        if x < 0 or x >= self.n or y < 0 or y >= self.n:
            x = int(self.x_pos)
            y = int(self.y_pos)

        # mine
        elif x == self.n - 1 and y == 0:
            if self.gold < self.max_gold:
                self.gold += 1
            x = int(self.x_pos)
            y = int(self.y_pos)

        # home
        elif x == 0 and y == self.n - 1:
            reward = self.gold
            self.gold = 0
            x = int(self.x_pos)
            y = int(self.y_pos)

        self.x_pos = x
        self.y_pos = y

        return reward

    def display(self):
        l = ["_"] * 5
        a = np.array([list(l)] * 5)
        a[self.n - 1][0] = "M"
        a[0][self.n - 1] = "H"
        a[self.x_pos][self.y_pos] = "A"

        print("Num Gold = ", str(self.gold))
        print(np.array(a))


#%%
n = 5
max_gold = 3
actions = 4
mean = 0
var = 1
q_table = np.random.normal(mean, var, (n, n, max_gold + 1, actions))


#%%
epsilon = 0.5
alpha = 0.5
gamma = 0.6

num_episodes = 1000
time = 40

# Train
for episode in range(num_episodes):
    env = Enviroment(n, max_gold)
    cumulative_reward = 0
    cr_list = []

    for t in range(1, time + 1):
        action = None
        i = int(env.x_pos)
        j = int(env.y_pos)
        k = int(env.gold)
        if np.random.rand() < epsilon:
            action = np.random.randint(0, actions)
        else:
            action = np.argmax(q_table[i][j][k])

        reward = env.step(action)
        cumulative_reward += np.power(gamma, t) * reward
        cr_list.append(cumulative_reward)
        i_ = int(env.x_pos)
        j_ = int(env.y_pos)
        k_ = int(env.gold)

        q_table[i][j][k][action] = (1 - alpha) * q_table[i][j][k][action] + alpha * (
            reward + gamma * np.max(q_table[i_][j_][k_])
        )

print(cr_list)


# %%
# Test
env = Enviroment(n, max_gold)
cumulative_reward = 0
cr_list = []
for t in range(1, time + 1):
    i = int(env.x_pos)
    j = int(env.y_pos)
    k = int(env.gold)
    action = np.argmax(q_table[i][j][k])
    reward = env.step(action)
    env.display()
    cumulative_reward += np.power(gamma, t) * reward
    cr_list.append(cumulative_reward)

print("\nCumulative Reward List")
print(cr_list)

print("\nActions List")
print(env.action_list)

# %%

# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
# importing libraries

import time
import matplotlib.pyplot as plt
#import pybullet_envs
import gym
from gym import wrappers
from torch.autograd import Variable
from collections import deque
# Creating the architecture of the Neural Network

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size= batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, \
        batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), \
               np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), \
               np.array(batch_dones).reshape(-1, 1)

# class cnn_image(nn.Module):
#     def __init__(self, input,action):
#         super(cnn_image, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, action_dim)
#
#         def forward(self, x):
#             x = x.view(-1, 1, 28, 28)
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.con2(x)), 2))
#             x = x.view(-1, 320)
#             x = F.relu(self.fc1(x))
#             print(x.shape)
#             y = self.max_action * torch.tanh(self.fc2(x))
#             print(y.shape)
#             return y

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()

        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        print("Forward")
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

class Actor(nn.Module): # define actor model with conv layers

    def __init__(self, state_dims, action_dim, max_action):
        # max_action is to clip in case we added too much noise
        super(Actor, self).__init__()  # activate the inheritance
        self.max_action = max_action
        self.conv1 = nn.Conv2d(1,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,20,kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.con2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        print(x.shape)
        y = self.max_action * torch.tanh(self.fc2(x))
        print(y.shape)
        print(type(self.max_action))
        print(self.max_action)
        return y

class Critic(nn.Module): # critic model with conv layers

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        #self.max_action = max_action
        #self.conv1 = nn.Conv2d(state_dim + action_dim, 10, kernel_size=3)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 200)
        self.fc2 = nn.Linear(200, 60)
        self.fc3 = nn.Linear(60,1)
        # Defining the second Critic neural network
        #self.max_action = max_action
        #self.conv3 = nn.Conv2d(state_dim + action_dim, 10, kernel_size=3)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4_drop = nn.Dropout2d()
        self.fc4 = nn.Linear(320, 200)
        self.fc5 = nn.Linear(200,60)
        self.fc6 = nn.Linear(200, 1)

    def forward(self, x, u):

        # Forward-Propagation on the first Critic Neural Network
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1, 28, 28)
        x1 = F.relu(F.max_pool2d(self.conv1(x1),2))
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x1 = x1.view(-1,320)
        x1 = F.relu(self.fc1(x1))
        xu1 = torch.cat([x1,u],1)
        x1 = F.relu(self.fc2(xu1))
        x1 = F.relu(self.fc3(x1))

        # Forward-Propagation on the second Critic Neural Network
        x2 = x1.view(-1,-1,28,28)
        x2 = F.relu(F.max_pool2d(self.conv1(x2), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x2)), 2))
        x2 = x1.view(-1, 300)
        x2 = F.relu(self.fc1(x2))
        xu2 = torch.cat([x2, u], 1)
        x2 = F.relu(self.fc2(xu2))
        x2 = F.relu(self.fc3(x2))
        return x1, x2

    def Q1(self, x, u):
        xq = x.view(-1, -1, 28, 28)
        xq = F.relu(F.max_pool2d(self.conv1(xq), 2))
        xq = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(xq)), 2))
        xq = xq.view(-1, 320)
        xq = F.relu(self.fc1(xq))
        xu3 = torch.cat([xq, u], 1)
        xq = F.relu(self.fc2(xu3))
        xq = F.relu(self.fc3(xq))

        return xq


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Building the whole Training Process into a class
# policy = TD3(state_dim, action_dim, max_action)
class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())


    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            state = torch.Tensor(batch_states).unsqueeze(1).to(device)
            next_state = torch.Tensor(batch_next_states).unsqueeze(1).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).reshape(batch_size,1).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action.reshape(batch_size,1))

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                print("doingjob")
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self) :# , filename, directory):
        #torch.save(self.actor.state_dict() , '%s/%s_actor.pth' % (directory, filename))
        #torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        #filename = C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small
        #directory = C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small\models
        torch.save(self.actor.state_dict(), 'C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small\models_actor.pth') #% (directory, filename))
        torch.save(self.critic.state_dict(), 'C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small\models_critic.pth')# % (directory, filename))

        #torch.save({'actor_self_dict':self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict()}, 'last_brain.pth')


    def load(self):# filename, directory):
         #filename = "C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small"
         #directory= "C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small\models"
         self.actor.load_state_dict(torch.load('C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small_actor.pth' ))#% (directory, filename)))
         self.critic.load_state_dict(torch.load('C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small_critic.pth'))# % (directory, filename)))
        #torch.save({'actor_self_dict':self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict()}, 'last_brain.pth')


    # # Implementing Deep Q Learnin
    # def update(self, reward, new_signal):
    #     new_state = torch.Tensor(new_signal).float().unsqueeze(0)
    #     self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
    #     action = self.select_action(new_state)
    #     if len(self.memory.memory) > 100:
    #         self.learn(batch_state, batch_next_state, batch_reward, batch_action)
    #         self.last_action = action
    #         self.last_state = new_state
    #         self.last_reward = reward
    #         self.reward_window.append(reward)
    #         if len(self.reward_window) > 1000:
    #             del self.reward_window[0]
    #         return action

    # def score(self):
    #     return sum(self.reward_window)/(len(self.reward_window)+1.)
    #     #torch.save({'state_dict': self.model.state_dict(),
    #      #               'optimizer' : self.optimizer.state_dict(),
    #       #             }, 'last_brain.pth')
    #     torch.save({'actor_self_dict':self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict()}, 'last_brain.pth')
    #
    # #### Training




    # def load(self):
    #     if os.path.isfile('last_brain.pth'):
    #         print("=> loading checkpoint... ")
    #         checkpoint = torch.load('last_brain.pth')
    #         self.model.load_state_dict(checkpoint['state_dict'])
    #         self.optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("done !")
    #     else:
    #         print("no checkpoint found...")
    # Making a load method to load a pre-trained model
    # def load(self, filename, directory):
    #     #self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    #     #self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
    #     torch.save({'actor_self_dict': self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict()},
    #                'last_brain.pth')
    #def update(self, reward, new_signal):
#         new_state = torch.Tensor(new_signal).float().unsqueeze(0)
#         self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
#         action = self.select_action(new_state)
#         if len(self.memory.memory) > 100:
#             batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
#             self.learn(batch_state, batch_next_state, batch_reward, batch_action)
#         self.last_action = action
#         self.last_state = new_state
#         self.last_reward = reward
#         self.reward_window.append(reward)
#         if len(self.reward_window) > 1000:
#             del self.reward_window[0]
#         return action
#
#     def score(self):
#         return sum(self.reward_window)/(len(self.reward_window)+1.)
#
#
#
#     def load(self):
#         if os.path.isfile('last_brain.pth'):
#             print("=> loading checkpoint... ")
#             checkpoint = torch.load('last_brain.pth')
#             self.model.load_state_dict(checkpoint['state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#             print("done !")
#         else:
#             print("no checkpoint found...")




# evaluate policy
# def evaluate_policy(policy, eval_episodes=10):
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             action = policy.select_action(np.array(obs))
#             obs, reward, done, _ = env.step(action)
#             avg_reward += reward
#     avg_reward /= eval_episodes
#     print("---------------------------------------")
#     print("Average Reward over the Evaluation Step: %f" % (avg_reward))
#     print("---------------------------------------")
#     return avg_reward




###################################

# class Network(nn.Module):
#
#     def __init__(self, input_size, nb_action):
#         super(Network, self).__init__()
#         self.input_size = input_size
#         self.nb_action = nb_action
#         self.fc1 = nn.Linear(input_size, 30)
#         self.fc2 = nn.Linear(30, nb_action)
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         q_values = self.fc2(x)
#         return q_values

# Implementing Experience Replay

# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#
#     def push(self, event):
#         self.memory.append(event)
#         if len(self.memory) > self.capacity:
#             del self.memory[0]
#
#     def sample(self, batch_size):
#         samples = zip(*random.sample(self.memory, batch_size))
#         return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning


# class Dqn():
#
#     def __init__(self, input_size, nb_action, gamma):
#         self.gamma = gamma
#         self.reward_window = []
#         self.model = Network(input_size, nb_action)
#         self.memory = ReplayMemory(100000)
#         self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
#         self.last_state = torch.Tensor(input_size).unsqueeze(0)
#         self.last_action = 0
#         self.last_reward = 0
#
#     def select_action(self, state):
#         probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
#         action = probs.multinomial()
#         return action.data[0,0]
#
#     def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
#         outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
#         next_outputs = self.model(batch_next_state).detach().max(1)[0]
#         target = self.gamma*next_outputs + batch_reward
#         td_loss = F.smooth_l1_loss(outputs, target)
#         self.optimizer.zero_grad()
#         td_loss.backward(retain_variables = True)
#         self.optimizer.step()
#
#     def update(self, reward, new_signal):
#         new_state = torch.Tensor(new_signal).float().unsqueeze(0)
#         self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
#         action = self.select_action(new_state)
#         if len(self.memory.memory) > 100:
#             batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
#             self.learn(batch_state, batch_next_state, batch_reward, batch_action)
#         self.last_action = action
#         self.last_state = new_state
#         self.last_reward = reward
#         self.reward_window.append(reward)
#         if len(self.reward_window) > 1000:
#             del self.reward_window[0]
#         return action
#
#     def score(self):
#         return sum(self.reward_window)/(len(self.reward_window)+1.)
#
#     def save(self):
#         torch.save({'state_dict': self.model.state_dict(),
#                     'optimizer' : self.optimizer.state_dict(),
#                    }, 'last_brain.pth')
#
#     def load(self):
#         if os.path.isfile('last_brain.pth'):
#             print("=> loading checkpoint... ")
#             checkpoint = torch.load('last_brain.pth')
#             self.model.load_state_dict(checkpoint['state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#             print("done !")
#         else:
#             print("no checkpoint found...")


#Td3 alg
# actor
# class Actor(nn.Module):
#
#     def __init__(self, state_dims, action_dim, max_action):
#         # max_action is to clip in case we added too much noise
#         super(Actor, self).__init__()  # activate the inheritance
#         self.layer_1 = nn.Linear(state_dims, 400)
#         self.layer_2 = nn.Linear(400, 300)
#         self.layer_3 = nn.Linear(300, action_dim)
#         self.max_action = max_action
#
#     def forward(self, x):
#         x = F.relu(self.layer_1(x))
#         x = F.relu(self.layer_2(x))
#         x = self.max_action * torch.tanh(self.layer_3(x))
#         return x

# class Critic(nn.Module): # critic model with conv layers
#
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         # Defining the first Critic neural network
#         self.max_action = max_action
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 3)
#
#
#         self.layer_1 = nn.Linear(state_dim + action_dim, 400)
#         self.layer_2 = nn.Linear(400, 300)
#         self.layer_3 = nn.Linear(300, 1)
#         # Defining the second Critic neural network
#         self.layer_4 = nn.Linear(state_dim + action_dim, 400)
#         self.layer_5 = nn.Linear(400, 300)
#         self.layer_6 = nn.Linear(300, 1)
#
#     def forward(self, x, u):
#         xu = torch.cat([x, u], 1)
#         # Forward-Propagation on the first Critic Neural Network
#         x1 = F.relu(self.layer_1(xu))
#         x1 = F.relu(self.layer_2(x1))
#         x1 = self.layer_3(x1)
#         # Forward-Propagation on the second Critic Neural Network
#         x2 = F.relu(self.layer_4(xu))
#         x2 = F.relu(self.layer_5(x2))
#         x2 = self.layer_6(x2)
#         return x1, x2
#
#     def Q1(self, x, u):
#         xu = torch.cat([x, u], 1)
#         x1 = F.relu(self.layer_1(xu))
#         x1 = F.relu(self.layer_2(x1))
#         x1 = self.layer_3(x1)
#         return x1

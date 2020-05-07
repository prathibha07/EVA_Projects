# Self Driving Car

# Importing the libraries
import os
import numpy as np
from random import random, randint
import random as rand
from numpy import *
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import torch
import sys
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import pyscreenshot as Imageshot
from numpy import asarray
from PIL import Image as PILImage

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen , ScreenManager

# Importing the Dqn object from our AI in ai.py
#from ai import Dqn
from ai import TD3, Actor,Critic, ReplayBuffer


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
seed = 0 # Random seed number
torch.manual_seed(seed)
np.random.seed(seed)
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
episode_reward = 0
maxepisode_timesteps = 500

torch.manual_seed(seed)
np.random.seed(seed)

state_dim = 5 # position, velocity# ,orientation,
action_dim = 1 #moving
max_action = 5
min_action = -5
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_timesteps = 0
done = True
t0 = time.time()

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
#brain = Dqn(5,3,0.9) # CHANGE
#brain = TD3(6,3,5)  # states, action, max_Action
action2rotation = [0, 5,-5] #action2rotation = [0,5,-5] #angle of rotation
brain = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# Initializing the map , keep i as 0
first_update = True
i = 0

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global img
    sand = np.zeros((longueur,largeur))
    # convert to L - take the lower value only, takes one channel
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255 # values btw 0&1
    goal_x = 360#1420
    goal_y = 315 #622
    first_update = False
    global swap
    swap = 0
    #max_timesteps= 1000 #train on colab for higher
    #first_update = False # move between 2 points


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget): # kivy initialization
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)


    def move(self, rotation): # move car and sensors
        print("car is moving")
        self.pos = Vector(*self.velocity) + self.pos # storing car position, updating car position after every move on the map
        print(rotation,type(rotation))
        self.rotation = rotation
        self.angle = self.angle + self.rotation


class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
       # print("serve_car")
       # To Randomly Initialize after every episode
        #xint = np.random.randint(0,self.width)
        #yint = np,random.randint(0,self.height)
       #print(xint,yint)
       # self.car.center = (xint,yint)
        self.car.center = self.center  # center of car
        self.car.velocity = Vector(6, 0)  # highest speed 6

    def calc_obs(self, xx, yy):
        crop_aroundArea = (self.car.x - 100, self.car.y - 100, self.car.x + 100, self.car.y + 100)
        new_obj = img.rotate(90, expand=True).crop(crop_aroundArea)
        #obj.save("xxcropdone" + str(i) + ".jpg")
        car_img  = PILImage.open("./images/mask.png").convert('L')  # Opens a image in RGB mode
        new_obj.paste(car_img.rotate(Vector(*self.car.velocity).angle((xx, yy)), expand=True), (90, 95))
        new_obj.thumbnail((28, 28))
        return new_obj

    def step(self, action, last_distance):
        print("Step")
        global goal_x
        global goal_y
        global done
        self.car.move(action)
        xx = goal_x - self.car.x  #goal - car location
        yy = goal_y - self.car.y
        done = False
        obsv = self.calc_obs(xx, yy)
        print(last_distance)
        # using 2 point distance formula
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        print("distance" + str(distance))

        # if value of car that particular pixel in sand array > 0 i.e means its on sand
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)  # reducethe vel
            # print log
            print(1, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
                  im.read_pixel(int(self.car.x), int(self.car.y)))
            # penalize for moving on sand
            reward = -1
        else:  # otherwise on the road
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)# increase vel slightly
            # living penality
            reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
                  im.read_pixel(int(self.car.x), int(self.car.y)))
            # new distance in the right direction, then give reward positive value
            if distance < last_distance:
                reward = 0.2 # reward for going towards goal
            # else:
            #     last_reward = last_reward +(-0.2)

        # not going near wall area
        # Adding done condition and negative reward for going near borders
        if self.car.x < 5:  # 5 pixels from the wall
            self.car.x = 5
            reward = -1
            done = True
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            reward = -1
            done = True # make done true as bad episode

        if self.car.y < 5:
            self.car.y = 5
            reward = -1
            done = True
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            reward = -1
            done = True
        # if within 25 pixels of the destination, then give reward
        if distance < 25:
            reward =25
            done =True # end if go near destination


        return obsv, reward, done, distance



    def update(self, dt):

        global brain
        #global last_reward
        global reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
       #global swap
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global done
        global seed  # Random seed number
        global start_timesteps  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
        global eval_freq  # How often the evaluation step is performed (after how many timesteps)
        global max_timesteps  # Total number of iterations/timesteps
        global save_models  # Boolean checker whether or not to save the pre-trained model
        global expl_noise  # Exploration noise - STD value of exploration Gaussian noise
        global batch_size  # Size of the batch
        global discount  # Discount factor gamma, used in the calculation of the total discounted reward
        global tau  # Target network update rate
        global policy_noise  # STD of Gaussian noise added to the actions for the exploration purposes
        global noise_clip  # Maximum value of the Gaussian noise added to the actions (policy)
        global policy_freq
        global max_action
        global episode_reward
        global episode_timesteps
        #global maxepisode_timesteps

        longueur = self.width
        print("firstUpdate")
        largeur = self.height

        #xx = goal_x - self.car.x  # goal - car location
        #yy = goal_y - self.car.y
        #observationspace = self.calc_obs(xx, yy)

        #max_timesteps = 10000
        maxepisode_timesteps = 500
        # We start the main loop over 500,000 timesteps
        # while total_timesteps < max_timesteps:
        #  print("total_timesteps:" + str(total_timesteps))
        if first_update:
            init()
            print("NewUpdate")
            if len(filename) >0:
                self.serve_car()
        observationspace = self.calc_obs(self.car.x, self.car.y)
        if len(filename) > 0:
            action = brain.select_action(np.array(observationspace))
            print(action, last_distance)
            observationspace, reward, done, distance = self.step(float(action), last_distance)
                # print(distance,done)
        else:
            print("trainmode")
            print("total_timesteps:" + str(total_timesteps))
            if done:
                print("entering done loop")
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,episode_reward))
                    brain.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                                policy_freq)
                    if save_models and not os.path.exists("./pytorch_models"):
                        os.makedirs("./pytorch_models")
                    brain.save("TD3Model" + str(episode_num), directory="./pytorch_models")

                # When the training step is done, we reset the state of the environment
                # obs = env.reset()
                self.serve_car()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if total_timesteps < start_timesteps:
                    # action = env.action_space.sample()
                print("random action")
                action = rand.randrange(-5, 5) * random.random()
                    # self.car.move(action)
            else:  # After 10000 timesteps, we switch to the model
                print("model action")
                action = brain.select_action(np.array(observationspace))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(min_action, max_action)
                # self.car.move(action)
            new_obs, reward, done, distance = self.step(float(action),last_distance)

        # We check if the episode is done
        # done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

        # We increase the total reward
            episode_reward += reward

        # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((observationspace, new_obs, action, reward, done))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            observationspace = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            #last_distance = distance

            if episode_timesteps == maxepisode_timesteps:
                    done = True
        last_distance = distance

# INFERENCE
class CarApp(App):

    def build(self):
        print("build")
        parent = Game()
        if len(filename) > 0:
            brain.load(filename, './pytorch_models/')
            print("Inference Mode")
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        # parent.update()
        return parent

# Running the whole thing
if __name__ == '__main__':
    global filename
    filename = ""
    #To check if to run in train mode or evaluation mode by passing a stored model
    if len(sys.argv) > 1:
        #print(sys.argv[1])
        filename = sys.argv[1]
        CarApp().run()
    else:
        print("200")
        CarApp().run()



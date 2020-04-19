# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
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

state_dim = 6 # position, velocity# ,orientation,
action_dim = 1
max_action = 5
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
im = CoreImage("C:/Users/prathibha7/PycharmProjects/eva/p2s10/Session7Small/Session7Small/images/MASK1.png")

# state_dim = 5
# action_dim =3
# max_action =5
# policy = TD3(state_dim, action_dim, max_action)
#replay_buffer = ReplayBuffer()
# evaluations = [evaluate_policy(policy)]=
# need to call actor , critic, replay memore,td3

# textureMask = CoreImage(source="./kivytest/simplemask1.png")

#policy = TD3(state_dim, action_dim, max_action)

#evaluations = [evaluate_policy(policy)]

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
    img = PILImage.open("C:/Users/prathibha7/PycharmProjects/eva/p2s10/Session7Small/Session7Small/images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    global swap
    swap = 0
    max_timesteps= 1000 #train on colab for higher
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
    # position_x = NumericProperty(0)
    # position_y = NumericProperty(0)
    # position = ReferenceListProperty(position_x,position_y)
    # orientation_x = NumericProperty(0)
    # orientation_y = NumericProperty(0)
    # orientation = ReferenceListProperty(orientation_x, orientation_y)
    # sensor1_x = NumericProperty(0)
    # sensor1_y = NumericProperty(0)
    # sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    # sensor2_x = NumericProperty(0)
    # sensor2_y = NumericProperty(0)
    # sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    # sensor3_x = NumericProperty(0)
    # sensor3_y = NumericProperty(0)
    # sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    # signal1 = NumericProperty(0)
    # signal2 = NumericProperty(0)
    # signal3 = NumericProperty(0)

    def move(self, rotation): # move car and sensors
        print("car is moving")
        self.pos = Vector(*self.velocity) + self.pos # storing car position, updating car position after every move on the map
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        #self.orientation = orientation ??
        # movement is based on the cropped imagethe car sees, need to use sand densities to enable car to differentiate road
        # image is sent to CNN ,CNN tensor output to td3
        # use PIl crop

        # we know car position, crop 50*50 around the center
        # feed it to cnn
        #cnn should give out sand densities
        # go till 20 pixels before corner or use padding
        # need to use sand densities at all 3 points around car center
        # need to find these points wrt car orientation
        # if sand[int(self.car.x), int(self.car.y)] > 0:
        #     self.car.velocity = Vector(0.5, 0).rotate(self.car.angle) # reduce velocity
        #     print(1, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
        #           im.read_pixel(int(self.car.x), int(self.car.y)))
        #
        #     last_reward = -1
        # else:  # otherwise
        #     self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        #     last_reward = -0.2
        #     print(0, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
        #           im.read_pixel(int(self.car.x), int(self.car.y)))
        #     if distance < last_distance:
        #         last_reward = 0.1
    # crop image use PIL crop
    # cropPic = PILimage.crop(self.car_x,self.car_y)


# self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.



        # def __init(self, input_data, nb_action):
        #     super(Network, self).__init__()
        #     self.input_size = input_size
        #     # self.hidden= 10 # adding a layer
        #     self.nb_action = nb_action
        #     self.fc1 = nn.Linear(input_size, 5)
        #     self.fc2 = nn.Linear(5, 30)  # m1=1x5, m2= 30x30
        #     self.fc3 = nn.Linear(30, nb_action)  # 30,action
        #     # self.fc2= nn.Linear() #










        # +10-10 around image
        #= int(np.sum(sand[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10, //
                         #         int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        # if all pixels are white/400 = 1 - on sand
        # if part of car on  sand then only part of pixels are white /400 =
        # above gives density of area car is in
        # need one more sensor variable which reads sand values through cnn


        # self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        # self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        # self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        # self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        # self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        # self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        # if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
        #     self.signal1 = 10. # edges , sensor values are big
        # if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
        #     self.signal2 = 10.
        # if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
        #     self.signal3 = 10.
        

# class Ball1(Widget):
#     pass
# class Ball2(Widget):
#     pass
# class Ball3(Widget):
#     pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    # ball1 = ObjectProperty(None)
    # ball2 = ObjectProperty(None)
    # ball3 = ObjectProperty(None)
    #
    def serve_car(self):
       # print("serve_car")
        self.car.center = self.center  # center of car
        self.car.velocity = Vector(6, 0)  # highest speed 6

    def calc_obs(self, xx, yy):
        crop_around = (self.car.x - 100, self.car.y - 100, self.car.x + 100, self.car.y + 100)
        new_obj = img.rotate(90, expand=True).crop(crop_around)
        #obj.save("xxcropdone" + str(i) + ".jpg")
        car_img = PILImage.open(
                "C:/Users/prathibha7/PycharmProjects/eva/p2s10/Session7Small/Session7Small/images/car.png").resize(
                (20, 10))  # Opens a image in RGB mode
        new_obj.paste(car_img.rotate(Vector(*self.car.velocity).angle((xx, yy)), expand=True), (90, 95))
        new_obj.thumbnail((28, 28))
        return new_obj

    def step(self, action, last_distance):
        print("Step")
        global goal_x
        global goal_y
        global done
        self.car.move(action)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        done = False
        obsv = self.calc_obs(xx, yy)
        # using 2 point distance formula
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        # if value of car that particular pixel in sand array > 0 i.e means its on sand
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)  # reducethe vel
            # print log
            print(1, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
                  im.read_pixel(int(self.car.x), int(self.car.y)))
            # penalize
            reward = -1
        else:  # otherwise on the road
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)# increase vel
            # living penality
            reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x), int(self.car.y),
                  im.read_pixel(int(self.car.x), int(self.car.y)))
            # new distance in the right direction, then give reward positive value
            if distance < last_distance:
                reward = 0.1
            # else:
            #     last_reward = last_reward +(-0.2)

        # not going near wall area
        if self.car.x < 5:  # 5 pixels from the wall
            self.car.x = 5
            reward = -1
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
        # if within 25 pixels of the destination, then swap
        if distance < 25:
            done =True # end if go near destination


        return obsv, reward, done, distance
        #     if swap == 1:
        #         goal_x = 1420
        #         goal_y = 622
        #         swap = 0
        #     else:
        #         goal_x = 9
        #         goal_y = 85
        #         swap = 1
        # last_distance = distance



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
        # self.ball1.pos = self.car.sensor1
        # self.ball2.pos = self.car.sensor2
        # self.ball3.pos = self.car.sensor3
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

        longueur = self.width
        print("firsUpdate")
        largeur = self.height
        if first_update:
            init()
            print("NEW update")
            xx = goal_x - self.car.x #goal - car location
            yy = goal_y - self.car.y
            observationspace = self.calc_obs(xx, yy)

        max_timesteps = 200
        # We start the main loop over 500,000 timesteps
        # while total_timesteps < max_timesteps:
        print("total_timesteps:" + str(total_timesteps))
        if done:
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                              episode_reward))
                brain.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                             policy_freq)
            # When the training step is done, we reset the state of the environment
            # obs = env.reset()
            Game().serve_car()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        if total_timesteps < start_timesteps:
            # action = env.action_space.sample()
            action = random.randrange(-5, 5) * random.random()

            # self.car.move(action)
        else:  # After 10000 timesteps, we switch to the model
            action = brain.select_action(np.array(observationspace))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(min_action, max_action)
                # self.car.move(action)
        new_obs, reward, done, distance = self.step(action, last_distance)

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
        last_distance = distance
        # action = brain.update(last_reward, observationspace)
        # scores.append(brain.score())
        # rotation = action2rotation[action]
        # self.car.move(rotation)



        #orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        #last_signal=[position, velocity, sensor1, orientation, -orientation] # include sensor strength
        # last signal defines the 5 states input to dnn , change is state def chanages
        # last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        #action = brain.update(last_reward, last_signal)
        #scores.append(brain.score())
        #rotation = action2rotation[action]
        #action = brain.update(last_reward,last_signal) # update brain with experience,states and rewards
        # action has sensor values + rewards
        #scores.append(brain.score()) # logging
        #rotation = action2rotation(action) # what type of action is taken
        #self.car.move(rotation)
        #distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        # next is updating posiition of ball, not needed here









# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("C:/Users/prathibha7/PycharmProjects/eva/p2s10/Session7Small/Session7Small/images/sand3.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        print("creating")
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        #brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
       # brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
#######



# We add the last policy evaluation to our list of evaluations and we save our model
# evaluations.append(evaluate_policy(policy))
# if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
# np.save("C:\Users\prathibha7\PycharmProjects\eva\p2s10\Session7Small\Session7Small\results\%s" %  evaluations)

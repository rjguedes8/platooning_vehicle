import glob
import os
import sys
import numpy as np
import cv2
import random
import time
from throttle.ppo import PPO
from steer.agent import Agent
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import math
from cnn import xception
import tensorflow as tf

EPISODES = 1000
SECONDS_PER_EPISODE = 60

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla 

class CarEnv:
    SHOW_CAM = True
    im_width = 640
    im_height = 480
    front_camera = None
    SECONDS_PER_EPISODE = 60

    def __init__(self, model):
        self.client = carla.Client('localhost', 2000) #connect to carla
        self.client.set_timeout(10.0) #2 seg to connect to carla
        self.world = self.client.load_world('Town01') #get enviroment
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0] #tesla model 3
        self.action = 1
        self.tm = self.client.get_trafficmanager(8000)
        self.tm_port = self.tm.get_port()
        self.img_names = []
        self.model = model

    def reset(self, point_position):
        self.collision_hist = []
        self.actor_list = []

        if point_position == 0: 
            self.point_position = random.choice(self.world.get_map().get_spawn_points())
        else:
            self.point_position = point_position

        # auto pilot car
        self.transform_auto = self.point_position
        self.vehicle_auto = self.world.spawn_actor(self.model_3, self.transform_auto)
        self.actor_list.append(self.vehicle_auto)

        # our agent
        self.point_position.location.x -= 8
        self.transform = self.point_position
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.sem_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.sem_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sem_camsensor = self.world.spawn_actor(self.sem_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sem_camsensor)
        self.sem_camsensor.listen(lambda data: self.process_sem(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer = 0))
        time.sleep(4)

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.sem_cam is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, brake=0.0))

        return self.sem_cam

    def collision_data(self, event):
        self.collision_hist.append(event)
    
    def process_img(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height,self.im_width, 4))
        i = i[:, :, :3]
        self.img_rgb = i

    def process_sem(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height,self.im_width, 4))
        i = i[:, :, :3]
        self.sem_cam= i[:,:,-1] 
    
    def step(self, action_steer, action_throttle, path, img_prb, distance, kmh):
        previous_distance = distance
        if img_prb >= 0.8:
            cv2.imwrite(path+'/'+str(datetime.now())+'.png', self.img_rgb)
        if self.vehicle.is_alive: 
            if action_throttle <= 0:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=float(action_steer), brake=float(action_throttle*-1)))
            else:
                self.vehicle.apply_control(carla.VehicleControl(throttle=float(action_throttle), steer=float(action_steer)))

            distance, state = self.get_current_state(self.sem_cam, previous_distance)
            reward_steer = 0
            reward_throttle = 0
        
            if len(self.collision_hist) != 0:
                done = True
                reward_throttle += -10
                reward_steer += -10

            elif sum(state[:3]) == 0:
                done = True
                reward_steer += -10

            elif distance > 25:
                done = True
                reward_throttle += -10

            else:
                #distance 
                if (distance < previous_distance) & (distance > 10):
                    reward_throttle += 5
                elif (distance >= previous_distance) & (distance < 8):
                    reward_throttle += 5
                
                if (int(kmh) == 0) & (float(action_throttle) <= 0.0) & (distance >= 13):
                    reward_throttle -= 5
                
                if (distance >= 8) & (distance <= 10):
                    reward_throttle += 5
                else:
                    reward_throttle -= 5

                #alignement
                if state[0] == 1:
                    reward_steer += 5
                else:
                    reward_steer -= 5
                
                if (state[1] == 1) & (action_steer > 0):
                    reward_steer -= 5
                elif (state[2] == 1) & (action_steer < 0):
                    reward_steer -= 5
                elif (state[1] == 1) & (action_steer < 0):
                    reward_steer += 5
                elif (state[2] == 1) & (action_steer > 0):
                    reward_steer += 5
                elif (state[0] == 1) & (action_steer != 0.0):
                    reward_steer -= 5
                elif (state[0] == 1) & (action_steer == 0.0):
                    reward_steer += 5
                
                done = False
                
            if self.episode_start + SECONDS_PER_EPISODE < time.time():
                done = True

        else:
            reward_steer = -10
            reward_throttle = -10
            done = True
            distance, state = self.get_current_state(self.sem_cam, previous_distance)
        
        return distance, state, reward_steer, reward_throttle, done, None
    
    def get_current_state(self, sem_camera, previous_distance):
        
        leader = (sem_camera==10)*1

        if sum(sum(leader)) > 0:
            pixel_leader = int((min(np.where(leader)[1])+max(np.where(leader)[1]))/2)
            aligned_with_leader = (300 <= pixel_leader) & (340 >= pixel_leader)*1
            leader_left = (300 > pixel_leader)*1
            leader_right = (340 < pixel_leader)*1
            tensor = tf.image.resize(cv2.cvtColor(leader.astype('float32'), cv2.COLOR_GRAY2RGB), [255, 255])
            cnn_dist = self.model.predict(tf.expand_dims(tensor, axis=0))[0][0]

        else:
            aligned_with_leader = 0
            leader_left = 0
            leader_right = 0
            cnn_dist=99
        
        return cnn_dist, [aligned_with_leader, leader_left, leader_right, cnn_dist/25, previous_distance/25]


# Throttle/Break agent
K_epochs = 10             # update policy for K epochs in one PPO update
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
action_std = 0.55  # 0.55 starting std for action distribution (Multivariate Normal)

agent_throttle = PPO(3, 1, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space=True, action_std_init=action_std) 
agent_throttle.load("models/throttle/throttle.pth")

#Steer Agent
agent_steer = Agent(n_actions=3, batch_size=32, n_epochs=10)
agent_steer.load_models()

# Xception model
tf.config.list_physical_devices('GPU')
model = xception.create_xception_model()

random.seed(1)
np.random.seed(1)

for episode in tqdm(range(0, EPISODES + 1), unit='episodes'):

    env = CarEnv(model)
    env.collision_hist = []
    img_prb = np.random.uniform(0,1,1)

    if img_prb >= 0.8:
        try:
            os.makedirs(os.getcwd()+f'/episodes/{episode}')
        except:
            pass
    path = os.getcwd()+f'/episodes/{episode}'
    episode_reward_steer = 0
    episode_reward_throttle = 0
    step = 1
    cannot_spawned = True
    while cannot_spawned:
        try:
            if episode == 0:
                sem_camera = env.reset(0)
                point_position = env.point_position
            else:
                sem_camera = env.reset(point_position)
            lead_dist, current_state = env.get_current_state(sem_camera, 6)
            cannot_spawned = False
        except:
            env = CarEnv(model)
            cannot_spawned = True

    while sum(current_state[:3]) == 0:
        env = CarEnv(model)
        cannot_spawned = True
        while cannot_spawned:
            try:
                if episode == 0:
                    sem_camera = env.reset(0)
                    point_position = env.point_position
                else:
                    sem_camera = env.reset(point_position)
                lead_dist, current_state = env.get_current_state(sem_camera, 6)
                cannot_spawned = False
            except:
                env = CarEnv(model)
                cannot_spawned = True

    episode_start = time.time()
    env.vehicle_auto.set_autopilot(True, env.tm_port)
    env.tm.vehicle_percentage_speed_difference(env.vehicle_auto, 50)

    time.sleep(2.0)

    env.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
    v = env.vehicle.get_velocity()
    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
    print(kmh)

    episode_start = time.time()

    while True:

        v = env.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        current_state.append(kmh/100)

        action_throttle = agent_throttle.select_action(current_state[3:])
        
        if action_throttle[0] > 1:
            action_throttle[0] = 1
        elif action_throttle[0] < -1:
            action_throttle[0] = -1

        steer_state = current_state[1:3]
        action, prob, val = agent_steer.choose_action(np.array([steer_state]).astype(np.float32))

        if action == 0:
            action_steer = -0.25
        elif action == 1:
            action_steer = 0.0
        else:
            action_steer = 0.25

        lead_dist, new_state, reward_steer, reward_throttle, done, _ = env.step(action_steer, action_throttle[0], path, img_prb, lead_dist, kmh)
        
        episode_reward_steer += reward_steer
        episode_reward_throttle += reward_throttle

        # saving reward and is_terminals
        agent_steer.store_transition(np.array([steer_state]).astype(np.float32), action, prob, val, reward_steer, done)

        agent_throttle.buffer.rewards.append(reward_throttle)
        agent_throttle.buffer.is_terminals.append(done)

        current_state = new_state
        step+=1
        time.sleep(0.05)

        if done:
            break

    agent_steer.learn()
    agent_steer.save_models()
    agent_throttle.update()
    agent_throttle.save("models/throttle/throttle.pth")

    f = open(os.getcwd()+'/stats.txt', 'a')
    f.write(f'Episode {episode}, Reward Steer {episode_reward_steer}, Reward Throttle {episode_reward_throttle}, Actions {step}\n')
    f.close()
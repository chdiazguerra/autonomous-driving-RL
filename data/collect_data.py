import os
import math
import random
from argparse import ArgumentParser
import pickle

import numpy as np
import cv2
import carla

from data.utils import to_rgb_array, depth_to_array, labels_to_array, distance_from_center, depth_to_logarithmic_grayscale, dist_to_roadline
from configuration.config import *

class Observation:
    def __init__(self, save_path, save_np=False):
        self.rgb = None
        self.depth = None
        self.semantic = None
        self.obstacle_dist = (0, 25) #(Frame detected, distance)
        self.save_path = save_path
        self.save_np = save_np
    def save_data(self, frame):
        if not self.save_np:
            cv2.imwrite(self.save_path + '/camera/%08d.png' % frame, cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.save_path + '/depth/%08d.png' % frame, self.depth)
            cv2.imwrite(self.save_path + '/semantic/%08d.png' % frame, self.semantic)
        else:
            np.save(self.save_path + '/camera/%08d' % frame, self.rgb)
            np.save(self.save_path + '/depth/%08d' % frame, self.depth)
            np.save(self.save_path + '/semantic/%08d' % frame, self.semantic)

def camera_callback(img, obs):
    if img.raw_data is not None:
        array = to_rgb_array(img)
        obs.rgb = array

def depth_callback(img, obs):
    if img.raw_data is not None:
        array = depth_to_logarithmic_grayscale(depth_to_array(img))
        obs.depth = array

def semantic_callback(img, obs):
    if img.raw_data is not None:
        array = labels_to_array(img)
        obs.semantic = array

def obstacle_callback(event, obs):
    frame = event.frame
    if 'vehicle' in event.other_actor.type_id:
        obs.obstacle_dist = (frame, event.distance)
    else:
        obs.obstacle_dist = (frame, 25.)

def get_spectator_transform(vehicle_transform, d=7):
    vehicle_transform.location.z += 3
    vehicle_transform.location.x += -d*math.cos(math.radians(vehicle_transform.rotation.yaw))
    vehicle_transform.location.y += -d*math.sin(math.radians(vehicle_transform.rotation.yaw))
    vehicle_transform.rotation.pitch += -20
    return vehicle_transform

def setup_sensors(ego_vehicle, blueprint_library, obs, world):
    sensors = []
    #Create sensors
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAM_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAM_HEIGHT))
    camera_bp.set_attribute('fov', str(CAM_FOV))
    camera_bp.set_attribute('sensor_tick', str(TICK))
    camera_transform = carla.Transform(carla.Location(x=CAM_POS_X, y=CAM_POS_Y , z=CAM_POS_Z),
                                      carla.Rotation(pitch=CAM_PITCH, yaw=CAM_YAW, roll=CAM_ROLL))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    camera.listen(lambda data: camera_callback(data, obs))
    sensors.append(camera)

    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(CAM_WIDTH))
    depth_bp.set_attribute('image_size_y', str(CAM_HEIGHT))
    depth_bp.set_attribute('fov', str(CAM_FOV))
    depth_bp.set_attribute('sensor_tick', str(TICK))
    depth_transform = carla.Transform(carla.Location(x=CAM_POS_X, y=CAM_POS_Y , z=CAM_POS_Z),
                                      carla.Rotation(pitch=CAM_PITCH, yaw=CAM_YAW, roll=CAM_ROLL))
    depth = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle)
    depth.listen(lambda data: depth_callback(data, obs))
    sensors.append(depth)

    semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    semantic_bp.set_attribute('image_size_x', str(CAM_WIDTH))
    semantic_bp.set_attribute('image_size_y', str(CAM_HEIGHT))
    semantic_bp.set_attribute('fov', str(CAM_FOV))
    semantic_bp.set_attribute('sensor_tick', str(TICK))
    semantic_transform = carla.Transform(carla.Location(x=CAM_POS_X, y=CAM_POS_Y , z=CAM_POS_Z),
                                      carla.Rotation(pitch=CAM_PITCH, yaw=CAM_YAW, roll=CAM_ROLL))
    semantic = world.spawn_actor(semantic_bp, semantic_transform, attach_to=ego_vehicle)
    semantic.listen(lambda data: semantic_callback(data, obs))
    sensors.append(semantic)

    obstacle_bp = blueprint_library.find('sensor.other.obstacle')
    obstacle_bp.set_attribute('only_dynamics', 'False')
    obstacle_bp.set_attribute('distance', '20')
    obstacle_bp.set_attribute('sensor_tick', str(TICK))
    obstacle_transform = carla.Transform()
    obstacle = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=ego_vehicle)
    obstacle.listen(lambda data: obstacle_callback(data, obs))
    sensors.append(obstacle)
    return sensors

def main(args):
    #Create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    camera_path = os.path.join(args.out_folder, 'camera')
    depth_path = os.path.join(args.out_folder, 'depth')
    semantic_path = os.path.join(args.out_folder, 'semantic')
    os.makedirs(camera_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(semantic_path, exist_ok=True)
    

    weather = getattr(carla.WeatherParameters, args.weather, carla.WeatherParameters.ClearNoon)
    tfm_port = args.world_port + 1

    sensors = []
    vehicles = []
    ego_vehicle = None
    original_settings = None
    if os.path.exists(f"{args.out_folder}/info.pkl"):
        with open(f"{args.out_folder}/info.pkl", 'rb') as f:
            info = pickle.load(f)
    else:
        info = {}
    try:
        #Connect client to server
        client = carla.Client(args.host, args.world_port)
        client.set_timeout(60.0)

        #Load world
        world = client.load_world(args.map)
        original_settings = world.get_settings()
        world.set_weather(weather)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TICK
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        carla_map = world.get_map()
        spectator = world.get_spectator()

        #Set traffic manager
        traffic_manager = client.get_trafficmanager(tfm_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(False)

        #Vehicle blueprint
        vehicle_bp = blueprint_library.find(EGO_BP)

        #Routes
        routes = [73, 14, 61, 241, 1, 167, 71, 233, 176, 39]
        spawns = [[147, 190, 146, 192, 240, 143, 195, 196, 197, 17, 14, 110, 111, 117, 201, 203, 115, 13, 58],
                  [11, 115, 95, 13, 10, 96, 58, 60, 114, 61, 232, 238, 8],
                  [114, 113, 235, 230, 238, 112, 201, 207, 102, 204, 117],
                  [88, 226, 228, 229, 231, 102, 239, 101, 114, 238, 76, 104],
                  [75, 105, 99, 106, 77, 200, 168, 107, 3, 139, 167, 18],
                  [92, 105, 200, 222, 77, 106, 1, 134, 99, 107],
                  [221, 220, 223, 68, 97, 224, 119],
                  [82, 78, 42, 40, 38, 48, 126, 174, 152, 90],
                  [152, 163, 63, 130, 234, 236, 65],
                  [79, 37, 90, 36, 34, 29, 31]]

        for episode in range(args.begin, args.nb_passes*len(routes)):
            print("Episode %d" % episode)
            route_id = episode % len(routes)
            ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[routes[route_id]])
            traffic_manager.set_desired_speed(ego_vehicle, 35.)
            traffic_manager.ignore_lights_percentage(ego_vehicle, 100.)
            traffic_manager.random_left_lanechange_percentage(ego_vehicle, 0)
            traffic_manager.random_right_lanechange_percentage(ego_vehicle, 0)
            traffic_manager.auto_lane_change(ego_vehicle, False)
            traffic_manager.distance_to_leading_vehicle(ego_vehicle, 1.0)
            ego_vehicle.set_autopilot(True, tfm_port)

            for spawn_id in spawns[route_id]:
                if np.random.uniform(0., 1.0) < 0.7:
                    bp = blueprint_library.find(random.choice(EXO_BP))
                    vehicle = world.try_spawn_actor(bp, spawn_points[spawn_id])
                    if vehicle is not None:
                        traffic_manager.set_desired_speed(vehicle, 20.)
                        traffic_manager.ignore_lights_percentage(vehicle, 25.)
                        vehicle.set_autopilot(True, tfm_port)
                        vehicles.append(vehicle)

            obs = Observation(args.out_folder, args.np)
            sensors = setup_sensors(ego_vehicle, blueprint_library, obs, world)

            for _ in range(10):
                world.tick()

            noise_steps = 0
            for i in range(args.nb_frames):
                control = ego_vehicle.get_control()
                if i % 28 == 0:
                    noise_steps = 0

                if noise_steps <= 7:
                    control.steer += np.random.normal(0.0, 0.05)
                    noise_steps += 1
                ego_vehicle.apply_control(control)

                w_frame = world.tick()

                spectator.set_transform(get_spectator_transform(ego_vehicle.get_transform()))

                car_loc = ego_vehicle.get_location()
                current_wp = carla_map.get_waypoint(car_loc)
                
                distance_center = distance_from_center(current_wp, current_wp.next(0.001)[0], car_loc)
                d_left, d_right, _, _ = dist_to_roadline(carla_map, ego_vehicle)
                d_roadline = min(d_left, d_right)

                ego_yaw = ego_vehicle.get_transform().rotation.yaw
                wp_yaw = current_wp.transform.rotation.yaw

                if ego_yaw < 0:
                    ego_yaw += 360
                if wp_yaw < 0:
                    wp_yaw += 360
                yaw_diff = abs(ego_yaw - wp_yaw)
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff

                speed = ego_vehicle.get_velocity()
                speed = math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

                obstacle_dist = obs.obstacle_dist[1] if abs(obs.obstacle_dist[0]-w_frame) < 2 else 25.

                if i%args.freq_save == 0:
                    info[w_frame] = [obstacle_dist, distance_center, yaw_diff, d_roadline, current_wp.is_junction, speed]
                    obs.save_data(w_frame)
                    with open(f"{args.out_folder}/info.pkl", 'wb') as f:
                        pickle.dump(info, f)


            for sensor in sensors:
                sensor.stop()
                if sensor.is_alive:
                    sensor.destroy()
            for vehicle in vehicles:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False, tfm_port)
                    vehicle.destroy()
            if ego_vehicle is not None:
                ego_vehicle.set_autopilot(False, tfm_port)
                ego_vehicle.destroy()
            sensors = []
            ego_vehicle = None
            vehicles = []

    finally:
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()

        if ego_vehicle is not None:
            ego_vehicle.set_autopilot(False, tfm_port)
            ego_vehicle.destroy()
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.set_autopilot(False, tfm_port)
                vehicle.destroy()

        if original_settings is not None:
            world.apply_settings(original_settings)
   
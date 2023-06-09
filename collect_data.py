import os
import math
import random
from argparse import ArgumentParser
import pickle
import queue

import numpy as np
import carla

from utils import to_rgb_array, depth_to_array, labels_to_array

SENSOR_TYPE = {'CAMERA': 0, 'DEPTH': 1, 'SEMANTIC': 2, 'OBSTACLE': 3}

def sensor_callback(sensor_data, sensor_queue, sensor_type):
    if sensor_type == SENSOR_TYPE['OBSTACLE']:
        actor = sensor_data.other_actor
        if 'vehicle' in actor.type_id:
            sensor_queue.put((sensor_data.frame, sensor_type, sensor_data.distance))
    else:
        sensor_queue.put((sensor_data.frame, sensor_type, image_to_array(sensor_data, sensor_type)))

def image_to_array(image, type_):
    if type_ == SENSOR_TYPE['CAMERA']:
        array = to_rgb_array(image)
    elif type_ == SENSOR_TYPE['DEPTH']:
        array = depth_to_array(image)
    elif type_ == SENSOR_TYPE['SEMANTIC']:
        array = labels_to_array(image)
    return array

if __name__=='__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--world-port', type=int, default=2000)
    argparser.add_argument('--host', type=str, default='localhost')
    argparser.add_argument('--map', type=str, default='Town01')
    argparser.add_argument('--weather', type=str, default='ClearNoon',
                           choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',
                                    'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset',
                                    'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon',
                                    'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset'])
    argparser.add_argument('--height', type=int, default=480)
    argparser.add_argument('--width', type=int, default=768)
    argparser.add_argument('--fov', type=int, default=100)
    argparser.add_argument('--nb_vehicles', type=int, default=50)
    argparser.add_argument('--out_sensors', type=str, default='./sensor_data')

    args = argparser.parse_args()

    #Create output directory
    os.makedirs(args.out_sensors, exist_ok=True)
    camera_path = os.path.join(args.out_sensors, 'camera')
    depth_path = os.path.join(args.out_sensors, 'depth')
    semantic_path = os.path.join(args.out_sensors, 'semantic')
    os.makedirs(camera_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(semantic_path, exist_ok=True)
    

    weather_presets = {'ClearNoon': carla.WeatherParameters.ClearNoon,
                       'ClearSunset': carla.WeatherParameters.ClearSunset,
                       'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                       'CloudySunset': carla.WeatherParameters.CloudySunset,
                       'WetNoon': carla.WeatherParameters.WetNoon,
                       'WetSunset': carla.WeatherParameters.WetSunset,
                       'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
                       'MidRainSunset': carla.WeatherParameters.MidRainSunset,
                       'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
                       'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
                       'HardRainNoon': carla.WeatherParameters.HardRainNoon,
                       'HardRainSunset': carla.WeatherParameters.HardRainSunset,
                       'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
                       'SoftRainSunset': carla.WeatherParameters.SoftRainSunset}

    sensors = []
    vehicles = []
    ego_vehicle = None
    original_settings = None
    sensor_queue = queue.Queue()
    try:
        #Connect client to server
        client = carla.Client(args.host, args.world_port)
        client.set_timeout(20.0)

        #Load world
        world = client.get_world() #client.load_world(args.map)
        original_settings = world.get_settings()
        world.set_weather(weather_presets[args.weather])
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        #Set traffic manager
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(False)
        traffic_manager.global_percentage_speed_difference(0)

        #Create vehicles
        ego_vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        while True:
            spawn_point = random.choice(spawn_points)
            ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, spawn_point)
            if ego_vehicle is not None:
                ego_vehicle.set_autopilot(True)
                traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
                traffic_manager.auto_lane_change(ego_vehicle, True)
                break
        
        for _ in range(args.nb_vehicles):
            while True:
                spawn_point = random.choice(spawn_points)
                vehicle = world.try_spawn_actor(random.choice(blueprint_library.filter('vehicle.*')),
                                                spawn_point)
                if vehicle is not None:
                    vehicle.set_autopilot(True)
                    vehicles.append(vehicle)
                    break

        #Create sensors
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', str(args.fov))
        camera_bp.set_attribute('sensor_tick', '0.2')
        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['CAMERA']))
        sensors.append(camera)

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(args.width))
        depth_bp.set_attribute('image_size_y', str(args.height))
        depth_bp.set_attribute('fov', str(args.fov))
        depth_bp.set_attribute('sensor_tick', '0.2')
        depth_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        depth = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle)
        depth.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['DEPTH']))
        sensors.append(depth)

        semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(args.width))
        semantic_bp.set_attribute('image_size_y', str(args.height))
        semantic_bp.set_attribute('fov', str(args.fov))
        semantic_bp.set_attribute('sensor_tick', '0.2')
        semantic_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        semantic = world.spawn_actor(semantic_bp, semantic_transform, attach_to=ego_vehicle)
        semantic.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['SEMANTIC']))
        sensors.append(semantic)

        obstacle_bp = blueprint_library.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('only_dynamics', 'False')
        obstacle_bp.set_attribute('distance', '10')
        obstacle_bp.set_attribute('sensor_tick', '0.2')
        obstacle_transform = carla.Transform()
        obstacle = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=ego_vehicle)
        obstacle.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['OBSTACLE']))
        sensors.append(obstacle)

        
        while True:
            w_frame = world.tick()
            
                

        

    finally:
        sensor_data = []
        while not sensor_queue.empty():
            data = sensor_queue.get()
            sensor_data.append(data)
        with open("data.pkl", 'wb') as f:
            pickle.dump(sensor_data, f)



        for sensor in sensors:
            sensor.destroy()

        if ego_vehicle is not None:
            ego_vehicle.destroy()
        for vehicle in vehicles:
            vehicle.destroy()

        

        
        
        # if original_settings is not None:
        #     world.apply_settings(original_settings)
   
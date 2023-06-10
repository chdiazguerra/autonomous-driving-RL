import os
import math
import random
from argparse import ArgumentParser
import pickle
import queue

import numpy as np
import carla

from utils import to_rgb_array, depth_to_array, labels_to_array, distance_from_center

SENSOR_TYPE = {'CAMERA': 0, 'DEPTH': 1, 'SEMANTIC': 2, 'OBSTACLE': 3}

def sensor_callback(sensor_data, sensor_queue, sensor_type, save_path = None):
    if sensor_type == SENSOR_TYPE['OBSTACLE']:
        actor = sensor_data.other_actor
        if 'vehicle' in actor.type_id:
            sensor_queue.put((sensor_data.frame, sensor_type, sensor_data.distance))
    else:
        array = image_to_array(sensor_data, sensor_type)
        sensor_queue.put((sensor_data.frame, sensor_type))
        np.save(save_path + '/%08d' % sensor_data.frame, array)

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
    argparser.add_argument('--out_folder', type=str, default='./sensor_data')
    argparser.add_argument('--nb_frames', type=int, default=10000)
    argparser.add_argument('--tick', type=float, default=0.5)

    args = argparser.parse_args()

    #Create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    camera_path = os.path.join(args.out_folder, 'camera')
    depth_path = os.path.join(args.out_folder, 'depth')
    semantic_path = os.path.join(args.out_folder, 'semantic')
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
    info = {}
    try:
        #Connect client to server
        client = carla.Client(args.host, args.world_port)
        client.set_timeout(20.0)

        #Load world
        world = client.load_world(args.map)
        original_settings = world.get_settings()
        world.set_weather(weather_presets[args.weather])
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        carla_map = world.get_map()
        spectator = world.get_spectator()

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
        camera_bp.set_attribute('sensor_tick', str(args.tick))
        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['CAMERA'], camera_path))
        sensors.append(camera)

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(args.width))
        depth_bp.set_attribute('image_size_y', str(args.height))
        depth_bp.set_attribute('fov', str(args.fov))
        depth_bp.set_attribute('sensor_tick', str(args.tick))
        depth_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        depth = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle)
        depth.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['DEPTH'], depth_path))
        sensors.append(depth)

        semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(args.width))
        semantic_bp.set_attribute('image_size_y', str(args.height))
        semantic_bp.set_attribute('fov', str(args.fov))
        semantic_bp.set_attribute('sensor_tick', str(args.tick))
        semantic_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        semantic = world.spawn_actor(semantic_bp, semantic_transform, attach_to=ego_vehicle)
        semantic.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['SEMANTIC'], semantic_path))
        sensors.append(semantic)

        obstacle_bp = blueprint_library.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('only_dynamics', 'False')
        obstacle_bp.set_attribute('distance', '20')
        obstacle_bp.set_attribute('sensor_tick', str(args.tick))
        obstacle_transform = carla.Transform()
        obstacle = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=ego_vehicle)
        obstacle.listen(lambda data: sensor_callback(data, sensor_queue, SENSOR_TYPE['OBSTACLE']))
        sensors.append(obstacle)

        for _ in range(args.nb_frames):
            w_frame = world.tick()

            transform = camera.get_transform()
            spectator.set_transform(transform)

            car_loc = ego_vehicle.get_location()
            current_wp = carla_map.get_waypoint(car_loc)
            
            distance = distance_from_center(current_wp, current_wp.next(0.001)[0], car_loc)

            yaw = ego_vehicle.get_transform().rotation.yaw
            dx = current_wp.next(0.001)[0].transform.location.x - current_wp.transform.location.x
            dy = current_wp.next(0.001)[0].transform.location.y - current_wp.transform.location.y
            yaw_wp = math.atan2(dy, dx)
            yaw_diff = yaw - math.degrees(yaw_wp)

            speed = ego_vehicle.get_velocity()
            speed = math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

            info[w_frame] = [distance, yaw_diff, speed, current_wp.is_junction]

    finally:
        sensor_data = []
        while not sensor_queue.empty():
            data = sensor_queue.get()
            sensor_data.append(data)
        with open(f"{args.out_folder}/data.pkl", 'wb') as f:
            pickle.dump(sensor_data, f)

        with open(f"{args.out_folder}/info.pkl", 'wb') as f:
            pickle.dump(info, f)

        for sensor in sensors:
            sensor.destroy()

        if ego_vehicle is not None:
            ego_vehicle.set_autopilot(False)
            ego_vehicle.destroy()
        for vehicle in vehicles:
            vehicle.set_autopilot(False)
            vehicle.destroy()

        if original_settings is not None:
            world.apply_settings(original_settings)
   
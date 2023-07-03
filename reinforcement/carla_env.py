import random
import math

import carla

from data.utils import to_rgb_array, depth_to_logarithmic_grayscale, depth_to_array, labels_to_array, low_resolution_semantics


class Observation:
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.semantic = None
        self.obstacle_dist = (0, math.inf) #(Frame detected, distance)

class CarlaEnv:
    def __init__(self, world_port=2000, host='localhost', map='Town01', weather='ClearNoon',
                 cam_height=256, cam_width=256, fov=100, nb_vehicles=40, tick=0.05,
                 nb_frames_max=10000, low_semantic=True):
        
        assert map in ['Town01', 'Town02']
        
        #Configurations
        self.client = carla.Client(host, world_port)
        self.client.set_timeout(60.0)

        self.map_name = map
        self.world = self.client.load_world(map)

        self.world.set_weather(getattr(carla.WeatherParameters, weather, carla.WeatherParameters.ClearNoon))

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = tick
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle_bp = self.world.get_blueprint_library().find('vehicle.audi.a2')

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(False)
        self.traffic_manager.global_percentage_speed_difference(0)

        self.cam_height = cam_height
        self.cam_width = cam_width
        self.fov = fov
        self.nb_vehicles = nb_vehicles
        self.tick = tick
        self.nb_frames_max = nb_frames_max
        self.low_semantic = low_semantic

        # Data
        self.frame = 0
        self.vehicles = []
        self.ego_vehicle = None
        self.end_location = None
        self.sensors = []
        self.obs = Observation()

        self.collision_hist = []

        #Spectator
        self.spectator = self.world.get_spectator()

        with open("log.txt", "a") as f:
            f.write("New episode\n")


    def reset(self):
        self.delete_actors()
        
        self.frame = 0
        self.collision_hist = []
        self.vehicles = []
        self.ego_vehicle = None
        self.sensors = []
        self.obs = Observation()

        self.spawn_ego_vehicle()
        self.spawn_vehicles()
        
        for _ in range(20):
            self.world.tick()

        return self.get_observation()
    
    def step(self, action):
        #Apply control
        #Tick
        #Terminated?
        #Reward
        #Return obs, reward, done, info

        self.world.tick()
        self.frame += 1

        transform = self.sensors[0].get_transform()
        self.spectator.set_transform(transform)

        
    
    def spawn_vehicles(self):
        spawn_points = self.world.get_map().get_spawn_points()
        for _ in range(self.nb_vehicles):
            while True:
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.try_spawn_actor(random.choice(self.blueprint_library.filter('vehicle.*')),
                                                spawn_point)
                if vehicle is not None:
                    vehicle.set_autopilot(True)
                    self.vehicles.append(vehicle)
                    break

    def spawn_ego_vehicle(self):
        while True:
            try:
                spawn_point, end_point = self.get_spawn_end_point()
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, spawn_point)

                self.end_location = end_point.location
                break
            except:
                continue
        self.set_rgb_camera()
        self.set_depth_camera()
        self.set_semantic_camera()
        self.set_collision_sensor()
        self.set_obstacle_sensor()

        self.ego_vehicle.set_autopilot(True)

    def set_rgb_camera(self):
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.cam_width))
        cam_bp.set_attribute('image_size_y', str(self.cam_height))
        cam_bp.set_attribute('fov', str(self.fov))
        cam_bp.set_attribute('sensor_tick', str(self.tick))

        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda data: self.process_rgb_img(data))

    def set_depth_camera(self):
        cam_bp = self.blueprint_library.find('sensor.camera.depth')
        cam_bp.set_attribute('image_size_x', str(self.cam_width))
        cam_bp.set_attribute('image_size_y', str(self.cam_height))
        cam_bp.set_attribute('fov', str(self.fov))
        cam_bp.set_attribute('sensor_tick', str(self.tick))

        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda data: self.process_depth_img(data))

    def set_semantic_camera(self):
        cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute('image_size_x', str(self.cam_width))
        cam_bp.set_attribute('image_size_y', str(self.cam_height))
        cam_bp.set_attribute('fov', str(self.fov))
        cam_bp.set_attribute('sensor_tick', str(self.tick))

        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda data: self.process_semantic_img(data))

    def set_collision_sensor(self):
        col_bp = self.blueprint_library.find('sensor.other.collision')
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(col_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda event: self.collision_hist.append(event))

    def set_obstacle_sensor(self):
        obs_bp = self.blueprint_library.find('sensor.other.obstacle')
        obs_bp.set_attribute('sensor_tick', str(self.tick))
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(obs_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda event: self.process_obstacle(event))

    def process_rgb_img(self, img):
        if img.raw_data is not None:
            array = to_rgb_array(img)
            self.obs.rgb = array

    def process_depth_img(self, img):
        if img.raw_data is not None:
            array = depth_to_logarithmic_grayscale(depth_to_array(img))
            self.obs.depth = array

    def process_semantic_img(self, img):
        if img.raw_data is not None:
            array = labels_to_array(img)
            if self.low_semantic:
                array = array.copy()
                low_resolution_semantics(array)
            self.obs.semantic = array

    def process_obstacle(self, event):
        if 'vehicle' in event.other_actor.type_id:
            self.obs.obstacle_dist = (self.frame, event.distance)
        else:
            self.obs.obstacle_dist = (self.frame, math.inf)

    def delete_actors(self):
        for actor in self.sensors+self.vehicles:
            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - destroy it
            if actor.is_alive:
                if isinstance(actor, carla.Vehicle):
                    actor.set_autopilot(False)
                actor.destroy()
        
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.set_autopilot(False)
            self.ego_vehicle.destroy()

        
        for _ in range(20):
            self.world.tick()
        
    def get_observation(self):
        pass

    def get_spawn_end_point(self):
        town1_sp1 = carla.Transform(carla.Location(x=256.551208, y=2.020164, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-0.000092, roll=0.000000))
        town1_end1 = carla.Transform(carla.Location(x=118.949997, y=55.840000, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=179.999756, roll=0.000000))
        
        town1_sp2 = carla.Transform(carla.Location(x=338.979980, y=249.429993, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-90.000298, roll=0.000000))
        town1_end2 = carla.Transform(carla.Location(x=92.109985, y=159.949997, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-90.000298, roll=0.000000))
        
        town1_sp3 = carla.Transform(carla.Location(x=22.179979, y=330.459991, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-0.000092, roll=0.000000))
        town1_end3 = carla.Transform(carla.Location(x=151.119736, y=198.759842, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-0.000092, roll=0.000000))
        
        points = {"Town01": [(town1_sp1, town1_end1), (town1_sp2, town1_end2), (town1_sp3, town1_end3)]}

        rand = random.choice(points[self.map_name])

        return rand

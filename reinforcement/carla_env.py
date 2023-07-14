import random
import math

import carla
import numpy as np

from data.utils import to_rgb_array, depth_to_logarithmic_grayscale, depth_to_array, labels_to_array, low_resolution_semantics, distance_from_center


class Observation:
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.semantic = None
        self.obstacle_dist = (0, math.inf) #(Frame detected, distance)

class Route:
    def __init__(self, init, end, orientation, junc_ids, junc_movement):
        assert orientation in ['N', 'S', 'W', 'E']
        assert len(junc_ids) == len(junc_movement)
        
        self.init = init
        self.end = end
        self.orientation = orientation
        self.junc_ids = junc_ids #Junctions ids
        self.junc_movement = junc_movement #Movement to take at each junction
        self.max_distance = self.distance_to_goal(self.init.location) + 100
    
    @classmethod
    def town1_1(cls):
        init = carla.Transform(carla.Location(x=229.781677, y=-1.959888, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=179.999634, roll=0.000000))
        end = carla.Transform(carla.Location(x=118.949997, y=55.840000, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=179.999756, roll=0.000000))
        
        junc_ids = [54, 332]
        junc_movement = [0, 2]

        orientation = 'W'

        return cls(init, end, orientation, junc_ids, junc_movement)

    @classmethod
    def town1_2(cls):
        init = carla.Transform(carla.Location(x=338.979980, y=249.429993, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-90.000298, roll=0.000000))
        end = carla.Transform(carla.Location(x=92.109985, y=159.949997, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-90.000298, roll=0.000000))
        
        junc_ids = [194, 255]
        junc_movement = [0, 2]

        orientation = 'N'

        return cls(init, end, orientation, junc_ids, junc_movement)
    
    @classmethod
    def town1_3(cls):
        init = carla.Transform(carla.Location(x=22.179979, y=330.459991, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-0.000092, roll=0.000000))
        end = carla.Transform(carla.Location(x=151.119736, y=198.759842, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-0.000092, roll=0.000000))
        
        junc_ids = [87, 255]
        junc_movement = [0, 2]

        orientation = 'E'

        return cls(init, end, orientation, junc_ids, junc_movement)

    @classmethod
    def get_possibilities(cls, name):
        if name == 'Town01':
            return [cls.town1_1, cls.town1_2, cls.town1_3]
        elif name == 'Town02':
            return []
        else:
            raise NotImplementedError()
        
    def distance_to_goal(self, loc):
        return math.sqrt((self.end.location.x - loc.x)**2 + (self.end.location.y - loc.y)**2)
    
    def is_oversteer(self, rotation):
        dalpha = 2
        if self.orientation == 'N':
            return 90-dalpha < rotation.yaw < 90+dalpha
        elif self.orientation == 'S':
            return -90-dalpha < rotation.yaw < -90+dalpha
        elif self.orientation == 'E':
            return (180-dalpha < rotation.yaw <= 180) or (-180 <= rotation.yaw < -180+dalpha)
        elif self.orientation == 'W':
            return (0-dalpha < rotation.yaw <= 0) or (0 <= rotation.yaw < 0+dalpha)
        else:
            raise NotImplementedError()

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
        self.map = self.world.get_map()

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
        self.sensors = []
        self.obs = Observation()
        self.route = None

        self.collision_hist = []

        #Spectator
        self.spectator = self.world.get_spectator()

    def reset(self, route_id=None):
        self.delete_actors()
        
        self.frame = 0
        self.collision_hist = []
        self.vehicles = []
        self.ego_vehicle = None
        self.sensors = []
        self.obs = Observation()
        self.intersections = 2

        self.spawn_ego_vehicle(route_id)
        self.spawn_vehicles()
        
        for _ in range(20):
            self.world.tick()

        return self.get_observation()
    
    def step(self, action):
        """_summary_

        Args:
            action (iterable): (steer, throttle, brake)

        Returns:
            tuple: Observation (image, movementType), where movementType is -1 for left, 0 for straight, 1 for right
            float: Reward
            bool: Done
            dict: Info
        """
        #Action = (steer, throttle, brake)
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)

        if action[1] >= action[2]:
            action[2] = 0.0
        else:
            action[1] = 0.0

        control = carla.VehicleControl(steer=action[0], throttle=action[1], brake=action[2])

        self.ego_vehicle.apply_control(control)

        #Tick
        self.world.tick()
        self.frame += 1

        #Rewards
        reward = 0
        done = False

        #Params
        beta = 3
        dmin = 1.5
        dmax = 3.0
        vmin = 10
        vmax = 60
        vtarget = 40
        amin = 20
        amax = 30
        obstacle_radius = 2.0
        dist_reach_goal = 2.0


        ego_transform = self.ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_rot = ego_transform.rotation

        dist_to_goal = self.route.distance_to_goal(ego_loc)
        v = self.ego_vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        #Terminated by max number of frames
        if self.frame >= self.nb_frames_max:
            done = True

        #Terminated by maximum distance to goal
        if dist_to_goal > self.route.max_distance:
            done = True

        #Terminated by collision
        if len(self.collision_hist) > 0:
            done = True
            reward = -1000

        #Terminated by oversteering
        if self.route.is_oversteer(ego_rot):
            done = True
            reward = -1000
        
        #Terminated by reaching the goal
        if dist_to_goal < dist_reach_goal:
            done = True
            reward = 500
        
        #Distance to goal
        if dist_to_goal <= dmax:
            reward += 5*(1-(dist_to_goal/dmax)**beta)
        else:
            reward += -5

        #Speed limit
        if kmh <= vmin:
            reward += (kmh/vmin)**beta
        elif kmh <= vtarget:
            reward += 1
        elif kmh <= vmax:
            reward += 1 - ((kmh-vtarget)/(vmax-vtarget))**beta
        else:
            reward += -1

        #Lane alignment
        wp = self.map.get_waypoint(ego_loc)
        wp_rot = wp.transform.rotation

        ego_yaw = ego_rot.yaw
        if ego_yaw < 0:
            ego_yaw += 360
        wp_yaw = wp_rot.yaw
        if wp_yaw < 0:
            wp_yaw += 360
        yaw_diff = abs(ego_yaw - wp_yaw)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        
        if yaw_diff <= amin:
            reward += 1 - (yaw_diff/amin)**beta
        elif yaw_diff < amax:
            reward += -1 + ((amax-yaw_diff)/(amax-amin))**beta
        else:
            reward += -1

        #Distance from lane center
        d = distance_from_center(wp.next(0.001)[0], wp, ego_loc)
        if d <= dmin:
            reward += 1 - (d/dmin)**beta
        elif d < dmax:
            reward += -1 + ((dmax-d)/(dmax-dmin))**beta
        else:
            reward += -1

        #Obstacle distance
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                loc = vehicle.get_location()
                dist = math.sqrt((loc.x-ego_loc.x)**2 + (loc.y-ego_loc.y)**2 + (loc.z-ego_loc.z)**2)
                if dist < obstacle_radius:
                    reward += (dist/obstacle_radius)**beta - 1

        transform = self.sensors[0].get_transform()
        self.spectator.set_transform(transform)

        info = {'speed': kmh, 'distance_to_goal': dist_to_goal, 'distance_from_lane_center': d, 'yaw_diff': yaw_diff}

        #Return obs, reward, done, info
        return self.get_observation(), reward, done, info
    
    def get_observation(self):
        #(RGB+DEPTH, LEFT/STRAIGHT/RIGHT)
        image = np.concatenate([self.obs.rgb, np.atleast_3d(self.obs.depth)], axis=-1)
        image = image/255.0

        ego_loc = self.ego_vehicle.get_location()
        waypoint = self.map.get_waypoint(ego_loc)

        movement = 1 #STRAIGHT

        if waypoint.is_junction:
            junc = waypoint.get_junction()
            if junc.id in self.route.junc_ids:
                ind = self.route.junc_ids.index(junc.id)
                movement = self.route.junc_movement[ind]
        
        return [image, movement]
    
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

    def spawn_ego_vehicle(self, route_id=None):
        while True:
            try:
                route = self.get_route(route_id)
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, route.init)
                self.route = route
                break
            except:
                continue
        self.set_rgb_camera()
        self.set_depth_camera()
        self.set_semantic_camera()
        self.set_collision_sensor()
        self.set_obstacle_sensor()

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

    def get_route(self, route_id=None):
        possible = Route.get_possibilities(self.map_name)
        
        if route_id is not None:
            choice = possible[route_id]
        else:
            choice = random.choice(possible)

        return choice()
    
    def set_weather(self, weather_id):
        try:
            weather = getattr(carla.WeatherParameters, weather_id)
            self.world.set_weather(weather)
        except:
            print(f"Weather {weather_id} not found")

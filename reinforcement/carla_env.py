import random
import math

import carla
import numpy as np

from data.utils import to_rgb_array, depth_to_logarithmic_grayscale, depth_to_array, labels_to_array, low_resolution_semantics, distance_from_center, dist_to_roadline


class Observation:
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.semantic = None
        self.obstacle_dist = (0, math.inf) #(Frame detected, distance)

class Route:
    def __init__(self, init, end, orientation, junc_ids, junc_movement, spawn_points, x_lims, y_lims):
        assert orientation in ['N', 'S', 'W', 'E']
        assert len(junc_ids) == len(junc_movement)
        
        self.init = init
        self.end = end
        self.orientation = orientation
        self.junc_ids = junc_ids #Junctions ids
        self.junc_movement = junc_movement #Movement to take at each junction
        self.max_distance = self.distance_to_goal(self.init.location) + 50
        self.spawn_points = spawn_points
        self.x_lims = x_lims
        self.y_lims = y_lims
    
    @classmethod
    def town1_1(cls):
        init = carla.Transform(carla.Location(x=153.759995+np.random.uniform(-0.5, 0.5), y=28.900000, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=90.000046, roll=0.000000))
        end = carla.Transform(carla.Location(x=170.547531, y=59.902679, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=0.000061+np.random.uniform(-3.0, 3.0), roll=0.000000))
        
        junc_ids = [332]
        junc_movement = [0]

        orientation = 'S'

        spawn_points = [
                        carla.Transform(carla.Location(x=153.76+np.random.uniform(-0.5, 0.5),
                                                        y=28.9+np.random.uniform(6, 15),
                                                        z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                        yaw=90.0+np.random.uniform(-3., 3.),
                                                        roll=0.0)),
                        carla.Transform(carla.Location(x=170.54+np.random.uniform(-25., -2.),
                                                        y=59.90+np.random.uniform(-0.5, 0.5),
                                                        z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                        yaw=0.0+np.random.uniform(-3., 3.),
                                                        roll=0.0)),
                        carla.Transform(carla.Location(x=191.08+np.random.uniform(-30., 0.),
                                                        y=55.84+np.random.uniform(-0.5, 0.5),
                                                        z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                        yaw=180.0+np.random.uniform(-3., 3.),
                                                        roll=0.0)),
                        carla.Transform(carla.Location(x=158.12+np.random.uniform(-1.5, -0.5),
                                                        y=45.08+np.random.uniform(-7., 0.0),
                                                        z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                        yaw=-90.0+np.random.uniform(-3., 3.),
                                                        roll=0.0))
                            ]
        
        x_lims = [151.5, 200.0]
        y_lims = [0, 63.9]

        return cls(init, end, orientation, junc_ids, junc_movement, spawn_points, x_lims, y_lims)

    @classmethod
    def town1_2(cls):
        init = carla.Transform(carla.Location(x=119.469986, y=129.750000+np.random.uniform(-0.5, 0.5), z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=179.999756, roll=0.000000))
        end = carla.Transform(carla.Location(x=92.109985, y=105.661537, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-90.000298+np.random.uniform(-3.0, 3.0), roll=0.000000))
        
        junc_ids = [306]
        junc_movement = [2]

        orientation = 'W'

        c = np.random.choice(2)
        spawn_points = [
                        carla.Transform(carla.Location(x=119.47+np.random.uniform(-15., -6.),
                                                       y=129.75+np.random.uniform(-0.5, 0.5),
                                                       z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                       yaw=180.0+np.random.uniform(-3., 3.),
                                                       roll=0.0)),
                        carla.Transform(carla.Location(x=92.11+np.random.uniform(-0.5, 0.5),
                                                       y=105.66+(1-c)*np.random.uniform(5, 25)+c*np.random.uniform(33,40),
                                                       z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                       yaw=-90.0+np.random.uniform(-3., 3.),
                                                       roll=0.0)),
                        carla.Transform(carla.Location(x=88.62+np.random.uniform(-0.5, 0.5),
                                                       y=101.83+np.random.uniform(0, 15),
                                                       z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                       yaw=90.0+np.random.uniform(-3., 3.),
                                                       roll=0.0)),
                        carla.Transform(carla.Location(x=105.57+np.random.uniform(0.0, 7.0),
                                                       y=133.77+np.random.uniform(-1.5, -0.5),
                                                       z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                       yaw=0.0+np.random.uniform(-3., 3.),
                                                       roll=0.0))
                        ]

        x_lims = [89.38, 150.0]
        y_lims = [90.0, 133.0]

        return cls(init, end, orientation, junc_ids, junc_movement, spawn_points, x_lims, y_lims)
    
    @classmethod
    def town1_3(cls):
        init = carla.Transform(carla.Location(x=395.959991+np.random.uniform(-0.5, 0.5), y=308.209991, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-90.000298, roll=0.000000))
        end = carla.Transform(carla.Location(x=396.637604, y=208.829865, z=0.300000),
                                    carla.Rotation(pitch=0.000000, yaw=-89.999939+np.random.uniform(-3.0, 3.0), roll=0.000000))
        
        junc_ids = []
        junc_movement = []

        orientation = 'N'

        spawn_points = [
                        carla.Transform(carla.Location(x=395.96+np.random.uniform(-0.5, 0.5),
                                                       y=308.21+np.random.uniform(-30., -6.),
                                                       z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                       yaw=-90.0+np.random.uniform(-3., 3.),
                                                       roll=0.0)),
                        carla.Transform(carla.Location(x=392.47+np.random.uniform(0.5, 1.5),
                                                       y=229.07+np.random.uniform(0, 30),
                                                       z=0.3),
                                        carla.Rotation(pitch=0.0,
                                                       yaw=90.0+np.random.uniform(-3., 3.),
                                                       roll=0.0))
                        ]
        
        x_lims = [390.0, 400.0]
        y_lims = [150.0, 350.0]

        return cls(init, end, orientation, junc_ids, junc_movement, spawn_points, x_lims, y_lims)

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
        dalpha = 5
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
    
    def is_out_of_limits(self, loc):
        return not (self.x_lims[0] < loc.x < self.x_lims[1] and self.y_lims[0] < loc.y < self.y_lims[1])

class CarlaEnv:
    def __init__(self, world_port=2000, host='localhost', map='Town01', weather='ClearNoon',
                 cam_height=256, cam_width=256, fov=100, nb_vehicles=40, tick=0.05,
                 nb_frames_max=10000, low_semantic=True, get_semantic=False):
        
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
        self.get_semantic = get_semantic

        # Data
        self.frame = 0
        self.vehicles = []
        self.ego_vehicle = None
        self.sensors = []
        self.obs = Observation()
        self.route = None

        self.collision_hist = []
        self.lane_invasion_hist = []
        self.current_movement = 1

        #Spectator
        self.spectator = self.world.get_spectator()

    def reset(self, route_id=None):
        self.delete_actors()
        
        self.frame = 0
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.vehicles = []
        self.ego_vehicle = None
        self.sensors = []
        self.obs = Observation()
        self.intersections = 2
        self.current_movement = 1
        self.route = self.get_route(route_id)

        self.spawn_ego_vehicle()
        self.spawn_vehicles()
        
        for _ in range(20):
            self.world.tick()

        return self.get_observation()
    
    def step(self, act):
        """_summary_

        Args:
            act (iterable): (steer, throttle)

        Returns:
            tuple: Observation (image, movementType), where movementType is 0 for left, 1 for straight, 2 for right
            float: Reward
            bool: Done
            dict: Info
        """
        #Action = (steer, throttle, brake)
        action = [0.0, 0.0, 0.0]

        action[0] = np.clip(act[0], -1, 1)

        if act[1] > 0:
            action[1] = np.clip(act[1], 0, 1)
        else:
            action[2] = np.clip(-act[1], 0, 1)

        control = carla.VehicleControl(steer=action[0], throttle=action[1], brake=action[2])

        self.ego_vehicle.apply_control(control)

        #Tick
        self.world.tick()
        self.frame += 1

        #Rewards
        reward = 0.0
        done = False

        #Params
        beta = 3
        dmin = 1.5
        dmax = 3.0
        vmin = 20
        vmax = 26
        vtarget = 25
        amin = 5
        amax = 20
        obstacle_radius = 2.0
        dist_reach_goal = 2.0
        speed_factor = 5.0


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
        
        #Terminated by lane invasion
        if len(self.lane_invasion_hist) > 0:
            done = True
            reward = -1000

        #Terminated by oversteering
        if self.route.is_oversteer(ego_rot):
            done = True
            reward = -1000
        
        #Terminated by reaching the goal
        if dist_to_goal < dist_reach_goal:
            done = True
            reward = 1000

        #Terminated by out of limits
        if self.route.is_out_of_limits(ego_loc):
            done = True
            reward = -1000

        #Distance to goal
        if dist_to_goal <= self.route.max_distance-50:
            reward += (1-(dist_to_goal/(self.route.max_distance-50))**beta)
        else:
            reward += -1

        #Speed limit
        if kmh <= vmin:
            reward += speed_factor*(kmh/vmin)**1
        elif kmh <= vtarget:
            reward += speed_factor
        elif kmh < vmax:
            reward += speed_factor*(1 - ((kmh-vtarget)/(vmax-vtarget))**(1/beta))
        else:
            reward += -speed_factor

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

        
        if self.current_movement == 1: #Avoid wrong readings when turning
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

        #Distance to roadline
        d_left, d_right, _, _ = dist_to_roadline(self.map, self.ego_vehicle)
        d_road = min(d_left, d_right)
        best_d = 1.0
        worst_d = 0.8
        if self.current_movement == 1:
            if d_road >= best_d:
                reward += 1.0
            elif d_road >= worst_d:
                reward += (d_road-worst_d)/(best_d-worst_d)
            else:
                reward += (d_road/worst_d)**beta - 1

        #Obstacle distance
        min_dist = math.inf
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                loc = vehicle.get_location()
                dist = math.sqrt((loc.x-ego_loc.x)**2 + (loc.y-ego_loc.y)**2 + (loc.z-ego_loc.z)**2)
                min_dist = min(min_dist, dist)
                if dist < obstacle_radius:
                    reward += (dist/obstacle_radius)**beta - 1

        #Leading vehicle distance
        frame, d_leading = self.obs.obstacle_dist
        if abs(frame-self.frame)>2:
            d_leading = math.inf
        if d_leading < obstacle_radius+2:
            reward += 10*((d_leading/(obstacle_radius+2))**beta - 1)

        #No movement
        if min_dist > 4.0 and kmh < 2:
            reward += -10

        #Steer differ from movement
        factor = 10
        min_steer = 0.1
        if self.current_movement == 0:
            if act[0] < -min_steer:
                reward += factor
            elif act[0] <= 0.0:
                reward += factor*(-act[0]/min_steer)**(1/beta)
            elif act[0] <= min_steer:
                reward += -factor*(act[0]/min_steer)**(1/beta)
            else:
                reward += -factor
        elif self.current_movement == 2:
            if act[0] >= min_steer:
                reward += factor
            elif act[0] >= 0.0:
                reward += factor*(act[0]/min_steer)**(1/beta)
            elif act[0] >= -min_steer:
                reward += -factor*(-act[0]/min_steer)**(1/beta)
            else:
                reward += -factor
        elif self.current_movement == 1 and (act[0] < -0.2 or act[0] > 0.2):
            reward += -factor

        transform = self.sensors[0].get_transform()
        self.spectator.set_transform(transform)

        info = {'speed': kmh, 'distance_to_goal': dist_to_goal, 'distance_from_lane_center': d, 'yaw_diff': yaw_diff,
                'distance_to_left': d_left, 'distance_to_right': d_right, 'distance_leading': d_leading,
                'collision': len(self.collision_hist) > 0, 'semantic': self.obs.semantic}

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

        self.current_movement = movement
        
        return [image, movement]
    
    def spawn_vehicles(self):
        #spawn_points = self.world.get_map().get_spawn_points()
        #possible_idx = self.route.spawn_points
        for spawn_point in self.route.spawn_points:
            #spawn_point = spawn_points[idx]
            if np.random.uniform(0., 1.0) < 0.9:
                vehicle = self.world.try_spawn_actor(self.world.get_blueprint_library().find('vehicle.audi.a2'),#random.choice(self.blueprint_library.filter('vehicle.*')),
                                                spawn_point)
                if vehicle is not None:
                    vehicle.set_autopilot(True)
                    self.traffic_manager.set_desired_speed(vehicle, 20.)
                    self.vehicles.append(vehicle)

    def spawn_ego_vehicle(self):
        while True:
            try:
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, self.route.init)
                break
            except:
                continue
        self.set_rgb_camera()
        self.set_depth_camera()
        if self.get_semantic:
            self.set_semantic_camera()
        self.set_collision_sensor()
        self.set_obstacle_sensor()
        self.set_lane_invasion_sensor()

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
        obs_bp.set_attribute('distance', '20')
        obs_bp.set_attribute('sensor_tick', str(self.tick))
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(obs_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda event: self.process_obstacle(event))

    def set_lane_invasion_sensor(self):
        obs_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(obs_bp, spawn_point, attach_to=self.ego_vehicle)
        self.sensors.append(sensor)

        sensor.listen(lambda event: self.lane_invasion_hist.append(event))

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

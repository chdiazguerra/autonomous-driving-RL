import math

import numpy

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a 2D array
    containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]


def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }
    array = labels_to_array(image)
    result = numpy.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[numpy.where(array == key)] = value
    return result


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(numpy.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = numpy.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth.astype(numpy.float32)


def depth_to_logarithmic_grayscale(normalized_depth):
    """
    Convert an image containing CARLA encoded depth-map to a logarithmic
    grayscale image array.
    "max_depth" is used to omit the points that are far enough.
    """
    # Convert to logarithmic depth.
    logdepth = numpy.ones(normalized_depth.shape) + \
        (numpy.log(normalized_depth) / 5.70378)
    logdepth = numpy.clip(logdepth, 0.0, 1.0)
    logdepth *= 255.0
    # Expand to three colors.
    return logdepth.astype(numpy.uint8)

def distance_from_center(previous_wp, current_wp, car_loc):
    """Computes the distance of the car from the center of the lane.

    Args:
        previous_wp: Previous Waypoint
        current_wp: Current Waypoint
        car_loc: Car location
    """

    prev_x = previous_wp.transform.location.x
    prev_y = previous_wp.transform.location.y

    curr_x = current_wp.transform.location.x
    curr_y = current_wp.transform.location.y

    car_x = car_loc.x
    car_y = car_loc.y

    a = curr_y - prev_y
    b = -(curr_x - prev_x)
    c = (curr_x - prev_x) * prev_y - (curr_y - prev_y) * prev_x

    d = abs(a * car_x + b * car_y + c) / (a ** 2 + b ** 2+1e-6) ** 0.5

    return d

def low_resolution_semantics(image):
    """Convert a CARLA semantic image (29 classes) to a low resolution semantic
    segmentation image (14 classes).
    WARNING: Image is overwritten."""
    mapping = {0: 0, #None
     1: 1, #Roads
     2: 2, #Sidewalks
     3: 3, #Buildings
     4: 4, #Walls
     5: 5, #Fences
     6: 6, #Poles
     7: 7, #TrafficLights
     8: 7, #TrafficSigns
     9: 8, #Vegetation
     10: 9, #Terrain -> Other
     11: 10, #Sky (added)
     12: 11, #Pedestrians
     13: 11, #Riders -> Pedestrian
     14: 12, #Cars -> Vehicles
     15: 12, #Trucks -> Vehicles
     16: 12, #Bus -> Vehicles
     17: 12, #Trains -> Vehicles
     18: 12, #Motorcycles -> Vehicles
     19: 12, #Bicycles -> Vehicles
     20: 9, #Static -> Other
     21: 9, #Dynamic -> Other
     22: 9, #Other -> Other
     23: 9, #Water -> Other
     24: 13, #RoadLines
     25: 9, #Ground -> Other
     26: 9, #Bridge -> Other
     27: 9, #RailTrack -> Other
     28: 9 #GuardRail -> Other
     }
    
    for i in range(8, 29):
        image[image == i] = mapping[i]

def dist_to_roadline(carla_map, vehicle):
    curr_loc = vehicle.get_transform().location
    yaw = vehicle.get_transform().rotation.yaw
    waypoint = carla_map.get_waypoint(curr_loc)
    waypoint_yaw = waypoint.transform.rotation.yaw
    yaw_diff = yaw - waypoint_yaw
    yaw_diff_rad = yaw_diff / 180 * math.pi

    bb = vehicle.bounding_box
    corners = bb.get_world_vertices(vehicle.get_transform())
    dis_to_left, dis_to_right = 100, 100
    for corner in corners:
        if corner.z < 1:
            waypt = carla_map.get_waypoint(corner)
            waypt_transform = waypt.transform
            waypoint_vec_x = waypt_transform.location.x - corner.x
            waypoint_vec_y = waypt_transform.location.y - corner.y
            dis_to_waypt = math.sqrt(waypoint_vec_x ** 2 + waypoint_vec_y ** 2)
            waypoint_vec_angle = math.atan2(waypoint_vec_y, waypoint_vec_x) * 180 / math.pi
            angle_diff = waypoint_vec_angle - waypt_transform.rotation.yaw
            if (angle_diff > 0 and angle_diff < 180) or (angle_diff > -360 and angle_diff < -180):
                dis_to_left = min(dis_to_left, waypoint.lane_width / 2 - dis_to_waypt)
                dis_to_right = min(dis_to_right, waypoint.lane_width / 2 + dis_to_waypt)
            else:
                dis_to_left = min(dis_to_left, waypoint.lane_width / 2 + dis_to_waypt)
                dis_to_right = min(dis_to_right, waypoint.lane_width / 2 - dis_to_waypt)

    return dis_to_left, dis_to_right, math.sin(yaw_diff_rad), math.cos(yaw_diff_rad)

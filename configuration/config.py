#Environment
WORLD_PORT = 2000
WORLD_HOST = 'localhost'
TICK = 0.05
TRAIN_MAP = 'Town01'
TEST_MAP = 'Town02'

#Camera
CAM_HEIGHT = 256
CAM_WIDTH = 256
CAM_FOV = 120 #90
CAM_POS_X = 2.1 #0.8
CAM_POS_Y = 0.0
CAM_POS_Z = 1.0 #1.7
CAM_PITCH = 10.0 #10
CAM_YAW = 0.0
CAM_ROLL = 0.0

#EGO VEHICLE
EGO_BP = 'vehicle.tesla.model3'

#EXO VEHICLES
EXO_BP = ['vehicle.tesla.model3', 'vehicle.ford.mustang', 'vehicle.dodge.charger_2020', 'vehicle.bmw.grandtourer']

#WEATHER
TRAIN_WEATHER = ['ClearNoon', 'ClearSunset', 'WetNoon', 'HardRainNoon']
TEST_WEATHER = []

#Autoencoder
AE_DATASET_FILE = '../autoencoderData/dataset.pkl'
AE_VAL_SIZE = 0.2
AE_IMG_SIZE = 'default'
AE_NORM_INPUT = True
AE_EMB_SIZE = 256
AE_BATCH_SIZE = 32
AE_EPOCHS = 200
AE_LR = 1e-3
AE_DIRPATH = '../AEModel'
AE_SPLIT = None
AE_PRETRAINED = None

AE_MODEL = 'AutoencoderSEM' #Autoencoder, AutoencoderSEM, VAE
AE_LOW_SEM = True
AE_USE_IMG_AS_OUTPUT = False
AE_ADDITIONAL_DATA = True #Not used if model is VAE


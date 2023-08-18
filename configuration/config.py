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
TEST_WEATHER = ['CloudyNoon', 'SoftRainSunset']

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
AE_PRETRAINED = '../autoencodersEntrenados/0/model.ckpt'

AE_MODEL = 'AutoencoderSEM' #Autoencoder, AutoencoderSEM, VAE
AE_LOW_SEM = True
AE_USE_IMG_AS_OUTPUT = False
AE_ADDITIONAL_DATA = True #Not used if model is VAE

#Expert data
EXPERT_DATA_FOLDER = '../datosExperto/withExoVehicles'

#Agent Training
AGENT_FOLDER = '../agents/ddpgColDeductiveRoute2'
TRAIN_EPISODES = 340
ROUTE_ID = 2
USE_EXO_VEHICLES = True

#DDPG Training
DDPG_USE_EXPERT_DATA = True
DDPG_EXPERT_DATA_FILE = '../datosExperto/withExoVehicles/Route_2.pkl'
DDPG_PRETRAIN_STEPS = 500
DDPG_USE_ENV_MODEL = True
DDPG_ENV_STEPS = 10
DDPG_BATCH_SIZE = 64
DDPG_NB_UPDATES = 5
DDPG_NOISE_SIGMA = 0.2
DDPG_SCH_STEPS = 5

#SAC Training
SAC_BATCH_SIZE = 64
SAC_NB_UPDATES = 50
SAC_ALPHA = 0.2
SAC_SCH_STEPS = 100
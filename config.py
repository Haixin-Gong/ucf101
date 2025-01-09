# Model type: 3dcnn, efficientnet
MODEL_TYPE = '3dcnn'

# Training parameters
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SEED = 42
TF_SEED = 768

# Frame parameters
FRAME_COUNT = 10
FRAME_STEP = 5
FRAME_SIZE = (224, 224)
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
CLIP_SIZE = (10, 224, 224, 3)
NUM_CLASSES = 10

# Device
DEVICE = 'GPU'

# Dataset paths
DATASET_PATH = "/home/gonghx/code/Dataset/kaggle/UCF101"
TRAIN_DIR = '/home/gonghx/code/Dataset/kaggle/UCF101/train'
VAL_DIR = '/home/gonghx/code/Dataset/kaggle/UCF101/val'
TEST_DIR = '/home/gonghx/code/Dataset/kaggle/UCF101/test'
TRAIN_CSV = '/home/gonghx/code/Dataset/kaggle/UCF101/train.csv'
VAL_CSV = '/home/gonghx/code/Dataset/kaggle/UCF101/val.csv'
TEST_CSV = '/home/gonghx/code/Dataset/kaggle/UCF101/test.csv'
# play params
SAVE_BOARD = True
PERSON = 'person'
MACHINE = 'machine'
DATASET_PATH_SAVE = 'games_data/new_data'
BRANCH_COUND = 10 # branch count of strategy model


# train params
VALUE_TARGET_ALLONES = False
DATASET_PATH_TRAIN = 'games_data/old_data/p2p'
# DATASET_PATH_TRAIN = 'games_data/new_data'
MODEL_NEW_TRAINED_PATH = 'models/new_train/'
STRATEGY_MODEL_NAME = 'models/strategy15.pth'
VALUE_MODEL_NAME = 'models/value15.pth'

STRATEGY_EPOCHS = 2
VALUE_EPOCHS = 1
BATCH_SIZE = 32


# game params
BOARD_SIZE = 15 # 个
CELL_SIZE = 40  # cm
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)



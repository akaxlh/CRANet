# Data Parameters
DATASET = 'ml-1m'
# DATASET = 'ml-10m'
# DATASET = 'netflix'
# DATASET = 'fsq'
RATE = 0.9
if DATASET == 'ml-20m':
	USER_NUM = 138493
	MOVIE_NUM = 26744
	DIVIDER = ','
	RATING = 'ratings.csv'
elif DATASET == 'ml-1m':
	USER_NUM = 6040
	MOVIE_NUM = 3706#3952
	DIVIDER = '::'
	RATING = 'ratings.dat'
elif DATASET == 'ml-10m':
	USER_NUM = 69878
	MOVIE_NUM = 10677
	DIVIDER = '::'
	RATING = 'ratings.dat'
elif DATASET == 'netflix':
	USER_NUM = 480189
	MOVIE_NUM = 17770#78305
	DIVIDER = ','
	RATING = 'combined_data_5_new.txt'
elif DATASET == 'fsq':
	USER_NUM = 24748
	MOVIE_NUM = 7763
	DIVIDER = '\t'
	TOP_K = 3

# Storage Parameters
LOAD_MODEL = ''
TRAIN_FILE = 'Datasets/' + DATASET + '/sparseMat_0.9_train.csv'
TEST_FILE = 'Datasets/' + DATASET + '/sparseMat_0.9_test.csv'
CV_FILE = 'Datasets/' + DATASET + '/sparseMat_0.9_cv.csv'

# Model Parameters
# Hyper-parameter values for ML-1M, ML-10M, Netflix and Foursquare respectively
MOVIE_BASED = True#True#True#True#False
W_WEIGHT = 0.05#0.05#0.05#0.01#0.01
V_WEIGHT = 0.05#0.05#0.01#0.01#0.01
LAMBDA = 0.001#0.001#0.001#0.0001#0.001
ENHANCE = 20#20#20#20#0.2

EPOCH = 120
LR = 1e-3
DECAY = .96
BATCH_SIZE = 32
LATENT_DIM = 500
if MOVIE_BASED:
	DECAY_STEP = MOVIE_NUM / BATCH_SIZE
else:
	DECAY_STEP = USER_NUM / BATCH_SIZE

SAVE_PATH = 'tem'
LOAD_MODEL = None
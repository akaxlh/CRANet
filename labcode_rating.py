import tensorflow as tf
from Params import *
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import numpy as np
from Utils.DataHandler import *
import pickle
import scipy
from scipy.sparse import csr_matrix
import sys
from tensorflow.core.protobuf import config_pb2
import os
from tensorflow.contrib.layers import xavier_initializer
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PhantomInput:
	def __init__(self, sess, datas, inputDim):
		self.sess = sess
		self.trainMat = datas[0]
		self.cvMat = datas[1]
		self.testMat = datas[2]
		inputDim = self.trainMat.shape[1]
		self.trainMask = self.trainMat != 0
		self.cvMask = self.cvMat != 0
		self.testMask = self.testMat != 0

		self.train_losses = list()
		self.train_RMSEs = list()
		self.test_losses = list()
		self.test_RMSEs = list()

		self.inputDim = inputDim

		self.metrics = dict()
		self.metrics['trainLoss'] = list()
		self.metrics['trainRMSE'] = list()
		self.metrics['cvLoss'] = list()
		self.metrics['cvRMSE'] = list()

		if FUSE_TRAIN_CV:
			self.trainMat = self.trainMat + self.cvMat
			self.trainMask = self.trainMat != 0
			self.cvMat = self.testMat
			self.cvMask = self.testMask

		log('Matrix Size: ' + str(self.trainMat.shape))

	def run(self):
		self.prepare_model()
		log('Model Prepared')
		stloc = 0
		if LOAD_MODEL != None:
			self.loadModel()
			stloc = len(self.metrics['trainLoss'])
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, EPOCH):
			loss, rmse = self.runEpoch(inpMat=self.trainMat, labelMat=self.trainMat, labelMask=self.trainMask,
				train=True)
			log('Epoch %d/%d, Train: Loss = %.3f, RMSE = %.3f' %\
				(ep, EPOCH, loss, rmse))
			if ep % 5 == 0:
				loss, rmse = self.runEpoch(inpMat=self.trainMat, labelMat=self.cvMat, labelMask=self.cvMask, steps=0.9)
				log('Epoch %d/%d, CV: Loss = %.3f, RMSE = %.3f' %\
					(ep, EPOCH, loss, rmse))
			if ep % 5 == 0:
				self.saveHistory()
			print('')
		loss, rmse = self.runEpoch(inpMat=self.trainMat, labelMat=self.testMat, labelMask=self.testMask)
		log('Overall, Test: Loss = %.3f, RMSE = %.5f' % (loss, rmse))
		self.saveHistory()

	def truncActivation(self, x, vmax, vmin, rate):
		return tf.minimum(vmin + (x-vmin)*rate,\
			tf.maximum(vmax+(x-vmax)*rate, x))

	def prepare_model(self):
		self.inputR = tf.placeholder(dtype=tf.float32, shape=[None, self.inputDim], name='inputR')
		self.mask = tf.placeholder(dtype=tf.float32, shape=[None, self.inputDim], name='mask')
		self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.inputDim], name='label')

		mu = tf.get_variable(name='mu', dtype=tf.float32, initializer=tf.zeros(shape=LATENT_DIM, dtype=tf.float32))
		b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.zeros(shape=self.inputDim, dtype=tf.float32))
		W = tf.get_variable(name='W', dtype=tf.float32, shape=[self.inputDim, LATENT_DIM], initializer=xavier_initializer(dtype=tf.float32))
		V = tf.get_variable(name='V', dtype=tf.float32, shape=[LATENT_DIM, self.inputDim], initializer=xavier_initializer(dtype=tf.float32))
		T = tf.get_variable(name='T', dtype=tf.float32, shape=[LATENT_DIM, LATENT_DIM], initializer=xavier_initializer(dtype=tf.float32))

		inputR = self.inputR
		inpNum = tf.reduce_sum(tf.sign(inputR), axis=-1, keepdims=True)
		avgWeight = (1 / (inpNum + 1e-6))

		tem = inputR @ W

		# phantomInp
		enhRep = tem * avgWeight * ENHANCE
		rep = tem + enhRep @ T

		# refnet
		# temmask = 1-tf.sign(inputR)
		# rep = ((tem * avgWeight * ENHANCE) @ tf.transpose(W) * temmask) @ W + tem

		latRep = tf.nn.sigmoid((rep + mu))
		recover = tf.identity(latRep @ V + b)

		notNonInput = tf.sign(tf.reduce_sum(tf.sign(inputR), axis=-1))
		self.defaultErr = tf.reduce_sum((1 - notNonInput) * (tf.reduce_sum(self.mask * tf.square(3 - self.label), axis=-1)))
		self.preLoss = tf.reduce_mean(notNonInput * tf.reduce_sum(self.mask * tf.square(recover - self.label), axis=-1))
		self.regLoss = W_WEIGHT * tf.reduce_sum(tf.square(W)) + V_WEIGHT * tf.reduce_sum(tf.square(V))

		refmat = ENHANCE * tf.transpose(W) @ W
		self.loss = self.preLoss + self.regLoss + LAMBDA * tf.reduce_sum(tf.square(T-refmat))
		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(LR, globalStep,
			DECAY_STEP, DECAY, staircase=True)

		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def runEpoch(self, inpMat, labelMat, labelMask, train=False, steps=-1):
		num = inpMat.shape[0]
		shuffledIds = np.random.permutation(num)
		epochLoss, epochRmse, epochNum = [0]*3
		temStep = int(np.ceil(num / BATCH_SIZE))
		if steps == -1 or steps > temStep:
			steps = temStep
		elif steps > 0 and steps < 1:
			steps = int(steps * temStep)
		for i in range(steps):
			st = i * BATCH_SIZE
			ed = min((i+1) * BATCH_SIZE, num)
			batchIds = shuffledIds[st: ed]
			temTrain = inpMat[batchIds].toarray()
			temLabel = labelMat[batchIds].toarray()
			temMask = labelMask[batchIds].toarray()
			target = [self.preLoss, self.loss, self.regLoss, self.defaultErr]
			if train:
				target = [self.optimizer] + target
			res = self.sess.run(target,
				feed_dict={
					self.inputR: temTrain,
					self.mask: temMask,
					self.label: temLabel,
				},
				options=config_pb2.RunOptions(
					report_tensor_allocations_upon_oom=True))
			preLoss, loss, regLoss, defaultErr = res[-4:]
			epochLoss += loss
			epochRmse += preLoss * (ed - st) + defaultErr
			epochNum += np.sum(temMask)

			log('Step %d/%d: loss = %.2f, regLoss = %.2f' %\
				(i, steps, loss, regLoss), save=False, oneline=True)
		epochRmse = np.sqrt(epochRmse / epochNum)
		epochLoss = epochLoss / steps
		if train:
			self.metrics['trainLoss'].append(epochLoss)
			self.metrics['trainRMSE'].append(epochRmse)
		else:
			self.metrics['cvLoss'].append(epochLoss)
			self.metrics['cvRMSE'].append(epochRmse)

		return epochLoss, epochRmse

	def saveHistory(self):
		if EPOCH == 0:
			return
		with open('History/' + SAVE_PATH + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + SAVE_PATH)
		log('Model Saved: %s' % SAVE_PATH)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + LOAD_MODEL)
		with open('History/' + LOAD_MODEL + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')

if __name__ == '__main__':
	if len(sys.argv) != 1:
		if len(sys.argv) == 3:
			SAVE_PATH = sys.argv[1]
			LOAD_MODEL = sys.argv[2]
		else:
			SAVE_PATH = sys.argv[1]

	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	trainMat = ReadMat(TRAIN_FILE)
	testMat = ReadMat(TEST_FILE)
	cvMat = ReadMat(CV_FILE)

	with tf.Session(config=config) as sess:
		PhantomInput = PhantomInput(sess, (trainMat, cvMat, testMat), USER_NUM if MOVIE_BASED else MOVIE_NUM)
		PhantomInput.run()

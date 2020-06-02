import tensorflow as tf
from Params import *
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import numpy as np
import pickle
from Utils.DataHandler import *
from scipy.sparse import csr_matrix
import sys
from tensorflow.core.protobuf import config_pb2
import os
import pickle
from Utils.NNLayers import FC, Bias, Activate, Regularize, BN
import Utils.NNLayers as NNs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Recommender:
	def __init__(self, sess, datas, inputDim):
		self.sess = sess
		self.trnMat, self.tstMat = datas
		self.inputDim = inputDim
		tstStat = np.reshape(np.array(np.sum(self.tstMat, axis=-1)), [-1])
		self.testIds = np.reshape(np.argwhere(tstStat!=0), [-1])
		print(tstStat.shape, len(self.testIds))

	def run(self):
		self.prepare_model()
		log('Model Prepared')
		stloc = 0
		if LOAD_MODEL != None:
			self.loadModel()
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, EPOCH):
			loss, precision, ndcg = self.runEpoch(inpMat=self.trnMat, labelMat=self.trnMat, train=True)
			log('Epoch %d/%d, Train: Loss = %.3f' % (ep, EPOCH, loss))
			if ep % 5 == 0:
				loss, precision, ndcg = self.runEpoch(inpMat=self.trnMat, labelMat=self.tstMat)
				log('Epoch %d/%d, CV, Loss = %.3f, Prc = %.3f, NDCG = %.3f' % (ep, EPOCH, loss, precision, ndcg))
				self.saveHistory()
			print()
		loss, precision, ndcg = self.runEpoch(inpMat=self.trnMat, labelMat=self.tstMat)
		log('Overall, Test: Loss = %.3f, Prc = %.4f, NDCG = %.4f' % (loss, precision, ndcg))
		self.saveHistory()

	def implicitReflection(self, inp):
		inpMask = tf.sign(inp)
		V = NNs.defineParam('V', [self.inputDim, LATENT_DIM], reg=True)
		T = NNs.defineParam('T', [LATENT_DIM, LATENT_DIM], reg=False)
		h1 = inp @ V
		h2 = h1 @ T * ENHANCE / (tf.reduce_sum(inpMask, axis=-1, keepdims=True) + 1e-6)
		h3 = tf.nn.sigmoid(Bias(h1 + h2))
		pred = FC(h3, self.inputDim, useBias=True, reg=True, activation='sigmoid')
		self.refLoss = tf.reduce_sum(tf.square(T - tf.transpose(V) @ V))
		return pred

	def prepare_model(self):
		self.inp = tf.placeholder(dtype=tf.float32, shape=[None, self.inputDim], name='inp')
		self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.inputDim], name='label')
		pred = self.implicitReflection(self.inp)
		labelMask = tf.to_float(self.label>0)
		self.pred = pred

		eps = 1e-6
		self.preLoss = -tf.reduce_sum(labelMask * tf.log(pred+eps) + (1-labelMask) * tf.log(1-pred+eps))
		self.regLoss = W_WEIGHT * Regularize() + LAMBDA * self.refLoss
		self.loss = self.preLoss + self.regLoss
		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(LR, globalStep, DECAY_STEP, DECAY, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def runEpoch(self, inpMat, labelMat, train=False):
		num = inpMat.shape[0]
		if train:
			sfIds = np.random.permutation(num)
		else:
			sfIds = self.testIds
			num = len(self.testIds)
		epochLoss, epochPrcNume, epochPrcDeno, epochNdcgNume, epochNdcgDeno = [0] * 5
		steps = int(np.ceil(num / BATCH_SIZE))
		for i in range(steps):
			st = i * BATCH_SIZE
			ed = min((i+1) * BATCH_SIZE, num)
			batchIds = sfIds[st: ed]
			temTrn = inpMat[batchIds].toarray()
			temLbl = labelMat[batchIds].toarray()
			target = [self.preLoss, self.loss, self.regLoss, self.pred]
			if train:
				target = [self.optimizer] + target
				mask = np.random.randint(0, 2, size=(ed-st, self.inputDim), dtype='int32')
				temTrn = temTrn * mask
				temLbl = temLbl * (1-mask)

			res = self.sess.run(target, 
				feed_dict={
					self.inp: temTrn,
					self.label: temLbl
				}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			preLoss, loss, regLoss, pred = res[-4:]
			epochLoss += loss
			if not train:
				prcNume, prcDeno, ndcgNume, ndcgDeno = self.getHit(pred, temLbl)
				epochPrcNume += prcNume
				epochPrcDeno += prcDeno
				epochNdcgNume += ndcgNume
				epochNdcgDeno += ndcgDeno

			log('Step %d/%d: loss = %.2f, regLoss = %.2f' % (i, steps, loss, regLoss), save=False, oneline=True)
		epochPrecision = epochPrcNume / (1e-6+epochPrcDeno)
		epochNdcg = epochNdcgNume / (1e-6+epochNdcgDeno)
		epochLoss /= steps
		return epochLoss, epochPrecision, epochNdcg

	def getHit(self, pred, label):
		user = pred.shape[0]
		item = pred.shape[1]
		prcNume, ndcgNume = [0] * 2
		prcDeno = user * TOP_K
		ndcgDeno = 0
		allItems = list(range(item))
		for i in range(user):
			tempred = list(zip(allItems, pred[i]))
			tempred.sort(key=lambda x: x[1], reverse=True)
			shoot = list(map(lambda x: x[0], tempred[:TOP_K]))
			for j in range(TOP_K):
				temid = shoot[j]
				if label[i][temid] > 0:
					prcNume += 1
					ndcgNume += np.reciprocal(np.log2(shoot.index(temid)+2))
			for j in range(np.sum(label[i]>0)):
				ndcgDeno += 1 / np.log2(j+2)
		return prcNume, prcDeno, ndcgNume, ndcgDeno

	def saveHistory(self):
		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + SAVE_PATH)
		log('Model Saved: %s' % SAVE_PATH)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + LOAD_MODEL)
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

	def trans(mat):
		mat = mat.toarray()
		mask = (mat > 0)
		ret = mat * (1 - mask) + np.minimum(mat * mask, 20.0) * 5 / 20
		return csr_matrix(ret)
	trainMat = trans(ReadMat(TRAIN_FILE))
	testMat = trans(ReadMat(TEST_FILE))

	with tf.Session(config=config) as sess:
		model = Recommender(sess, (trainMat, testMat), USER_NUM if MOVIE_BASED else MOVIE_NUM)
		model.run()
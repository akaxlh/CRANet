import pickle

def ReadMat(file):
	with open(file, 'rb') as fs:
		ret = pickle.load(fs)
	return ret
import pickle


def save_dataset(data_dict, path):
	with open(path, 'wb') as file:
		pickle.dump(data_dict, file)
	print('[INFO] Dataset succefully saved')

def load_dataset(path):
	with open(path, 'rb') as file:
		dataset = pickle.load(file)
	return dataset
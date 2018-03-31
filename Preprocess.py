# Generates training and testing datasets as text files from dataset.txt

# For shuffling dataset.
from random import shuffle

# For processing data.
import re # Regular expressions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# For reading / writing python objects
import os.path
import pickle

def GenerateDataSets (datafile, trainfile, testfile) :
	f = open(datafile)
	lines = f.read().split('\n')[:-1]
	shuffle(lines)
	trn = int( 0.8 * len(lines) )
	with open(trainfile, 'w') as trainingset :
		trainingset = open(trainfile, 'w')
		for line in lines[0:trn] :
			trainingset.write(line)
			trainingset.write('\n')
	with open(testfile, 'w') as testingset :
		for line in lines[trn:] :
			testingset.write(line)
			testingset.write('\n')

def ProcessFile (filename) :
	f = open(filename, 'r')
	delims = '[ \t\n.,;:?\'\"\d()\\-+!@#$%^&*()<>/\\[\\]|]+'
	stop_words = set(stopwords.words('english'))
	porter_stemmer = PorterStemmer()
	lines = [ line.split('\t') for line in f.read().split('\n')[:-1] ]
	# Remove stop words and apply Porter stemming on the rest
	for line in lines : 
		line[0] = line[0] == 'spam'
		line[1] = [porter_stemmer.stem(word) for word in
			filter(None, re.split(delims, line[1]))
			if word not in stop_words]
	return lines

def PreprocessData (trainfile, testfile, dictfile, trvec, tsvec) :
	processed_training_data = ProcessFile(trainfile)
	processed_testing_data = ProcessFile(testfile)
	dictionary = {}
	for line in processed_training_data :
		for word in line[1] :
			dictionary[word] = {0}
	for line in processed_testing_data :
		for word in line[1] :
			dictionary[word] = {0}
	dictsize = 0
	for word in dictionary.keys() :
		dictionary[word] = dictsize
		dictsize += 1

	# Generate training vectors and write them to file
	with open(trvec, 'wb') as training_datafile :
		vectors = []
		for line in processed_training_data :
			# Generate 1-hot vector and write to file.
			vec = [0] * dictsize
			for word in line[1] :
				vec[ dictionary[word] ] = 1
			vectors.append( (line[0], vec) )
		pickle.dump(vectors, training_datafile,
				pickle.HIGHEST_PROTOCOL)

	# Generate testing vectors and write them to file
	with open(tsvec, 'wb') as testing_datafile :
		vectors = []
		for line in processed_testing_data :
			# Generate 1-hot vector and write to file.
			vec = [0] * dictsize
			for word in line[1] :
				vec[ dictionary[word] ] = 1
			vectors.append( (line[0], vec) )
		pickle.dump(vectors, testing_datafile,
			pickle.HIGHEST_PROTOCOL)

	# Write dictionary to file
	with open(dictfile, 'wb') as dictionary_file :
		pickle.dump(dictionary, dictionary_file,
				pickle.HIGHEST_PROTOCOL)

	# Return dictionary
	return dictionary

if __name__ == '__main__' :
	if(not os.path.exists('Trainingset.txt')
	or not os.path.exists('Testingset.txt')):
		GenerateDataSets('Dataset.txt',
			'Trainingset.txt', 'Testingset.txt')
		PreprocessData('Trainingset.txt', 'Testingset.txt',
			'dict.pkl', 'trvec.pkl', 'tsvec.pkl')

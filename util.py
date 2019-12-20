"""
Util file for SVM, Random Forest and KNN
Author: Jack St Clair
"""

import numpy as np
import optparse
import pickle
import random

def parse_args():
    """Parse command line arguments (models)."""
    parser = optparse.OptionParser(description='parse the command line\
     arguments')
    parser.add_option('-m', '--model', type='string', help='the kind' +\
        ' of model to use', default='KNN')
    (opts, args) = parser.parse_args()

    mandatories = ['model']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts
'''
def data_preprocess():
    dataset = open("aclImdb/train/labeledBow.feat", "r")
    #data = np.zeros(89527,25000)
    #target = np.zeros(25000)
    data = np.array([[],[]])
    target = np.array([])
    line_num = 0
    for line in dataset:
        if line_num == 200:
            print(data)
            print(target)
        list = line.split()
        target = np.append(target, list[0])
        for num in range(1,len(list)):
            example_list = list[num].split(":")
            word_index = [example_list[0]]
            number = [example_list[1]]
            word = np.array([word_index, number])
            data = np.append(data, word, axis = 1)
        line_num += 1
    corpus = {"data":data, "target":target}
    return corpus
'''
def data_preprocess():
    """
    preprocess data into a dictionary structure for sklearn. Bins
    the data into two classes of good and be reviews.
    """
    #dataset = open("aclImdb/train/labeledBow.feat", "r")
    dataset = open("aclImdb/train/shuffledlabeledBow.feat", "r")
    data = np.zeros((5000,2000), dtype=int) #shape (n_samples, n_features)
    target = np.zeros(5000)
    line_num = 0
    for line in dataset:
        print(line_num)
        if line_num <= 4999:
            list = line.split()
            if int(list[0]) >= 7:
                target[line_num] = 1
            if int(list[0]) <= 4:
                target[line_num] = -1
            for num in range(1,len(list)): #tabbed forward with block below
                example_list = list[num].split(":") #list of form ['word','count']
                word_index = int(example_list[0])
                if word_index <= 1999:
                    number = int(example_list[1]) #added int
                    data[line_num, word_index] = number #swapped these see line 54 comment
        line_num += 1
    corpus = {"data":data, "target":target}
    file = open('train_data.pkl', 'wb')
    pickle.dump(corpus, file)
    file.close()
    return corpus

def data_preprocess_non_binary():
    """
    the same as the above function but this preprocessor keeps
    all the unique review scores instead of binning them.
    """
    #dataset = open("aclImdb/train/labeledBow.feat", "r")
    dataset = open("aclImdb/train/shuffledlabeledBow.feat", "r")
    data = np.zeros((5000,2000), dtype=int) #shape (n_samples, n_features)
    target = np.zeros(5000)
    line_num = 0
    for line in dataset:
        print(line_num)
        if line_num <= 4999:
            list = line.split()
            target[line_num] = int(list[0])
            for num in range(1,len(list)): #tabbed forward with block below
                example_list = list[num].split(":") #list of form ['word','count']
                word_index = int(example_list[0])
                if word_index <= 1999:
                    number = int(example_list[1]) #added int
                    data[line_num, word_index] = number #swapped these see line 54 comment
        line_num += 1
    corpus = {"data":data, "target":target}
    file = open('nb_train_data.pkl', 'wb')
    pickle.dump(corpus, file)
    file.close()
    return corpus

def test_data_preprocess():
    #dataset = open("aclImdb/train/labeledBow.feat", "r")
    dataset = open("aclImdb/test/shuffledlabeledBow.feat", "r")
    data = np.zeros((5000,2000), dtype=int) #shape (n_samples, n_features)
    target = np.zeros(5000)
    line_num = 0
    for line in dataset:
        print(line_num)
        if line_num <= 4999:
            list = line.split()
            if int(list[0]) >= 7:
                target[line_num] = 1
            if int(list[0]) <= 4:
                target[line_num] = -1
            for num in range(1,len(list)): #tabbed forward with block below
                example_list = list[num].split(":") #list of form ['word','count']
                word_index = int(example_list[0])
                if word_index <= 1999:
                    number = int(example_list[1]) #added int
                    data[line_num, word_index] = number #swapped these see line 54 comment
        line_num += 1
    corpus = {"data":data, "target":target}
    file = open('test_data.pkl', 'wb')
    pickle.dump(corpus, file)
    file.close()
    return corpus

def test_data_preprocess_non_binary():
    #dataset = open("aclImdb/train/labeledBow.feat", "r")
    dataset = open("aclImdb/test/shuffledlabeledBow.feat", "r")
    data = np.zeros((5000,2000), dtype=int) #shape (n_samples, n_features)
    target = np.zeros(5000)
    line_num = 0
    for line in dataset:
        print(line_num)
        if line_num <= 4999:
            list = line.split()
            target[line_num] = int(list[0])
            for num in range(1,len(list)): #tabbed forward with block below
                example_list = list[num].split(":") #list of form ['word','count']
                word_index = int(example_list[0])
                if word_index <= 1999:
                    number = int(example_list[1]) #added int
                    data[line_num, word_index] = number #swapped these see line 54 comment
        line_num += 1
    corpus = {"data":data, "target":target}
    file = open('nb_test_data.pkl', 'wb')
    pickle.dump(corpus, file)
    file.close()
    return corpus

def shuffle_lines():
    lines = open("aclImdb/train/labeledBow.feat").readlines()
    random.shuffle(lines)
    open('shuffledlabeledBow.feat', 'w').writelines(lines)

if __name__ == "__main__":
    data_preprocess()
    data_preprocess_non_binary()
    test_data_preprocess()
    test_data_preprocess_non_binary()

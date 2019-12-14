"""
Util file for SVM, Random Forest and KNN
Author: Jack St Clair
"""

import numpy as np
import optparse
import pickle

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
    dataset = open("aclImdb/train/labeledBow.feat", "r")
    data = np.zeros((89527,25000), dtype=int)
    target = np.zeros(25000)
    line_num = 0
    for line in dataset:
        print(line_num)
        list = line.split()
        if int(list[0]) >= 7:
            target[line_num] = 1
        if int(list[0]) <= 3:
            target[line_num] = 0
        for num in range(1,len(list)):
            example_list = list[num].split(":")
            word_index = int(example_list[0])
            number = example_list[1]
            data[word_index, line_num] = number
        line_num += 1
    corpus = {"data":data, "target":target}
    file = open('train_data.pkl', 'wb')
    pickle.dump(corpus, file)
    file.close()
    return corpus

data_preprocess()

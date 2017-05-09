import csv
import numpy as np

# Function: toInt
# Tranform each elements of array to int type
# Return: matrix
def toInt(array):
    matrix = np.mat(array)
    m, n = np.shape(matrix)
    newMat = np.zeros( (m, n) )

    for i in xrange(m):
        for j in xrange(n):
            newMat[i, j] = int(matrix[i, j])

    return newMat

# Function: loadTrainData
# load training data from csv format file
# Return: list
def loadTrainData(trainDataFilePath):
    trainData = []

    with open(trainDataFilePath) as file:
        lines = csv.reader(file)
        for line in lines:
            trainData.append(line)

    del trainData[0]
    trainData = np.array(trainData)
    label = trainData[:, 0]
    feature_train = trainData[:, 1:]
    return feature_train, label

if __name__ == '__main__':
    feature_train, label = loadTrainData('C:\\computer_science_y\\version_control\\Kaggle\\Digit_Recognizer\\Data\\test.csv')
    feature_train = toInt(feature_train)
    label = toInt(label)

    print feature_train[:2]
    print "\n"
    print label[:5]



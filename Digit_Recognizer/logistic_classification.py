import pandas as pd
import numpy as np

# get data
test  = pd.read_csv('/Users/huayongpan/Documents/Kaggle_solution/Kaggle/Digit_Recognizer/Data/test.csv')
train = pd.read_csv('/Users/huayongpan/Documents/Kaggle_solution/Kaggle/Digit_Recognizer/Data/train.csv')

# print the structure of data
print(train.head())

# plot the distribution of the samples
# import matplotlib.pyplot as plt
# plt.hist(train["label"])
# plt.title("Frequency Histogram of Numbers in Training Data")
# plt.xlabel("Number Value")
# plt.ylabel("Frequency")

# Plot the sample as matrix
# Show it as digit
# import math
# # plot the first 25 digits in the training set.
# f, ax = plt.subplots(5, 5)
# # plot some 4s as an example
# for i in range(1,26):
#     # Create a 1024x1024x3 array of 8 bit unsigned integers
#     data = train.iloc[i,1:785].values #this is the first number
#     nrows, ncols = 28, 28
#     grid = data.reshape((nrows, ncols))
#     n=math.ceil(i/5)-1
#     m=[0,1,2,3,4]*5
#     ax[m[i-1], n].imshow(grid)

# PCA normalizes the data
from sklearn import decomposition
from sklearn import datasets

## PCA decomposition
# And show the 200 more important components
# pca = decomposition.PCA(n_components=200) #Finds first 200 PCs
# pca.fit(train.drop('label', axis=1))
# plt.plot(pca.explained_variance_ratio_)
# plt.ylabel('% of variance explained')
#plot reaches asymptote at around 50, which is optimal number of PCs to use.

## PCA decomposition with optimal number of PCs
#decompose train data
# Here keep the 50 more important components
pca = decomposition.PCA(n_components=50) #use first 3 PCs (update to 100 later)
pca.fit(train.drop('label', axis=1))
PCtrain = pd.DataFrame(pca.transform(train.drop('label', axis=1)))
PCtrain['label'] = train['label']

#decompose test data
#pca.fit(test)
PCtest = pd.DataFrame(pca.transform(test))


# Show the training data on the 3 most important components axies
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax = fig.add_subplot(111)
#
# x =PCtrain[0]
# y =PCtrain[1]
# z =PCtrain[2]
#
# colors = [int(i % 9) for i in PCtrain['label']]
# ax.scatter(x, y, z, c=colors, marker='o', label=colors)
#
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
#
# plt.show()

from sklearn.neural_network import MLPClassifier
y = PCtrain['label'][0:20000]
X=PCtrain.drop('label', axis=1)[0:20000]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10,), random_state=1)
clf.fit(X, y)

from sklearn import  metrics
#accuracy and confusion matrix
predicted = clf.predict(PCtrain.drop('label', axis=1)[20001:420000])
expected = PCtrain['label'][20001:42000]

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

output = pd.DataFrame(clf.predict(PCtest), columns =['Label'])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)

output['ImageId']=output['ImageId']+1

output.to_csv('/Users/huayongpan/Documents/Kaggle_solution/Kaggle/Digit_Recognizer/Data/output.csv', index=False)

# from csv format file
# # Return: list
# def loadTrainData(trainDataFilePath):
#     trainData = []
#
#     with open(trainDataFilePath) as file:
#         lines = csv.reader(file)
#         for line in lines:
#             trainData.append(line)
#
#     del trainData[0]
#     trainData = np.array(trainData)
#     label = trainData[:, 0]
#     feature_train = trainData[:, 1:]
#     return feature_train, label
#
# if __name__ == '__main__':
#     feature_train, label = loadTrainData('C:\\computer_science_y\\version_control\\Kaggle\\Digit_Recognizer\\Data\\test.csv')
#     feature_train = toInt(feature_train)
#     label = toInt(label)
#
#     print feature_train[:2]
#     print "\n"
#     print label[:5]

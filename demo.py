from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

names = [
	"Nearest Neighbors",
	"Naive Bayes",
	"Neural Net"
]

clfs = [
	KNeighborsClassifier(3),
	GaussianNB(),
	MLPClassifier(alpha=1)]

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

i = 0
# iterate over classifiers
for clf in clfs:
	clf = clf.fit(X, Y)
	prediction = clf.predict([[190, 70, 43]])
	print("the predicton for %s is %s"%(names[i],prediction))
	i = i + 1

from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt 
from keras.layers import Dense
from keras import Model,Input
def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A
def pred(W, X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis = 0)


def display(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    # plt.show()


class SoftmaxReg:
  def __init__(self):
    return None
  def build(self, in_dim):
    input = Input(in_dim) 
    reluLayer = Dense(3,activation='relu')(input)
    output= Dense(3,use_bias=True,activation = 'sigmoid')(reluLayer)
    self.model = Model(input,output)
    return self.model
  def train(self,x_train,y_train):
    self.model.compile(optimizer="SGD",loss="sparse_categorical_crossentropy")
    hist =self.model.fit(x_train,y_train,epochs=2000)
    return hist
  def save(self,model_path):
    self.model.save(model_path)
  def load(self,model_path):
    self.models.load_model(model_path)
  def predict(self,x_test):
    pred = self.model.predict(x_test)
    return np.argmax(pred, axis=1)

  def summary(self):
    return self.model.summary()
  def get_trained_params(self):
    print('number of layer',len(self.model.layers))
    return self.model.layers[1].get_weights()

#step 1: generate data and visualize
means = [[2, 2], [-2, 3],[-2,-3]]
cov = [[1, 0], [0, 1]] 
N = 20
np.random.seed(20520052)

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1,X2), axis = 0)
original_label = np.asarray([0]*N + [1]*N+[2]*N).T
# print(original_label)
# print(X)
display(X,original_label)


# print('X_train',x_train)


#step 2: Create and build model
SoftmaxModel = SoftmaxReg()
SoftmaxModel.build(2)
SoftmaxModel.summary()

#step 3: training model
hist=SoftmaxModel.train(X,original_label)
#step 4: predict model and visualize
params= SoftmaxModel.get_trained_params()
print(params)

b = params[1][0]
w = params[0]


xm = np.arange(np.min[:,0], np.max[:,0], 0.025)
xlen = len(xm)
ym = np.arange(np.min[:,1], np.max[:,1], 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

XX = np.concatenate((xx1, yy1), axis = 0).T

z = SoftmaxModel.predict(XX)
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, 200, cmap='jet', alpha = .1)
plt.show()
# Plot also the training points
# plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')


# plt.axis('equal')

# plt.show()
plt.plot(hist.history['loss'])
plt.show()
# #step 5: save model

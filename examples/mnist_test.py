''' 
Use all import (the asterisk - *) and import:

1. models
2. methods

to configure your neural network structure and get full control 
'''

from models import *
from methods import *


def get_mnist():
    # for importing the dataset
    from sklearn.datasets import fetch_openml
    from sklearn.utils import shuffle

    print('fetching...')
    mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    print("[mnist] is fetched.")

    x, t = mnist
    x = x / np.max(x) # normalization

    x, t = shuffle(x, t, random_state=42)
    x = x.reshape((len(x), 1, 28, 28)) # to 3D

    y = []
    for n in t:
        nodes = [0] * 10
        nodes[int(n)] = 1
        y.append(nodes)
    y = np.array(y, dtype=float)
    return x, y, t

x, y, t = get_mnist()
# neural network setups
cnn = nn(
    
        Conv(k_shape=(3, 3), n_kernels=4, activation=ReLU),
        MaxPool((2, 2)),
        Conv(k_shape=(3, 3), n_kernels=8, activation=ReLU),
        MaxPool((2, 2)),
        Flatten(),
        Dense(128, ReLU),
        Dense(10, softmax)
    
    )

# You can pass the additional arguments here.
cnn.compile(
        input_shape=(128, 1, 28, 28), 
        possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        cost= CCE,
        optimizer= Adam(lr=0.001)
)   

# split data
n = int(0.85 * len(x))
xtrain, ytrain, ttrain, xtest, ytest, ttest = x[:n], y[:n], t[:n], x[n:], y[n:], t[n:]  

print('training...')
acc1 = cnn.learn(xtrain, ytrain, ttrain, epochs=10)

print("testing...")
loss2, acc2 = cnn.test(xtest, ytest, ttest)

# single-input test
i = 0
while isinstance(i, int) and i < len(xtest):
    i = int(input(f'index({len(ttest) - 1}):'))
    cnn.feedforward(xtest[i].reshape(1, 1, 28, 28))
    print(cnn.output(), ttest[i])
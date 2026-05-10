# for neural network configuration
from models import *
from methods import *

# for plotting
import matplotlib.pyplot as plt
import numpy as np

# for importing the dataset
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

print('fetching...')
mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

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

print("[mnist] is fetched.")

# neural network setups
cnn = nn(
            Conv(k_shape=(3, 3), n_kernels=2, initializer_w=he_normal),
            MaxPool((2, 2)),
            Conv(k_shape=(3, 3), n_kernels=4, initializer_w=he_normal),
            MaxPool((2, 2)),
            Flatten(),
            Dense(128, ReLU, initializer_w=he_normal),
            # Dense(16, ReLU),
            Dense(10, softmax, initializer_w=he_normal),
            possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            cost= CCE,
            optimizer= Adam(lr=0.001)
            )

# data split
n = int(0.85 * len(x))
batch_size, epochs = 128, 10
xtrain, ytrain, ttrain, xtest, ytest, ttest = x[:n], y[:n], t[:n], x[n:], y[n:], t[n:]  

# train
print('training...')
loss1, acc1 = cnn.learn(xtrain, ytrain, ttrain, epochs=epochs, batch_size=batch_size)

# test
print("testing...")
loss2, acc2 = cnn.test(xtest, ytest, ttest, batch_size)

# plots
t_axis = [i for i in range(len(xtest) // batch_size)]
l_axis = [i for i in range(len(acc2))]

plt.axis([0,  epochs - 0.5, 0, 110])
plt.xticks(t_axis)
plt.plot(l_axis, acc2, color='red', label='testing_accuracy')
plt.plot(l_axis, loss2, color='orange', label='testing_loss', linestyle='--')

plt.xlabel("Batch")
plt.ylabel("Percentage")
plt.title("Model Learning")
plt.show()

# single-input test
i = 0
while isinstance(i, int) and i < len(xtest):
    i = int(input(f'index({len(ttest) - 1}):'))
    cnn.feedforward(xtest[i])
    print(cnn.output(), ttest[i])
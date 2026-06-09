# What is it?

A **neural network framework** implemented entirely from scratch. It reconstructs fundamental deep learning components, including _convolution kernels, pooling operations, dense-to-convolution auto transitions, channel handling, optimizers, activation functions, and parameter initialization strategies_.  

The framework emphasizes **modularity, transparency, and user control**, enabling direct inspection and control of the learning mechanics.  

# Why?

To develop a first-principles understanding of neural network behavior rather than relying on high-level libraries.  
Re-implementing core mechanisms provides deeper insight into learning dynamics, gradient flow, and architectural design trade-offs.  

# Setup

**Requirements**:  
- C++20  
- CMake  
- pybind11
- Eigen 5.0.0
- numpy
- matplotlib

Then, type:  
```bash
cmake -S src -B build
cmake --build build
```
This will create the necessary `.pyd` file.  

After that, run:  
```
python -m __init__
```
To check that everything is working.  

# Usage

Import the modules:  

```python
from netjet.models import *
from netjet.methods import *
```  

Instantiate a neural network and define the architecture:  

```python
mlp = nn(
        Flatten(),
        Dense(16, ReLU),
        Dense(16, ReLU),
        Dense(10, softmax)
)
```

compile the model:  
```python
mlp.compile(
        input_shape=(128, 1, 28, 28), # (batch_size, channels, height, width)
        possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        cost= CCE,
        optimizer= Adam(lr=0.001)
)
```
you would see something like this:
```bash
(25%) : fused Flatten (0) --> Input Node.
(50%) : fused Dense (1) --> Flatten (0).
...
(100%) : fused ...
Adam optimization is configured.
(*) The network is compiled successfully.       (0.01s)
```
Train the model:  
```python
loss1, acc1 = mlp.learn(xtrain, ytrain, ttrain, epochs=8)
```
```bash
(#) accuracy per epoch (1): 78.66%       (2.03s)
(#) accuracy per epoch (2): 92.11%       (2.07s)
...
(#) accuracy per epoch (...): ...       (2.12s)
(*) Training is complete.       (16.53s)
```

Also you can inspect your training by setting `Debug_plot=True` in `learn`:  
<img width="930" height="376" alt="Python 3 11 6_9_2026 2_20_54 PM" src="https://github.com/user-attachments/assets/77c9f3e5-2719-4b5d-8f3b-de65018fa011" />

> [!NOTE]
> Enabling `Debug_plot` can slow learning noticably, so use it wisely to refine your model.

Evaluate on test data:  
```python
loss2, acc2 = mlp.test(xtest, ytest, ttest)
```
> [!NOTE]
> * The above structure achieved `~94-95%` testing accuarcy on MNIST.  
> * `Dense`, `Flatten`, and `Reshape` layers are relatively fast.  

To use convolutional layers:  
```python
cnn = nn(
            Conv(k_shape=(3, 3), n_kernels=2, activation=ReLU),
            MaxPool((2, 2)),
            Conv(k_shape=(3, 3), n_kernels=4, activation=ReLU),
            MaxPool((2, 2)),
            Flatten(),
            Dense(16, ReLU),
            Dense(10, softmax)
)

```  
> [!NOTE]
> * With `8 epochs`, a `batch size` of `128` and `Adam`, The above structure achieved `~96-97%` testing accuarcy on MNIST (~15s/epoch).  
> * Convolution and pooling operations are now moderately faster.
---

For more examples, you may experiment with `mnist_test.py` in `examples` and run it using:  
```bash
python -m examples.mnist_test
```

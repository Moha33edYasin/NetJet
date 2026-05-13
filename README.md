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

Then, type:  
```bash
cmake -S src -B build
cmake --build build
```
This will create the necessary `.pyd` file.  

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
        Dense(10, softmax),
        possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        cost= CCE, # cross categorical entropy
        optimizer= Adam(lr=0.001)
)
```  

Train the model:  
```python
loss1, acc1 = mlp.learn(xtrain, ytrain, ttrain, epochs=8, batch_size=128)
```  

Evaluate on test data:  
```python
loss2, acc2 = mlp.test(xtest, ytest, ttest, batch_size=128)
```  
> [!NOTE]
> * The above structure achieved `~0.94-0.95` testing accuarcy on MNIST.  
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
            Dense(10, softmax),
            possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            cost= CCE,
            optimizer= Adam(lr=0.001)
)
```  
> [!NOTE]
> With `8 epochs` and a `batch size` of `128`, The above structure achieved `~0.97-0.97.5` testing accuarcy on MNIST.  

> [!WARNING]
> Convolution and pooling operations are not yet fully optimized and may run slower than usual.  

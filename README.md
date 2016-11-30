# Bidu: Simplest Deep Learning library for TensorFlow as backend

###Long short description:

Bidu is a high-level neural networks library, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow).
It was developed with a focus on enabling fast experimentation.
Bidu aims to do a simpler approach, create a subset of new apis that connects directly to tensorflow.
Since we created based on top of [keras](https://github.com/fchollet/keras), which the intent is to create a much better api for many deep learning frameworks, bidu is just a simplification of keras, but using a more advanced orcherstration of ingestion of data.

Since Bidu is focused on TensorFlow, we want to add the support for [Apache Spark](http://spark.apache.org)

*Being able to go from idea to result with the least possible delay is key to doing good research.*

Use Bidu if you need a deep learning library that:

- Allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.
- Supports arbitrary connectivity schemes (including multi-input and multi-output training).
- Runs seamlessly on CPU and GPU.


Bidu is compatible with: __Python 2.7-3.5__.


------------------


## Guiding principles

- __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

- __Minimalism.__ Each module should be kept short and simple. Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate.

- __Easy extensibility.__ New modules are dead simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making bidu suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.


------------------


## Getting started: 30 seconds to bidu

The core data structure of bidu is a __model__, a way to organize layers. The main type of model is the [`Sequential`] model, a linear stack of layers.

Here's the `Sequential` model:

```python
from bidu.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from bidu.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```

Once your model looks good, configure its learning process with `.compile()`:
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of bidu is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
```python
from bidu.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:
```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:
```python
model.train_on_batch(X_batch, Y_batch)
```

Evaluate your performance in one line:
```python
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
```

Or generate predictions on new data:
```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

Building a question answering system, an image classification model, a Neural Turing Machine, a word2vec embedder or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?


------------------


## Installation

bidu uses the following dependencies:

- numpy, scipy
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.


*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).


To install bidu, `cd` to the bidu folder and run the install command:
```sh
sudo python setup.py install
```

## credits
Bidu is a fork of [Keras](https://github.com/fchollet/keras) which does connect directly to Tensorflow and theranos.

# RockTheBoar

Code written as part of the [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge). The overal goal of this competition was to design an algorithm to accurately mask cars in images. 

## [TensorFlow implementation](https://github.com/nesar/RockTheBoar/tree/master/tensorflow_gan)

This network was used for our final submission. The architecture is a 12 layer autoencoder-style fully convolutional neural network - 6 convolutional layers followed by 6 deconvolutional layers. Dice co-effecient was used as a accuracy metric, and Adaptive Moment Estimation (Adam) is used for gradient descent optimization. 

Functions of some of the key scripts:

* _convnet_architecture_12l.py_: The network architecture, written in tensorflow. 
* _train_network.py_: The training framework.
* _input_pipeline.py_: Image loading functions. 
* _infer_function.py_: Loads the trained model for inference on test images.
* _filter_csv.py_: Post-processing pipeline with tuned filtering, smoothing and tuned filtering.

The network is trained on images of cars, with corresponding masks as targets. The fully trained network provides a pixel-wise "probability of car" estimate. 

## [Keras implementation](https://github.com/nesar/RockTheBoar/tree/master/keras_implementation)

Initial Prototyping for a simple convolutional network of 3 layers is implemented. The input/output pipelines were forked from [Bruno G. do Amaral](https://www.kaggle.com/bguberfain)'s Naive Keras model. Network architecture can be edited in _simple_model.py_.

## Authors
Donald Lee-Brown([@dleebrown](https://github.com/dleebrown)), Sinan Deger([@sinandeger](https://github.com/sinandeger)) and Nesar Ramachandra ([@nesar](https://github.com/sinandeger))

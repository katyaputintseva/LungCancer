1. Reproducing Hammack's CNN model

Optimizer: Nadam (lr=0.01) 
Loss function: Binary crossentropy
Objectives: Malignancy, False positiveness
Branching: dense block

False positive samples: None
Train/Test split: 75/25
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.1)
Number of neurons in the Dense layer: 32
Output layer activation function: sigmoid
Regularization degree: 0
Dropout rate: 0
Dropout type: Gaussian Dropout

Number of epochs: 30
Number of samples per batch: 20
Number of batches per epoch: 150

----

2.png 		

Optimizer: Nadam (lr=0.01)
Loss function: MSE
Objectives: Malignancy
Branching: dense block

False positive samples: None
Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.1)
Number of neurons in the Dense layer: 32
Output layer activation function: sigmoid
Regularization degree: 0
Dropout rate: 0
Dropout type: Gaussian Dropout

Number of epochs: 15 or 25
Number of samples per batch: 20
Number of batches per epoch: 150

----

3.png Change of the loss function to MSE

Optimizer: Nadam (lr=0.01)
Loss function: MSE
Objectives: Malignancy
Branching: dense block

False positive samples: None
Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: ReLU
Number of neurons in the Dense layer: 32
Output layer activation function: sigmoid
Regularization degree: 0
Dropout rate: 0
Dropout type: Gaussian Dropout

Number of epochs: 15 or 25
Number of samples per batch: 20
Number of batches per epoch: 150

----

4.png Addition of False Positives + adding regularization + adding other objectives

Optimizer: Nadam (lr=0.01)
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: dense block

False positive samples: True
Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: ReLU
Number of neurons in the Dense layer: 32
Output layer activation function: sigmoid
Regularization degree: 1e-3
Dropout rate: 1e-3
Dropout type: Gaussian Dropout

Number of epochs: 50
Number of samples per batch: 20
Number of batches per epoch: 150

----

5.png Increasing number of samples per batch/epoch + decreasing droupout and regularization

Optimizer: Nadam (lr=0.01)
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: dense block

False positive samples: True
Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: ReLU
Number of neurons in the Dense layer: 32
Output layer activation function: sigmoid
Regularization degree: 1e-4
Dropout rate: 1e-4
Dropout type: Gaussian Dropout

Number of epochs: 50
Number of samples per batch: 50
Number of batches per epoch: 150

----

6. Changing optimizer + tuning regularization degree and dropout rates + trying other activation functions and simple dropout

Optimizer: sgd+nesterov (lr=0.01)
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: dense block

False positive samples: None
Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-48-64-65
Number of neurons in the Dense layer: 32

6.1.png			

Regularization degree: 1e-4
Dropout rate: 1e-4
Dropout type: Gaussian Dropout
Number of epochs: 50
Number of samples per batch: 50
Number of batches per epoch: 150
Inner activation functions: ReLU
Output layer activation function: sigmoid

6.2.png			

Regularization degree: 5e-4
Dropout rate: 5e-4
Dropout type: Gaussian Dropout
Number of epochs: 15
Number of samples per batch: 50
Number of batches per epoch: 150
Inner activation functions: ReLU
Output layer activation function: sigmoid

6.3.png

Regularization degree: 1e-3
Dropout rate: 1e-3
Dropout type: Regular Dropout
Number of epochs: 15
Number of samples per batch: 50
Number of batches per epoch: 100
Inner activation functions: LeakyReLU(0.3)
Output layer activation function: softmax

----

7.png Learning rate scheduler + Deep branching

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: conv block #5

Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-48-64-65
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 1e-3
Dropout type: Regular Dropout

Number of epochs: 15
Number of samples per batch: 50
Number of batches per epoch: 100

7.1.png

False positive samples: None

7.2.png

False positive samples: 50% of negative samples

7.3.png 2 stages + early stopping (1e-3 after 4 epochs)

2 training stages:
1: Number of epochs: 15
Number of samples per batch: 50
Number of batches per epoch: 110
False positive samples: None

2: Number of epochs: 15
Number of samples per batch: 50
Number of batches per epoch: 150
False positive samples: 50% of negative samples


----

8.png Data augmentation

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: conv block #5

Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-48-64-65
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 1e-3
Dropout type: Regular Dropout

8.1.png

False positive samples: None
Number of epochs: 15
Number of samples per batch: 50
Number of batches per epoch: 110

8.2.png

False positive samples: 50% of negative samples
Number of epochs: 20
Number of samples per batch: 50
Number of batches per epoch: 300

8.3.png

False positive samples: None
Number of epochs: 30
Number of samples per batch: 50
Number of batches per epoch: 110

8.4.png

False positive samples: None
Number of epochs: 30
Number of samples per batch: 50
Number of batches per epoch: 110
Dropout rate: 1e-2 

8.5.png

False positive samples: 50% of negative samples
Number of epochs: 30
Number of samples per batch: 50
Number of batches per epoch: 110
Dropout rate: 1e-2 
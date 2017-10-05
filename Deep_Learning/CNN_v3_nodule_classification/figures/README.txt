1. Building a nodule/non-nodule classifier

1.1.png

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: Binary crossentropy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 1e-2
Dropout type: Regular Dropout

Number of epochs: 20
Number of samples per batch: 50
Number of batches per epoch: 110

Early Stopping: 1e-3 after 4 epochs


1.2.png

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: Binary crossentropy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.1
Dropout type: Regular Dropout

Number of epochs: 30
Number of samples per batch: 50
Number of batches per epoch: 110

Early Stopping: 1e-3 after 4 epochs


1.3.png

Optimizer: Adam (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: Binary crossentropy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.1
Dropout type: Regular Dropout

Number of epochs: 20
Number of samples per batch: 30
Number of batches per epoch: 50

Early Stopping: 1e-3 after 4 epochs


1.4.png

Optimizer: Adam (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: Binary crossentropy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.2
Dropout type: Regular Dropout

Number of epochs: 15
Number of samples per batch: 30
Number of batches per epoch: 50

Early Stopping: 1e-3 after 4 epochs


1.5.png

Optimizer: Nadam (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: Binary crossentropy
Accuracy: Categorical accuracy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.3
Dropout type: Regular Dropout

Number of epochs: 10
Number of samples per batch: 30
Number of batches per epoch: 50

Early Stopping: 1e-3 after 4 epochs


1.6.png

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: Categorical crossentropy
Accuracy: Categorical accuracy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 5e-3
Dropout rate: 0.3
Dropout type: Regular Dropout

Number of epochs: 30
Number of samples per batch: 30
Number of batches per epoch: 100

Early Stopping: 1e-3 after 4 epochs


1.7.png

Optimizer: Nadam()
Loss function: Categorical crossentropy
Accuracy: Categorical accuracy
Objectives: nodule/non-nodule

False positive samples: None
Train/Test split: 80/20
Data augmentation: None

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 10
Output layer activation function: softmax
Regularization degree: 1e-2
Dropout rate: 0.5
Dropout type: Regular Dropout

Number of epochs: 10
Number of samples per batch: 30
Number of batches per epoch: 100

Early Stopping: None

----------

2. Building feature regressor

2.1.png		SGD

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: conv block #5

False positive samples: None
Negative:Positive ratio: 0.5:0.5
Validation: Processed with Classifier, only nodules used
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.2
Dropout type: Regular Dropout

Number of epochs: 10
Number of samples per batch: 30
Number of batches per epoch: 50

Early Stopping: 1e-3 after 4 epochs


2.2.png		Nadam

Optimizer: Nadam (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: conv block #5

False positive samples: None
Negative:Positive ratio: 0.5:0.5
Validation: Processed with Classifier, only nodules used
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.2
Dropout type: Regular Dropout

Number of epochs: 10
Number of samples per batch: 30
Number of batches per epoch: 50

Early Stopping: 1e-3 after 4 epochs


2.3.png		FP addition

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: conv block #5

False positive samples: 50% of negative samples
Negative:Positive ratio: 0.3:0.7
Validation: Processed with Classifier, only nodules used
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.2
Dropout type: Regular Dropout

Number of epochs: 10
Number of samples per batch: 30
Number of batches per epoch: 50

Early Stopping: 1e-3 after 4 epochs


2.4.png		Long training

Optimizer: sgd+nesterov (lr scheduler { <2:1e-2, <5:1e-3, <10:5e-4, else:5e-5 })
Loss function: MSE
Objectives: Malignancy, Lobulation, Spiculation, Diameter
Branching: conv block #5

False positive samples: 50% of negative samples
Negative:Positive ratio: 0.4:0.6
Validation: Processed with Classifier, only nodules used
Data augmentation: Flips around random axes (implemented for positive samples only)

Filter size: (3,3,3)
Number of filters per conv block: 8-24-64-72-72
Inner activation functions: LeakyReLU(0.3)
Number of neurons in the Dense layer: 32
Output layer activation function: softmax
Regularization degree: 1e-3
Dropout rate: 0.3
Dropout type: Regular Dropout

Number of epochs: 30
Number of samples per batch: 60
Number of batches per epoch: 110

Early Stopping: 1e-3 after 4 epochs
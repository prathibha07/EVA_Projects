## Architectural Basics

1. How many layers :
The number of layers depends on the complexity of the network  and what we hope to achieve through it. 
In general it is preferred to stop going deep at the layer where the recetive field is equal to the size of the image(sometimes object) 

2. MaxPooling :
Reduces dimensionality of the input by 50% and leads to a loss of resolutionmof the image by 25%. It helps in reducing the number of layers and doesn't add to the number of parameters, thus reducing computational cost 

3. 1x1 Convolutions :
We use the 1x1 convolutional filters to reduce the filter dimension.With more number of filters, computations increases drastically even with small filter size. Thus 1x1 serves to combine similar features together and reduce the number of parameters. These 1x1 conv layers can also be used to increase the number of filters based on the requirement of the network/architechture.  

4. 3x3 Convolutions :
3X3 popular and easy convolution method. GPUs are built around this computaton.

5. Receptive Field :
It refers to the region of the input field which the CNN is looking at.  It acts as a window to see the image in the previous layer. Though all the pixels may not contain the information, the field/space serves to map the features the CNN will be viewing. It is important to have a receptive field equal to the size of the object in a image (or image) to help the network perform better. 
Example : Consider an image of a dog, if the network is only deep enough to see(receptive field) the eyes of the dog, it won't be able to predict the entire image of the dog(as the network hasn't seen the entire dog) leading to drop in accuracy. 

6. SoftMax :
Softmax is like probability, helps in picking a clear winner among the pixels. This works for the usual classification problems except for low error tolerance cases like in the medical domain.  

7. Learning Rate :
Learning rate helps to adjust th weights of the network with respect to loss gradient. Typically learnig rate is assigned randomly and after its impact on the accuracy , a best value is chosen after a few trials. 

8. Kernels and how do we decide the number of kernels?
Kernel is a window which contains some information (say in terms of matrix) and slides over the Input to give a resultant output.  This filter moves over the entire image, and may contain overlapping elements while the image is kept constant. 
Number of kernels is based on the number of classes and how many images we have per class. If the dataset has images with many features or many classes, number of kernels increases. 

9. Batch Normalization :
Helps to regularise the network. We normalize the input layer by adjusting and scaling the activations. For example, when we have features from 0 to 1 and some from 1 to 1000, we should normalize them to speed up learning. To increase the stability of a neural network, batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

10. Image Normalization :
Image normalisation is changing the pixel value of each pixel in the image, leading to change in the data and features. 

11. Position of MaxPooling : 
Since MaxPooling leads to loss of information, it is preferred to use at 3 to 5 layers after the first convolutional layer, where the information regarding edges and gradient is bound to be available in a large network. Performing MaxPooling before edges or gradients leads to drop in accuracy of the network. 

12. Concept of Transition Layers :

13. Position of Transition Layer :

14. Number of Epochs and when to increase them :
Number of epochs can be increased when accuracy of the network should be increased. But increasing can also lead to overfitting. 
Larger number of epochs  : Increases the computation time and sometime lead to accuracy drop of the model.

15. DropOut :
Randomly dropping out nodes during training. These units are not considered during a particular forward or backward pass. At each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p. 

16. When do we introduce DropOut, or when do we know we have some overfitting :
DropOuts helps to prevent over-fitting. We know our network is overfitting when it is performing well on the training data but not on the testing data. 

17. The distance of MaxPooling from Prediction : 
MP should be atleast a few layers above the prediction layer. MP results in loss of information , which may work well at the initial few layers( as we might only want to concentrate on the important features needed) but during the last layer information loss leads to lowere prediction whcih hurts the network. It is preferred for the network to be able to make a decsion about the important features by itself rather than removing certain information which might or might not be necessary.

18. The distance of Batch Normalization from Prediction :
Batch Normalisation normalises all the images in that particular batch. BN helps to standardise the input images with mean 0 and SD 1. It shoudln't be added just before the prediction layer as it is better for the prdiction layer if it is able to see the important information in the layer rather than the normalised one. 

19. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered) : 
For a particular input size, we  stop convolutons after a few layers when we know the receptive field achieved is good enough for the last layer to understand the features from the input.  

20. How do we know our network is not going well, comparatively, very early
Overfitting

21. Batch Size, and effects of batch size
Batch size is 
Larger batch size :
Depending on the GPU the time to compute per epoch may decrease, needs batch normalisation to regularise.

22. When to add validation checks

23. LR schedule and concept behind it :m
Reduces the learning rate over time. This has the effect of quickly learning good weights early and fine tuning them later. One of the easy methods is to decrease the learning rate gradually based on the epoch. 

24. Adam vs SGD
Sigmoid takes a real value as input and outputs a value between 0 and 1. It’s non-linear and continuously differentiable. Sigmoid works "like" probability and works well for models where we have to predict the probability as an output.
Adam optimization algorithm is an extension to stochastic gradient descent and is used to update network weights iterative based in training data. A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.

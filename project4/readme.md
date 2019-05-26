## Architectural Basics


1. How many layers :
The number of layers depends on the complexity of the network  and what we hope to achieve through it. 
In general it is preferred to stop going deep at the layer where the recetive field is equal to the size of the image(sometimes object) 

2. MaxPooling :
Since MP leads to loss of information, it is preferred to use at 3to 5 layers after the first convolutional layer, where the information regarding edges and gradient is bound to be available in a large network. Performing MP before edges or gradients are seenleads to drop in accuracy of the network.  It helps in reducing the number of layers and doesn't add to the number of parameters. 

3. 1x1 Convolutions :
We use the 1x1 convolutional filters to reduce the filter dimension.With more number of filters, computations increases drastically even with small filter size. Thus 1x1 serves to combine similar features together and reduce the number of parameters. These 1x1 conv layers can also be used to increase the number of filters based on the requirement of the network/architechture.  

4. 3x3 Convolutions :
3X3 popular and easy conv method , GPUs are built around this computaton.

5. Receptive Field :
It refers to the region of the input field which the CNN is looking at.  It acts as a window to see the image in the previous layer. Though all the pixels may not contain the information, the field/space serves to map the features the CNN will be viewing. Usually the network tries to concentrate on the region at the center/middle of the field as it is expected to contain most of the information, and hence it is important to have a recpetive field equal to the size of the object in a image (or image )to help the network perform better. 
Example : Consider an image which is of a dog, if the network is only deep enough to see(receptive field) the eyes of the dog, it won't be able to predict the entire image of the dog(as the network hasn't seen the entire dog) leading to drop in accuracy. 

6. SoftMax,
Softmax is like probability, helps in picking a clear winner among the pixels. This works much better for general classification( of objects) rather than prediction of the object in an image( suspicious mass in a CT scan) 

7. Learning Rate :
Learning rate helps to adjust th weights of the network with respect to loss gradient. Typically learnig rate is assigned randomly and after its impact on the accuracy , a best value is chosen after a few trials. 

8. Kernels and how do we decide the number of kernels?
Kernel is a window which contains some information (say in terms of matrix) and slides over the Input to give a resultant output.  This filter moves over the entire image, and may contain overlapping elements while the image is kept constant. 
Number of kernels can depend on the number of inofrmation needed to be extracted from the network. Ex: for a color image the minimum number of channels/kernels needed is 3( one each for Red, Green, Blue)

9. Batch Normalization :
Helps to regularise the network 

10. Image Normalization :

11. Position of MaxPooling :

12. Concept of Transition Layers :

13. Position of Transition Layer :

14. Number of Epochs and when to increase them :
Number of epochs can be increased when accuracy of the network should be increased. But increasing can also lead to overfitting. 

15. Larger number of epochs  : 
Increases the computation time and sometime lead to accuracy drop of the model. 


16. DropOut :
When do we introduce DropOut, or when do we know we have some overfitting

17. The distance of MaxPooling from Prediction : 
MP should be atleast a few layers above the prediction layer. MP results in loss of information , which may work well at the initial few layers( as we might only want to concentrate on the important features needed) but during the last layer information loss leads to lowere prediction whcih hurts the network. It is preferred for the network to be able to make a decsion about the important features by itself rather than removing certain information which might or might not be necessary.

18. The distance of Batch Normalization from Prediction :
Batch Normalisation normalises all the images in that particular batch. BN helps to standardise the input images with mean 0 and SD 1. It shoudln't be added just before the prediction layer as it is better for the prdiction layer if it is able to see the importatn information in the layer rather than the normalised one. 

19. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
For a particular input size, we  stop convolutons after a few layers when we know the receptive field acheived is good enough for the last layer to understand the features from the input.  

20. How do we know our network is not going well, comparatively, very early
Overfitting

21. Batch Size, and effects of batch size


22. Larger batch size- more time to compute per epoch , needs batch normalisation to regularise things

23. When to add validation checks

24. LR schedule and concept behind it

25. Adam vs SGD

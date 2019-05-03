### Project_1
This project gives a brief introduction to various types of primary image processing steps in OpenCV. The projects focuses to help in understanding the role of kernels for pattern recognition in convolution. 

### Assignment 1A
Contains an ipynb notebook file demonstrating the primary image processing steps in OpenCV. 

### Assignment 1B

##### What are Channels and Kernels (according to EVA)?
>Filters/Kernels/Feature Extractor -  Kernel is a window which contains some information (say in terms of matrix) and slides over the Input to give a resultant output.  This filter moves over the entire image, and may contain overlapping elements while the image is kept constant. 
And what are Weights one may ask : Weights are the values of the kernel , making the output features as the weighted sums of the input features. Kernels help in convolution . The weights of the filter decide to help in selecting the features.

>Channels  - are containers for similar features. 
In commuication , whereas a channel is an independent passage/medium which allows the mutiple signlas to pass through a single mechanism. 
Whereas in conv. its defined to posses similar features.  
Ex : A streamed media format broadcast may have red, green, blue, alpha, left audio, and right audio.
In printing, four color standard channels are cyan, magenta, yellow, and black. 
Remote control toys have channels through which signal from the remote are sent through kernels to coordinate signals to wheels or accessories.

<image src="pictures/ckt.png" width="300" height="250" align="middle"> </image>


--------------------------------------

##### Why should we only (well mostly) use 3x3 Kernels?
> A stack of two 3 × 3 conv. has an effective receptive field of 5 × 5; three such layers have a 7 × 7 effective receptive field. Incorporating three non-linear rectification layers instead of a single one, makes the decision function more discriminative. 

>On the implementation level, 3 × 3 convolution is faster than 5 × 5/7*7 convolutions  (by "faster" I mean the number of computations needed considering a convolution is O(pqXY) for p× q kernels on X × Y images) 
Speed-wise (except maybe if we're using FFT convolutions) , it's hard to beat  3 × 3.  

> We can decrease the number of parameters used: 
Consider  5X5input convolved on a 3x3 kernel TWICE resulting in 1x1 . 
   ---Num of Paramters = 9+9 =18
where as 5X5input convolved on a 5x5 kernel ONCE resulting in 1x1 
---Num of Paramters = 25
-Thus, increasing number of layers and reducing the parameters for the network to learn ,helps build a better network with more non- linear features. 
 Also , due to lower number of prameters/ weights, it becomes computationally efficient. 

> Odd filters help to seperate datapoints easily (as they have a center line) 

![](pictures/conv.gif)

------------------------------------------

##### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
> 99 times 

> 199 -197-195-193-191-189-187-185-183-181-179-177-175-173-171-169-167-165-163-161-159-157 -155-153-151-149-147-145-143-141-139-137-135-133-131-129-127-125-123-121-119-117-115-113-111-109-107-105-103-101-99-97-95-93-91-89-87-85-83-81-79-77-75-73-71-69-67 -65-63-61-59-57 -55-53-51-49-47-45-43-41-39-37-35-33-31-29-27-25-23-21-19-17-15-13-11-9-7 -5-3-1


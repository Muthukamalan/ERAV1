# PART - I

## Backpropagration


![W5](https://github.com/Muthukamalan/ERAV1/assets/50898904/920e0f34-b7b2-4871-90b5-b38d1da353ab)
$$\frac{\partial E~Total}{\partial w_5} = (a~o_1 - t_1) * a~o_1 * (1-a~o_1) * a~h_1$$


![W6](https://github.com/Muthukamalan/ERAV1/assets/50898904/97a6a3dd-9b85-4404-b812-6ad7643be285)
$$\frac{\partial E~Total}{\partial w_6} =  (a~o_1 - t_1) * a~o_1 * (1-a~o_1) * a~h_2$$


![W7](https://github.com/Muthukamalan/ERAV1/assets/50898904/711b52c6-5a6c-492f-af17-b6039487eeb0)
$$\frac{\partial E~Total}{\partial w_7} =  (a~o_2 - t_2) * a~o_2 * (1-a~o_2) * a~h_1$$


![W8](https://github.com/Muthukamalan/ERAV1/assets/50898904/13c59b6b-01e3-4889-a3cc-89e3c7458969)
$$\frac{\partial E~Total}{\partial w_8} =  (a~o_2 - t_2) * a~o_2 * (1-a~o_2) * a~h_2$$



![a_h1](https://github.com/Muthukamalan/ERAV1/assets/50898904/9b868b6a-b221-4b1a-bc54-ee1dfae2db22)
$$\frac{\partial E~Total}{\partial a~h_1} =  ((a~o_1 - t_1) * a~o_1 * (1-a~o_1) * w_5)+ ((a~o_2 - t_2) * a~o_2 * (1-a~o_2) * w_7)$$


![a_h2](https://github.com/Muthukamalan/ERAV1/assets/50898904/98cc8e26-5073-4c45-a9ba-cf025f2d6b14)
$$\frac{\partial E~Total}{\partial a~h_2} =  ((a~o_1 - t_1) * a~o_1 * (1-a~o_1) * w_6)+ ((a~o_2 - t_2) * a~o_2 * (1-a~o_2) * w_8)$$


![W1](https://github.com/Muthukamalan/ERAV1/assets/50898904/7855d999-0059-4e34-83fb-eda46138ffa0)
$$\frac{\partial E~Total}{\partial w_1} =  ((a~o_1 - t_1) * a~o_1 * (1-a~o_1) * w_5)+ ((a~o_2 - t_2) * a~o_2 * (1-a~o_2) * w_7) *  a~h_1 (1-a~h_1) * i_1$$


![W2](https://github.com/Muthukamalan/ERAV1/assets/50898904/3c8e34f7-cb92-48cb-8e75-662e2470dfef)
$$\frac{\partial E~Total}{\partial w_2} =((a~o_1 - t_1) * a~o_1 * (1-a~o_1) * w_5)+ ((a~o_2 - t_2) * a~o_2 * (1-a~o_2) * w_7) *  a~h_1 (1-a~h_1) * i_2$$


![W3](https://github.com/Muthukamalan/ERAV1/assets/50898904/98b5e4b1-cbd3-46cd-bf21-b0a250189465)
$$\frac{\partial E~Total}{\partial w_3} = ((a~o_1 - t_1) * a~o_1 * (1-a~o_1) * w_6)+ ((a~o_2 - t_2) * a~o_2 * (1-a~o_2) * w_8) * a~h_2 * ( 1-a~h_2) * i_1$$


![W4](https://github.com/Muthukamalan/ERAV1/assets/50898904/e39aca8a-3c2c-4c05-8d89-9b179b87d05f)
$$\frac{\partial E~Total}{\partial w_4} = ((a~o_1 - t_1) * a~o_1 * (1-a~o_1) * w_6)+ ((a~o_2 - t_2) * a~o_2 * (1-a~o_2) * w_8) * a~h_2 * ( 1-a~h_2) * i_2$$




Learning Rate =1
![backprop eta=1](https://github.com/Muthukamalan/ERAV1/assets/50898904/87ac0263-6e7a-4da3-8dff-e4a89d29da9a)

Learning Rate =10
![backprop eta=10](https://github.com/Muthukamalan/ERAV1/assets/50898904/81cd1486-1420-4c32-ad35-b6250f3a9e44)


















# PART - II
![model_architecture](https://github.com/Muthukamalan/ERAV1/assets/50898904/60f1bbc2-ea09-41d2-8d75-fcf9ada36248)

Some important topics are:

1. How many layers,
    i)    INPUT LAYER                                             (28 * 28 * 1)
    ii)   Conv2d(in_channel = 1 , out_channels = 16, kernel=3 )   (28 * 28 * 16)
    iii)  Conv2d(in_channel = 16, out_channels = 32, kernel=3 )   (28 * 28 * 32)
    iv)   Conv2d(in_channel= 32 , out_channels = 16 , kernel=1)   (28 * 28 * 16)
    v)    MaxPool(kernel=2 , stride=2 )                           (14 * 14 * 16)
    vi)   Conv2d(in_channel=16 , out_channels=16 , kernel= 3)     (14 * 14 * 16)
    vii)  Conv2d(in_channel=16 , out_channels=64, kernel= 3)      (14 * 14 * 64)
    viii) Conv2d(in_channel=64 , out_channels=16 , kernel=1 )     (14 * 14 * 16)
    ix)   MaxPool(kernel= 2, stride=2 )                           (7  * 7  * 16)
    x)    Conv2d(in_channel=16 , out_channels=10 , kernel=1 )     (3  * 3  * 10)
    xi)   AvgPool(kernel=3  )                                     (1  * 1  * 10)
    xii)  OUTPUT LAYER




2. MaxPooling
-   that calculates the maximum value for patches of 2d matrix 
![maxpool](https://github.com/Muthukamalan/ERAV1/assets/50898904/6b1fc2eb-35d0-494a-a981-e01b503f68c2)


3. 1x1 Convolutions 
-   (1 x 1 x K) sized filter is applied over all feature maps( h * w *  *feature_map*) across input and results  (h* w* K)
![1*1*kernel](https://github.com/Muthukamalan/ERAV1/assets/50898904/76ba0ab5-3887-4ad3-a57f-91fbf9b3575a)   


4. 3x3 Convolutions
![3*3*kernel](https://github.com/Muthukamalan/ERAV1/assets/50898904/882c17d7-9722-426d-a55a-358afd5e5844)

5. Receptive Field
-   Number of Neurons inputs pixels seen by 1 pixel of current neuron

6. SoftMax
-    function that turns a vector of K real values into a vector of K real values that sum to 1s
$$\sigma(\vec{z}_{i})= \frac{e^{z_{i}}} { \sum_{j=1}^{K} e^{z_{j}} } $$

7. Learning Rate

8. Kernels and how do we decide the number of kernels?

9. Batch Normalization

10. Image Normalization

11. Position of MaxPooling

12. Concept of Transition Layers
-   once we capture details from particular block. we should shrink image resolution but at same time it should resolve issues like checker_board and computational problems. that's why transition layer helps

13. Position of Transition Layer
-   should play a role after we done with particular block  


14. DropOut
-   While training we randomly hide some neurons to make it more generalizable
-   kind of act as regularization
-   don't use at end of the layer
![dropout](https://github.com/Muthukamalan/ERAV1/assets/50898904/2cb1203a-e1aa-4f98-a01e-28c2d7f559e4)


15. When do we introduce DropOut, or when do we know we have some overfitting
![overfitting](https://github.com/Muthukamalan/ERAV1/assets/50898904/10149940-4e4d-44e9-8728-5e913e26ac65)


16. The distance of MaxPooling from Prediction
17. The distance of Batch Normalization from Prediction
-   The distance from prediction to MaxPool and BatchNorm is should be as far as possible

18. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)

19. How do we know our network is not going well, comparatively, very early

20. Batch Size, and Effects of batch size
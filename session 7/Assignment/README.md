# Assignment
The targets are:

99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
Less than or equal to 15 Epochs
Less than 8000 Parameters




# APPROACH

1. Picking Right Skeleton  (Net1,Net2,Net3,Net4,Net5)
2. Make it Lighter         (Net6)
3. ADD BATCHNORM           (Net7)
4. ADD Regularizer         (Net8,Net8_1)
5. Increase Capacity       
6. Augumentation           (Net9,Net9_3)
7. Optimizer               (Net_93+Adam)




# Solution

Log for FINAL MODEL:
```log
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 11, 24, 24]          --
|    └─Conv2d: 2-1                       [-1, 8, 26, 26]           72
|    └─BatchNorm2d: 2-2                  [-1, 8, 26, 26]           16
|    └─ReLU: 2-3                         [-1, 8, 26, 26]           --
|    └─Dropout2d: 2-4                    [-1, 8, 26, 26]           --
|    └─Conv2d: 2-5                       [-1, 11, 24, 24]          792
|    └─BatchNorm2d: 2-6                  [-1, 11, 24, 24]          22
|    └─ReLU: 2-7                         [-1, 11, 24, 24]          --
|    └─Dropout2d: 2-8                    [-1, 11, 24, 24]          --
├─Sequential: 1-2                        [-1, 10, 12, 12]          --
|    └─Conv2d: 2-9                       [-1, 10, 24, 24]          110
|    └─MaxPool2d: 2-10                   [-1, 10, 12, 12]          --
├─Sequential: 1-3                        [-1, 31, 8, 8]            --
|    └─Conv2d: 2-11                      [-1, 12, 10, 10]          1,080
|    └─BatchNorm2d: 2-12                 [-1, 12, 10, 10]          24
|    └─ReLU: 2-13                        [-1, 12, 10, 10]          --
|    └─Dropout2d: 2-14                   [-1, 12, 10, 10]          --
|    └─Conv2d: 2-15                      [-1, 14, 8, 8]            1,512
|    └─BatchNorm2d: 2-16                 [-1, 14, 8, 8]            28
|    └─ReLU: 2-17                        [-1, 14, 8, 8]            --
|    └─Dropout2d: 2-18                   [-1, 14, 8, 8]            --
|    └─Conv2d: 2-19                      [-1, 31, 8, 8]            3,906
|    └─BatchNorm2d: 2-20                 [-1, 31, 8, 8]            62
|    └─ReLU: 2-21                        [-1, 31, 8, 8]            --
|    └─Dropout2d: 2-22                   [-1, 31, 8, 8]            --
├─Sequential: 1-4                        [-1, 10, 8, 8]            --
|    └─Conv2d: 2-23                      [-1, 10, 8, 8]            310
├─AdaptiveAvgPool2d: 1-5                 [-1, 10, 1, 1]            --
==========================================================================================
Total params: 7,934
Trainable params: 7,934
Non-trainable params: 0
Total mult-adds (M): 1.05
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.29
Params size (MB): 0.03
Estimated Total Size (MB): 0.32
==========================================================================================
conv1.0.weight		 torch.Size([8, 1, 3, 3])
conv1.1.weight		 torch.Size([8])
conv1.1.bias		 torch.Size([8])
conv1.4.weight		 torch.Size([11, 8, 3, 3])
conv1.5.weight		 torch.Size([11])
conv1.5.bias		 torch.Size([11])
trans1.0.weight		 torch.Size([10, 11, 1, 1])
conv2.0.weight		 torch.Size([12, 10, 3, 3])
conv2.1.weight		 torch.Size([12])
conv2.1.bias		 torch.Size([12])
conv2.4.weight		 torch.Size([14, 12, 3, 3])
conv2.5.weight		 torch.Size([14])
conv2.5.bias		 torch.Size([14])
conv2.8.weight		 torch.Size([31, 14, 3, 3])
conv2.9.weight		 torch.Size([31])
conv2.9.bias		 torch.Size([31])
trans2.0.weight		 torch.Size([10, 31, 1, 1])
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 11, 24, 24]          --
|    └─Conv2d: 2-1                       [-1, 8, 26, 26]           72
|    └─BatchNorm2d: 2-2                  [-1, 8, 26, 26]           16
|    └─ReLU: 2-3                         [-1, 8, 26, 26]           --
|    └─Dropout2d: 2-4                    [-1, 8, 26, 26]           --
|    └─Conv2d: 2-5                       [-1, 11, 24, 24]          792
|    └─BatchNorm2d: 2-6                  [-1, 11, 24, 24]          22
|    └─ReLU: 2-7                         [-1, 11, 24, 24]          --
|    └─Dropout2d: 2-8                    [-1, 11, 24, 24]          --
├─Sequential: 1-2                        [-1, 10, 12, 12]          --
|    └─Conv2d: 2-9                       [-1, 10, 24, 24]          110
|    └─MaxPool2d: 2-10                   [-1, 10, 12, 12]          --
├─Sequential: 1-3                        [-1, 31, 8, 8]            --
|    └─Conv2d: 2-11                      [-1, 12, 10, 10]          1,080
|    └─BatchNorm2d: 2-12                 [-1, 12, 10, 10]          24
|    └─ReLU: 2-13                        [-1, 12, 10, 10]          --
|    └─Dropout2d: 2-14                   [-1, 12, 10, 10]          --
|    └─Conv2d: 2-15                      [-1, 14, 8, 8]            1,512
|    └─BatchNorm2d: 2-16                 [-1, 14, 8, 8]            28
|    └─ReLU: 2-17                        [-1, 14, 8, 8]            --
|    └─Dropout2d: 2-18                   [-1, 14, 8, 8]            --
|    └─Conv2d: 2-19                      [-1, 31, 8, 8]            3,906
|    └─BatchNorm2d: 2-20                 [-1, 31, 8, 8]            62
|    └─ReLU: 2-21                        [-1, 31, 8, 8]            --
|    └─Dropout2d: 2-22                   [-1, 31, 8, 8]            --
├─Sequential: 1-4                        [-1, 10, 8, 8]            --
|    └─Conv2d: 2-23                      [-1, 10, 8, 8]            310
├─AdaptiveAvgPool2d: 1-5                 [-1, 10, 1, 1]            --
==========================================================================================
Total params: 7,934
Trainable params: 7,934
Non-trainable params: 0
Total mult-adds (M): 1.05
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.29
Params size (MB): 0.03
Estimated Total Size (MB): 0.32
==========================================================================================
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.2458 Batch_id=468 Accuracy=90.67: 100%|██████████| 469/469 [00:22<00:00, 20.78it/s]
Test set: Average loss: 0.0679, Accuracy: 9802/10000 (98.02%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.1232 Batch_id=468 Accuracy=96.76: 100%|██████████| 469/469 [00:21<00:00, 21.32it/s]
Test set: Average loss: 0.0517, Accuracy: 9843/10000 (98.43%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0771 Batch_id=468 Accuracy=97.56: 100%|██████████| 469/469 [00:21<00:00, 21.46it/s]
Test set: Average loss: 0.0377, Accuracy: 9885/10000 (98.85%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0307 Batch_id=468 Accuracy=97.69: 100%|██████████| 469/469 [00:23<00:00, 19.81it/s]
Test set: Average loss: 0.0352, Accuracy: 9883/10000 (98.83%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0267 Batch_id=468 Accuracy=97.97: 100%|██████████| 469/469 [00:23<00:00, 19.95it/s]
Test set: Average loss: 0.0322, Accuracy: 9898/10000 (98.98%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0544 Batch_id=468 Accuracy=98.15: 100%|██████████| 469/469 [00:23<00:00, 19.62it/s]
Test set: Average loss: 0.0327, Accuracy: 9886/10000 (98.86%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 7
Train: Loss=0.0437 Batch_id=468 Accuracy=98.62: 100%|██████████| 469/469 [00:22<00:00, 21.09it/s]
Test set: Average loss: 0.0210, Accuracy: 9932/10000 (99.32%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 8
Train: Loss=0.0197 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:21<00:00, 21.45it/s]
Test set: Average loss: 0.0198, Accuracy: 9939/10000 (99.39%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: Loss=0.0470 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:22<00:00, 21.09it/s]
Test set: Average loss: 0.0185, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: Loss=0.0193 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:21<00:00, 21.39it/s]
Test set: Average loss: 0.0187, Accuracy: 9935/10000 (99.35%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 11
Train: Loss=0.0042 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:22<00:00, 21.14it/s]
Test set: Average loss: 0.0186, Accuracy: 9933/10000 (99.33%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 12
Train: Loss=0.0274 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:23<00:00, 19.81it/s]
Test set: Average loss: 0.0185, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 13
Train: Loss=0.0832 Batch_id=468 Accuracy=98.90: 100%|██████████| 469/469 [00:22<00:00, 21.08it/s]
Test set: Average loss: 0.0178, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 14
Train: Loss=0.0105 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:22<00:00, 21.02it/s]
Test set: Average loss: 0.0176, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 15
Train: Loss=0.0368 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:22<00:00, 21.23it/s]
Test set: Average loss: 0.0173, Accuracy: 9943/10000 (99.43%)

Adjusting learning rate of group 0 to 1.0000e-04.
```
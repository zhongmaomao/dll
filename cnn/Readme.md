We change the alexnet into cifar-10 Picture Classification task  

The net structures are as follows:  

AlexNet(  
  (conv): Sequential(  
    (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (1): ReLU()  
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
    (3): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (4): ReLU()  
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
    (6): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (7): ReLU()  
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
    (9): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (10): ReLU()  
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  )  
  (fc): Sequential(  
    (0): Dropout(p=dpo, inplace=False)  
    (1): Linear(in_features=256, out_features=120, bias=True)  
    (2): ReLU()  
    (3): Dropout(p=dpo, inplace=False)  
    (4): Linear(in_features=120, out_features=84, bias=True)  
    (5): ReLU()  
    (6): Dropout(p=dpo, inplace=False)  
    (7): Linear(in_features=84, out_features=10, bias=True)  
  )  
)  
  
In which, we disscussed the influence of the Hyperparameter dpo(the probability of dropout in Dropout Layer)  

And the results are as follows:  

![image](Accuracy%20with%20Different%20Dropout%20Probability(in%20Validation%20Set).png)  

![image](Accuracy%20Deviation(Train%20-%20Validation)%20with%20Different%20Dropout%20Probability.png)  
(Notice that the accuracy in train set is computed with Dropout Layer while validation set on the contrary)

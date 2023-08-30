# Thickness Estimation from Time-of-Flight Data


## About the Project
This project was prepared as the final assignment of "Recent Advances in Machine Learning" course during my master study in University of Siegen. 

</br>

## 1 - Introduction
The distance information can be obtained using a type of active sensor called ToF(Time of Flight) camera. It works by sending modulated light and then capturing the reflected light, which contains data about the distance it has traveled. Each material surface, and consequently, each material, reacts differently to incoming light. Previous studies have attempted to estimate the type of material using this property.

Moreover, the thickness of a material significantly impacts the behavior of incoming light. In this study, various neural network architectures were designed to predict the thickness of materials using ToF data. By testing with different parameters and continuously improving these architectures, accurate results have been achieved, with an error rate of less than 1 millimeter in certain cases.

</br>

## 2 - Data Analysis
The data contains complex measurements of wax samples in two different backgrounds: foam board and mirror. Measurements were made at 8 different frequencies, from 20MHz to 160MHz, and each value consists of real and imaginary parts.   
Therefore, the final dataset, which will be used with the neural network, consists of 16 columns obtained from these measurements  and the observation angle. Since the angle affects the measurement, it was added to the dataset as a feature.

The final dataset is transformed into tabular data with 17 feature columns and 1 column which represent thickness values. Thus, this problem can now be considered as a regression task. Before training the model, each dataset was split into 80\% training and 20\% testing. Also, 90\% of the training set was reserved for training and 10\% for validation during training.

</br>

## 3 - Model Training
Models are designed and trained using the Keras library, taking advantage of two useful features during training: reducing the learning rate when the validation loss no longer improves, and early stopping, which stops the training if the validation loss does not improve for a certain number of epochs.    
Therefore, models were not trained with a specific epoch value. The Keras library itself has decided to stop. The Mean Absolute Error(MAE) value was taken as a reference during model training.  In each attempt, an effort was made to achieve a lower MAE value.

</br>

### 3.1 - Fully connected neural networks
It was started using only dense layers and was improved at every step. Adding Batch Normalization shortened the training time and reduced the mean absolute error value. But adding dropout layers caused a negative effect. Thus, they did not be used in further steps. Adding L2 regularization to dense layers, gave the best results in comparison to L1 and L1+L2 regularization. This model is shown in Table 1 as "Model*".
This best model was modified with different parameters such as Selu, Elu, Sigmoid activation functions and Adam, RMSProp optimizers. But they did not have a significant positive effect.

| Feature of network structure                                     | Foam background dataset | Mirror background dataset |
|------------------------------------------------------------------|-------------------------|---------------------------|
| Only Dense Layers                                                | 1.0579                  | 1.3847                    |
| Dense Layers and Batch Normalization                             | 0.9054                  | 1.1651                    |
| Dense Layers, Batch Normalization and Dropout                    | 0.9328                  | 1.1831                    |
| Dense, Batch Normalization and L1 regularizers                   | 0.9351                  | 1.1894                    |
| **Dense, Batch Normalization and L2 regularizers (Model\*)** | **0.8993**              | **1.1578**                |
| Dense, Batch Normalization and L1+L2 regularizers                | 0.9159                  | 1.1920                    |
| Model* with Selu Activaton Function                              | 0.9088                  | 1.1585                    |
| Model* with Elu Activaton Function                               | 0.9153                  | 1.1796                    |
| Model* with Sigmoid Activaton and RMS optimizer                  | 0.9441                  | 1.2450                    |
| Model* with 512 neurons in the first layer                       | 0.9103                  | 1.1743                    |
| Model* with 1024 neurons in the first layer                      | 1.0514                  | 1.3221                    |
| Model* with 128 neurons in the first layer                       | 0.9089                  | 1.1597                    |


</br>


### 3.2 - Convolutional neural networks
After the best model was obtained with the fully connected neural network, continued by adding convolution layers. However, when these layers were placed in the middle of the fully connected neural network, accurate results were not obtained. Therefore, it was decided to position them before the FCNN.   
When convolutional layers were used without max pooling, they produced better results than the same structure with max pooling, as shown in Table 2. Thus, the model that had the best results was obtained by using three convolution layers and then five dense layers with L2 regularization and Relu activation function.

| Feature of network structure                   | Foam background dataset | Mirror background dataset |
| ---------------------------------------------- | ----------------------- | ------------------------- |
| Model* with Conv. layers in the middle         | 2.0544                  | 2.3670                    |
| First CNN(input 256 neurons), then Model*     | 0.9383                  | 1.2184                    |
| CNN(input 128 neurons) with pooling, then Model* | 0.9453                  | 1.2117                    |
| **CNN without pooling, then Model***           | **0.8830**              | **1.1427**                |

</br>

### 3.3 - Other architectures
Since tabular data was used in this study, a review of the literature was done to see if they could be used in other methods. Table 3 shows the outcomes of using two methods.  

TabNet is an innovative deep learning model for tabular data. It makes learning more effective by concentrating on the most crucial elements by using sequential attention to identify key features at each step. Tabnet can be installed with pip command as a package and used with both PyTorch and Keras.


Wide \& Deep Neural Networks combine the benefits of remembering particular data patterns and being able to understand new patterns. The wide part means linear model and it is good at remembering specific combinations of data, while the deep neural networks can learn to recognize new combinations of data. By using this method, the model performs better and is better suited for tabular data analysis.   
A dense layer with 1 unit was used as a linear model and then it combined with the best model obtained with fully connected neural networks above. 

| Name                             | Foam background dataset | Mirror background dataset |
| -------------------------------- | ----------------------- | ------------------------- |
| TabNet                           | 1.4113                  | 1.4819                    |
| Wide & Deep Neural Networks      | 0.8988                  | 1.1773                    |

</br>

## Conclusion
This study demonstrates the effectiveness of neural networks in estimating material thickness from ToF data, with potential applications in various industries. The achieved results of less than 1 millimeter error rate in certain cases, open up exciting opportunities for enhancing material sensing capabilities and non-destructive material analysis in practical scenarios.

As further study, additional training data can be used or pretrained models can be trained with the transfer learning method to improve the results.

This projrect is based on Machine learning (ML) and Artificial intelligence (AI) 

                      Introduction 
 
Dental caries is one of the most highly chronic oral disease affecting populations of various ages worldwide and was reported in 2010 as the 10th most prevalent Condition. According to the World Health Organization (WHO), approximately around 1.8 billion caries cases are recorded every year.  The current works to tackle the dental caries issue are either restrained or insufficient. Saliva, hereditary, bacteria and diet are some factors leading to dental caries Two methods are used to diagnose dental caries: radiographic procedures, consisting of detecting dental caries using digital X-ray images and clinical method, which is classified as a visual examination of oral condition to detect caries. These works have one thing in common, the usage of only one type of data (either image data or clinically collected data) for dental caries  detection. 
Meanwhile, combining several types of oral healthcare data could generate a more credible diagnosis, therein removing the uncertainty of the possible sources the disease. This uncertainty can cause the ratio of dental caries population to increase drastically. Therefore, although dental radiography (including panoramic, periapical, and bitewing views), and explorer (or dental probe), which are widely used and regarded to be highly reliable diagnostic tools for the detection of dental caries, much of the screening and final diagnosis tends to rely on empirical evidence. 
Recently, one aspect of artificial intelligence and deep learning—convolutional neural networks (CNNs)—has demonstrated excellent performance in computer vision including object, facial and activity recognition, tracking, and three-dimensional mapping and localization. Medical segmentation and diagnosis are one of the most important fields in which image processing and pattern recognition procedures have been adopted. In particular, detection and classification of diabetic retinopathy, skin cancer, and pulmonary tuberculosis using deep learning-based CNN models have already demonstrated very high accuracy and efficiency, with promising clinical applications in contrast, however, there have been few studies based on deep CNN architectures in the dental field, and research investigating detection and diagnosis of dental caries is also more limited. Accordingly, the aim of the present study was to evaluate  the efficacy of deep CNN algorithms for the detection and diagnosis of dental caries in periapical radiographs Machine Learning (ML) has been used in recent years to help improve a diverse DSP system. Some works were conducted in oral health using ML, and several prediction algorithms have been applied to detect dental caries. Support Vector Machines (SVM) were used to classify dental root caries Random Forest (RF) and Artificial Neural Network (ANN) were used and provided good results in detecting dental caries. 



OBJECTIVE 

•	Background: Dental caries is a growing concern affecting a large number of people. Existing methods to protect people from dental caries are deficient.

•	Need for a Solution: A Decision Support System (DSS) is crucial to provide accurate insights into dental caries. Machine Learning (ML) has been increasingly used to enhance DSS systems.

•	Previous Work: ML, especially Support Vector Machines (SVM), Random Forest (RF), and Artificial Neural Network (ANN), has been applied to predict and detect dental caries. SVM was used for classifying dental root caries. RF and ANN demonstrated good results in detecting dental caries.

•	Novel Approach: The proposed solution introduces a novel approach for dental caries prediction. A multi-modal deep neural network with two pathways is utilized. The model has the ability to learn patterns from heterogeneous features of different data sources.

•	Hybrid Model : A Hybrid model is employed, consisting of Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN).This model leverages the strengths of both CNN and ANN to learn from diverse data sources.

•	Information Utilization:  The multi-modal approach harnesses the capability of providing more information by using different data sources .The model aims to make precise and accurate predictions of dental caries within a given population.


In summary, the proposed solution addresses the deficiency in existing methods by introducing a novel multi-modal deep neural network, leveraging the power of Hybrid models for dental caries prediction, and learning from diverse data sources to enhance accuracy.




OBJECTIVE 

•	Background: Dental caries is a growing concern affecting a large number of people. Existing methods to protect people from dental caries are deficient.

•	Need for a Solution: A Decision Support System (DSS) is crucial to provide accurate insights into dental caries. Machine Learning (ML) has been increasingly used to enhance DSS systems.

•	Previous Work: ML, especially Support Vector Machines (SVM), Random Forest (RF), and Artificial Neural Network (ANN), has been applied to predict and detect dental caries. SVM was used for classifying dental root caries. RF and ANN demonstrated good results in detecting dental caries.

•	Novel Approach: The proposed solution introduces a novel approach for dental caries prediction. A multi-modal deep neural network with two pathways is utilized. The model has the ability to learn patterns from heterogeneous features of different data sources.

•	Hybrid Model : A Hybrid model is employed, consisting of Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN).This model leverages the strengths of both CNN and ANN to learn from diverse data sources.

•	Information Utilization:  The multi-modal approach harnesses the capability of providing more information by using different data sources .The model aims to make precise and accurate predictions of dental caries within a given population.


In summary, the proposed solution addresses the deficiency in existing methods by introducing a novel multi-modal deep neural network, leveraging the power of Hybrid models for dental caries prediction, and learning from diverse data sources to enhance accuracy.






 
                                                  Chapter - 2 
Description and Work done 
 
Data Processing 
As mentioned in the data collection stage, the proposed algorithm uses two types of data, a numerical dataset and image data sets we have used image data sets in our trained model of our project having an image dataset with a total of 1554 samples with caries 1155 samples and without caries 399 samples. These datasets must undergo through a series of preprocessing steps before being used by the proposed hybrid model to display high performance. After the data collection we make  one repository on git hub for the storage and processing  of our data set  after that we use google colab for implementing our code that includes all important library and   cloneing data set repository , defining deep learning model , setting up of shapes and number of classes  setting up data generators ; after all this  we train the model after training and validation of Accuracy\loss value we display  ROC and AUC Curve  and confusion matrix .

 
 
 

                                                                                    Flow chat 




Data Sets
 
URL(Kaggle):- Click Here for Data Sets 
Total images = 1554
Training Data 
Total images = 1260
Carries images - 945 
Non Carries images - 315 
Testing Data  
Total images = 294
Carries images - 210 
Non caries images - 84 
                                  
                                    


                                  




                                         Data set images 

 
 
 
                                               



We have come to our project’s conclusion and we came to know that numerous experiments were performed on the datasets, and the results revealed that the proposed model was very effective. When compared to numerous earlier single modality models for the prediction of dental caries, the model presented in this work displayed better accuracy, precision, recall, f1score, and AUC-ROC Curve. As a result, this highlights the high likelihood of using multi-modality in dental caries prediction. 
 
During the experimentation stage, there were some limitations that had to be overcome, such as the lack of a large collection of X-ray dental images, by using dental images and data augmentation as described in the data preparation section. For our upcoming work, the labeling of X-ray images dataset is currently being performed. We express our gratitude to everyone who has contributed to the success of this project, whether through collaboration, guidance, or support. This journey has been enlightening, and we look forward to the continued evolution of project  in the future.
 
                                          Result

The final result of our model is having 211 caries and 84 non carries in confusion matrix and  the graph of our model is show below in this project file , therefore the sensitivity of our project is 0.7857  and the specificity is 0.9524 and accuracy  is 0.9048  .
The other results like  precision , recall value , f1 score  are as follow :
For non carries:
•	Precision : 0.92
•	Recall value : 0.95 
•	F1 score : 0.93

For carries:
•	Precision : 0.87
•	Recall value : 0.79
•	F1 score : 0.82
 

 
 	  
 
 
 
 
Finally, to summarize the performance of the proposed model
 by comparing the frequency of the predicted classes to the
 expected classes, we performed a confusion matrix of the proposed model
 


Sensitivity	0.7857
Specificity	0.9524
Accuracy 	0.9048





	               F1-score	                Support
Accuracy	                  0.90	                    294




 
	precision	recall	F1- score	Support
Non-carries	0.92	0.95	0.93	210
Carries 	0.87	0.79	0.82	84
Macro avg	0.90	0.88	0.89	294
Weight avg	0.91	0.91	0.91	294
 






we proceeded by computing the AUC-ROC of the model.
 The higher the AUC-ROC value, the better the model
 performance when distinguishing between the label
 “non-carries” and label “carries” classes. The proposed
 model yielded an AUC-ROC of 0.96. Figure 4 shows the performance evaluation of the proposed model in the case 
of AUC-ROC.

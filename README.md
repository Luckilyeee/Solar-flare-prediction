# Solar-flare-prediction
This project is about Solar flare prediction based on the time series of Solar MagneticField Parameters. The dataset is from Kaggle: **https://www.kaggle.com/c/bigdata2020-flare-prediction/data**. The dataset consists of 5 classes,namely X, M, C, B, and Q. To conduct the solar flare prediction task, the X- and M-classes of solar flares are considered to be positive class to represent the flaring active regions, the C-, B-, and Q-classesof solar flares are considered to be negative class to representthe non-flaring regions. The goal of this project is to conduct a binary classification between flaring (X-  and  M-classes) and non-flaring (C-, B-, and Q-classes) Active Regions. 

I apply traditional machine learning models (Interval-based model,Shaplet-based  model,  and  Dictionary-based  model)  and  deeplearning models (Convolutional Neural Network and MultivariateLong  Short  Term  Memory  Fully  Convolutional  Networks)  to achieve  the  prediction  task.  Their  performances  are  evaluatedby using different metrics–Accuracy, Precision, Recall, F1 score,True  skill  statistic(TSS).

The dataPrepare.py file is used to generate balanced datasets for our experiments. 

The Traditional_binary_33*60.ipynb (scenario (i) of small dataset) and updating 2_Traditional_binary_33*60.ipynb (scenario (ii) of larger dataset) files are the experimental details for tradictional machine learning in solar flare prediction. 

The 1_Deep_learning_binary_33_60.ipynb (scenario (i) of small dataset), 3_Deep_learning_binary_33_60.ipynb (scenario (ii) of larger dataset) , and 2_Deep_learning_binary_33_60.ipynb (much more larger dataset) files are the experimental details for deep learning in solar flare prediction. 


# Mushroom-Classification
## Abstract

An analysis was performed on the UCI and Phillips University mushroom dataset to classify edible and inedible mushrooms.  The data from the two sources were combined together, made tidy and then augmented with dummy variables to ensure all metrics were accounted for using integer inputs.  The classifications are based on physical features of the mushrooms such as its cap, gills, stalk, rings, and etc. Each classification model developed was measured by its accuracy, confusion matrix, recall, precision, f-score, kappa score, and AUC.  After training, KNN, SVM, Adaboost, decision tree, and random forest, they were all above 90% accuracy, but the random forest classifier stood out as the best with an astounding 99% classifying accuracy in our testing, with a 25% training percentage. 

## Background 

There are over 14,000 identified species of mushrooms, many of which are widely enjoyed as a culinary delicacy, and others that even a gram of which could result in a slow and painful death.  Naturally, this rather stark dichotomy necessitates a level of confidence in your decisions should you choose to go out and forage for your own mushrooms, whether it be for culinary or medicinal purposes.  Types of mushrooms vary depending on regional location, and because of this, specific species identification can be difficult, and the novice mushroomer should stick to genus’ of which every species is edible, such as chanterelles, and most morels.  Unlike these quite friendly genuses, there are much more dangerous ones that include some edible, and some deadly mushrooms, one such genus being Amanita, which contains the very tasty Amanita Fulva, to the very deadly Amanita Virosa.  The combined data set we analyzed contains many species from the amanita genus as well as Lepiota, both of which contain edible and inedible mushrooms.  By analyzing specific aspects of the mushroom's appearance, cap color, gill attachment,  gill color, and others, a model could be trained to distinguish poisonous mushrooms from edible ones.

## Problem

The goal of this project is to develop multiple classification models and find the best performing model that can classify whether a mushroom is edible or not. The data was gathered by combining two datasets and then preprocessed for training. The models developed were binary classification models such as KNN and Decision Tree. Each model was tuned to find the most optimal versions of each model. Tuning methods such as GridSearchCV, n_estimators, and loops were used to tune these models. To validate the models and compare them, we used metrics such as accuracy and Cohen’s kappa.

## Dataset

The mushroom dataset used was obtained by the UCI Machine Learning Repository and Phillips University Marburg. This dataset contains two mushroom families that include the Agaricus and Lepiota containing 58 attributes. These attributes are the characteristics of the mushrooms which allows the variety of attributes to be used to detect whether mushrooms are edible or inedible.
	Most machine learning algorithms do not process categorical data therefore the dataset was transformed into numerical form where a string is assigned a numerical value in place of the string. Since the observations are ordinal, dummy variables can be used to represent categorical data by assigning 0’s and 1’s to the data. This helps the computer understand the relationship between the observations and attributes based on the specific algorithm. The number of unique observations per attribute determines the number of columns in the dummy variable dataset.

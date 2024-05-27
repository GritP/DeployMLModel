# Model Card

## Model Details
This <b>classification model</b> is trained on 1994 Census Bureau data from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). Its aim is to predict whether an individual's yearly income exceeded $50,000 based on the following demographic and socio-economic information:
- sex
- race
- marital status
- age
- native-country	
- education	
- relationship status
- occupation
- hours-per-week
- workclass	
- capital-gain	
- capital-loss	

To this end a <b>Histogram-based Gradient Boosting Classifier</b> from scikit-learn version 1.3.2 was used with the following options which were obtained by testing a few combinations without systematic hyperparameter optimization:
- max_leaf_nodes = 25
- early_stopping = False

The trained model is saved to the model/ folder as a pickle file.

## Intended Use
The model is intended to give a rough indication of an individual's income level in 1994 based on a limited set of attributes. As such, it is mainly an example used to practice tackling classification problems.

## Training Data
The Census dataset contains 32,561 records, 80% of those were used as training data stratified by the target variable. The target variable "salary" has two values: <=50K and >50K which were binarized. Minimal data cleansing was performed by removing leading and trailing whitespaces in the downloaded csv file.

The dataset contains 8 categorical and 6 numerical features. Categorical features were one-hot encoded. Scaling continuous variables has been tested but did not lead to a performance gain. This should be investigated further.

The fitted label binarizer and one hot encoder are also saved as pickle files in the model/ folder to reuse for evaluation and inference.

## Evaluation Data
This comprises 20% of the census dataset stratified by target variable. The saved encoder and binarizer have to be executed on the test data before inference. This is part of the process_data function in the data module.

## Metrics
The performance of the trained classification model is assessed using precision, recall, and F1 score.
For the entire evaluation dataset the following performance measures were obtained:
| Metric      | Value      |
| ------------- | ------------- |
| precision | 0.7835 |
| recall | 0.6741 |
| F1   | 0.7247 |

When looking at individual slices of data, the worst performance was observed for education categories (especially for levels of high school graduates or lower) and for some native countries like Iran, Canada, and Mexico. The latest results on different slices can be found in slices_output.csv in the ml/ folder.

## Ethical Considerations
The used dataset is 30 years old and incomplete in the sense that it only contains 32,561 records and only a limited number of attributes. Therefore, no general conclusions on the income of people of any demographic can be drawn. Furthermore, wrong predictions are expected for demographic groups which are underrepresented in the training data.

## Caveats and Recommendations
One obvious caveat is the age of the dataset. If current results are required, the model should be retrained on more up-to-date data. Additionally, there is class imbalance in the target variable with about 75% of the records belonging to individuals who earn less than $50,000/y. This should be kept in mind but is in general not expected to negatively affect model performance. As mentioned above the reason why scaling continuous variables does not improve model performance should be investigated. One possible explanation could be that age which has quite a big range and is available for every record is particularly important for predicting whether someone earns more or less than $50,000 a year. This has been verified for other tree-based algorithms which provide feature importances.
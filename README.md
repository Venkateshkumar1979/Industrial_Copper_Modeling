Project Demo Video: https://www.linkedin.com/posts/venkateshkumars_activity-7100298887684345856-y9EO?utm_source=share&utm_medium=member_desktop
# Industrial_Copper_Modeling
Problem Statement:
The copper industry deals with less complex data related to sales and pricing. However, this data suffered from issues such as skewness and noisy data, which affected the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. 

Decision Tree Regression model addressed these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 

Another area where the copper industry faces challenges is in capturing the leads. 

Decision Tree Classification model is used for evaluating and classifying leads based on how likely they are to become a customer . 

The STATUS variable with WON being considered as Success and LOST being considered as Failure.

The solution includes the following steps:

•	Exploring skewness and outliers in the dataset.

•	Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.

•	ML Regression model which predicts continuous variable ‘Selling_Price’.

•	ML Classification model which predicts Status: WON or LOST.

•	Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost)

Approach: 

•	Data Understanding

•	Data Preprocessing: 

●	Handle missing values with mean/median/mode.

●	Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform 
it back to original scale eg:dollars), boxcox transformation, or other techniques, to handle high skewness in continuous variables.

●	Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable.

•	EDA: Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot.

•	Feature Engineering: Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. And drop highly correlated columns using SNS HEATMAP.

•	Model Building and Evaluation:

●	Split the dataset into training and testing/validation sets. 

●	Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. 

●	Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.

●	Interpret the model results and assess its performance based on the defined problem statement.

●	Same steps for Regression modelling.(note: dataset contains more noise and linearity between independent variables so itll perform well only with tree based models)

•	Model GUI: Using streamlit module, created an interactive page with

•	task input( Regression or Classification) and 

•	create an input field where you can enter each column value except ‘Selling_Price’ for regression model and  except ‘Status’ for classification model. 

•	performed the same feature engineering, scaling factors, log/any transformation steps which used for training ML model and predict this new data from streamlit and display the output.

•	Pickle module was used to dump and load models such as encoder(onehot/ label/ str.cat.codes /etc), scaling models(standard scaler), ML models. First fit and then transform in separate line and use transform only for unseen data.

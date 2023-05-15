# Predicting posture of a person based on their personality.
## By: Suraj Rana Vujjini
<br>

### Link to the presentation:<br>
### Link to the video: 
<br>
<br>

## 1. Introduction:

Back pains are very common in the working population. It does not have any underlying disease that causes it, which makes it difficult to predict it. It could be caused by working out or lifting too much, prolonged sitting and lying down, sleeping in an uncomfortable position, or wearing a poorly fitting backpack. 

There is enough evidence exists on contribution of personality and posture to long-term pain management, pointing to a direct contribution of the mind-body axis. In this study we can find the relationships between posture and personality and hopefully find a model to predict the posture of a person based on their personality type obtained from the Myers-Briggs personality test.

MBTI Test: The Myers-Briggs Type Indicator, or MBTI for short is used in this project, it is a personality assessment tool that was developed in the early 20th century. The test is based on the idea that people experience the world through four primary functions: sensation, intuition, feeling, and thinking. These functions are then combined with two orientations: extraversion (E) and introversion (I). This test measures these four functions and two orientations to identify a person's personality type, which can help to better understand their behavior and preferences. There are 16 possible personality types, such as ENFP (extraverted, intuitive, feeling, perceiving) or ISTJ (introverted, sensing, thinking, judging), etc.

While it is popular, it is important to note that it has been criticized for its validity and reliability. But it still remains a popular tool for self-discovery and understanding, and it can provide valuable insights into individual differences and preferences.

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/f2988873-d918-4762-8def-390ea536a7f3)

## 2. Method

The data consists of 100 individuals, consisting of 50 men and 50 women, whose ages ranged from 13 to 82 years old. These individuals were all French-Canadian and lived in Canada between the Québec and Sorel-Tracy areas. We used the Myers-Briggs Type Indicator questionnaire to determine their personality traits and the Biotoxin analysis and report to identify any postural deviations in their biomechanical profiles.

## 3. Dataset

This dataset has been obtained from an online open ata publishing sites - Kaggle.

There are a totalof 97 rows and 20 columns in the dataset.

### Columns:

S No – Serial number of the rows <br>
AGE – Age of the participants<br>
HEIGHT – Height of the participants<br>
WEIGHT – Weight of the participants<br>
SEX – Gender of the participants<br>
ACTIVITY LEVEL – Relative level of activity of the participants<br>
Pain columns - The pain scale data consisted of a number between 0(low) to 10 (high)<br>
PAIN 1 – Pain in the neck<br>
PAIN 2 - Pain in the thoracic<br>
PAIN 3 - Pain in the lumbar<br>
PAIN 4 - Pain in the sacral<br>
MBTI - personality inventory / Myers-Briggs personality test and their respective scores in the next columns<br>
E – Participant prefers Extraversion<br>
I - Participant prefers introversion<br>
S - Participant prefers Sensing while taking information<br>
N - Participant prefers Intuition while taking information<br>
T – Participant prefers Thinking while making decisions<br>
F – Participant prefers Feeling while making decisions<br>
J - Participant prefers Judging<br>
P - Participant prefers Perceiving<br>
POSTURE – Posture of the participant<br>
<br>
The posture of the person is categorized into 4 types:<br>
A - Ideal posture<br>
B - Kyphosis-Lordosis<br>
C - Flat back<br>
D - Sway-back<br>

The columns “MBTI”, “E”, “I”, “S”, “N”, “T”, “F”, “J”, “P” represents the scores of the participants from the Myers-Briggs personality test. This test aims to find the personality of the participant.

I generated a heatmap using the Seaborn library to visualize any missing values in the dataset. The heatmap shows that there are no missing values in the dataset, as there are no white spots visible in the plot. This is a crucial step in data preprocessing to ensure that the dataset is complete and accurate before proceeding with any analysis or modeling. 

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/06f9ccb9-11dc-4a46-a339-751b73937e08)

## 4.	Exploratory Data Analysis (EDA)

I generated a heatmap of the correlation matrix for the dataset using the Seaborn library. The resulting plot shows the correlation values between each pair of variables in the dataset, with darker colors indicating higher correlation values. The plot also displays the exact correlation values in each cell. This helps to identify any strong positive or negative correlations between variables, as well as any redundant variables or variables that are highly correlated with each other. 

There is no significant correlation among the columns, this may suggest that there are no strong relationships between the variables, but it is important to note that correlation alone does not paint the whole picture, but there may be some columns, when combined with others, can provide valuable insights and predictive power.

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/8132cf6f-66da-48ab-a4f2-38a39e4821af)

There are almost as many female participants as male participants in the dataset. 

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/02f0b07c-8cdc-42d2-aa4c-50d575dc0905)

Coming to the types of posture, posture type B is the most occurring among the participants.

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/578f4403-25e1-4e47-a747-6a5db86ad7f2)

### Activity Level

The activity level is categorized into three values, High, Moderate and Low.
As seen in the graph, participants with low activity level have worst posture and the participants with high activity level have better posture. 

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/108b3699-b324-4b3b-9f6f-25110c597e76)

### Age groups of the participants

The average age of the participants is 43.86 which is a relatively younger group of people. To better visualize the data, I created a new column which contains the age slabs so that visualization of the data is easier. This is done using the help of bins and cut functions. The ages are defined into 10-year slabs. 

Most of the participants are in the age between 40 to 50 years.

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/57ae54d9-f342-4c0b-b922-ff80a4179e0c)

When comparing posture with respect to age, participants in the ages 50-60 have the worst posture of all, followed by ages 40-50 and 70-80. Overall, we can see that the older population has worst posture. Which is understandable.

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/5e744d92-058c-4f1d-bafb-0d45896ae86c)

### Height

I again created a new column that groups the height into height slabs for better visualization. This is done with the help of bins and cut functions to create height slabs in which the participants are slotted into. 

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/aa65610b-9251-4574-88d7-b1fd1fa44d42)

Most of the participants are between 65 to 70 inches followed closely by 60 to 65 inches tall. Overall, most of the participants are between 60 to 70 inches tall. 

To compare the posture with respect to the height, I created a line graph. On Y-axis it contains the posture score, which is the sum of posture values of the participants in their respective slabs. And it contains the end values of a slab on the X-axis. Though there is no well-defined trend in the graph, it can be clearly seen that the posture gets worse with respect to the height. 

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/6236c5a4-14cb-4528-ad99-f54eea5f79e9)


## 5. Classification
I used three types of classification models, which are 

•	Logistic regression

•	Decision tree

•	Gradient boosting

I performed logistic regression two times, one using the MBTI column and the other without using the MBTI column. I assigned integer value to MBTI column with reference from a website. But the accuracy of the model was greater when the MBTI column was not used. This could be due to improper assignment of values to MBTI column. I used the help of pre-processing pipelines to be used in both decision tree classifier and gradient boosting technique. 

### Pre-processing

I defined a processing pipeline for the dataset using the Pipeline and ColumnTransformer functions from the Scikit-learn library. The purpose of this pipeline is to preprocess the numerical and categorical features of the dataset separately, and then combine them into a single processed dataset.

The “num_pipeline” variable defines a pipeline for the numerical features of the dataset, which consists of a single step – “StandardScaler”. This step standardizes the numerical features by subtracting the mean and dividing by the standard deviation, which scales the features to have a mean of zero and a variance of one.

The “cat_pipeline” variable defines a pipeline for the categorical features of the dataset, which consists of a single step - OneHotEncoder. This step encodes the categorical features into integer values using one-hot encoding, which creates a binary column for each category in the feature. The handle_unknown='error' parameter is used to raise an error if there are any unknown categories in the data, and the drop='first' parameter is used to drop the first category column for each feature to avoid multicollinearity.

The “processing_pipeline” variable merges the numerical and categorical pipelines into a single processing pipeline using the ColumnTransformer function. This function applies the specified transformers to the specified columns of the dataset, and concatenates the resulting transformed columns into a single processed dataset. The transformers parameter is used to specify the list of transformers to apply to each set of columns, where the first tuple in the list specifies the numerical transformer and the second tuple specifies the categorical transformer. The “numerical_list” and “categorical_list” variables are used to specify the list of numerical and categorical columns in the dataset, respectively.

Here are the unwanted columns: 'S No’, ‘PAIN 1','PAIN 2', 'PAIN 3', 'PAIN 4','MBTI','POSTURE'

### Logistic Regression

I made use of the following columns for logistic regression:

['AGE', 'HEIGHT', 'WEIGHT', 'SEX', 'ACTIVITY LEVEL', 'MBTI', 'E', 'I', 'S', 'N', 'T', 'F', 'J','P']

I left out the pain values as they are not part of the MBTI test, and the main aim of the project is to predict posture just based on MBTI values (Personality test). 

The test_size parameter is set to 0.2, which means that 20% of the data is allocated to the testing dataset, and 80% of the data is used for the training dataset. The random_state parameter is set to 124 to ensure that the same random split is generated every time the code is run.

I performed logistic regression two times, one with the use of mapped MBTI column and the other without the mapping. 

The mapping was based on values from a journal published online: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0037450

Here is the mapping information:

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/c86db2e6-88cf-4fcf-9c44-92a6b321816e)

Here are the results with mapping:
 
![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/6f292f2d-e91e-401c-be3c-d96aac8e96b0)

Accuracy: 40%

And without mapping (completely ruling out MBTI column):

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/28eb3da1-4c43-4726-a52f-b67e21b297ba)

Accuracy: 55%

The accuracy is actually better without the use of the mapped MBTI column. This could be due to improper mapping of the column, or because the MBTI values in the dataset are already the values derived from the EISNTFJP column. 

### Decision Tree

The modeling_pipeline variable defines the machine learning pipeline using the Pipeline function. It takes the processing_pipeline defined earlier and the DecisionTreeClassifier function as its two steps. The DecisionTreeClassifier is a classification model that builds a decision tree from the training data to predict the target variable.

The generate_splits function defines a splitting function to split the dataset into training and testing sets using the train_test_split function from Scikit-learn. The function returns the split dataset. I allocated 20% of the data to testing dataset and 80% of the data to the training dataset. 

Here is the resulting classification report:

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/2d8e443d-2525-488e-871b-06a75d116e4c)

This time, the accuracy of the model is 60% which is a little better over the logistic regression model.

### Gradient boosting

For this classification, I made use of the same preprocessing pipeline used for decision tree classifier, and also similar split function to split the dataset into test and train dataset. 

Here is the classification report:

![image](https://github.com/SurajRanaVujjini/data603-sp22/assets/90857782/19204618-cf67-4643-bf95-64280ab4765c)

The accuracy of this model is the same as the last one, which is 60%. The precision, recall and f1-score values of both the models seem to be similar as well. 

## 6. Conclusion

This analysis suggests that there is a correlation between personality traits of a person and their posture. Even though the models have low accuracy that may not be sufficient for certain applications, the Precision, recall, and F1 values of the model are good. It is important to understand that the small size of the dataset may have limited the ability of the model to learn more complex patterns and relationships. This shows that with a large enough data pool, we could theoretically predict the posture of a person based on their personality.  Analysis of a bigger dataset might give better results as the program will have a larger dataset to train on.


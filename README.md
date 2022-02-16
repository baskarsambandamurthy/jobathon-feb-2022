# JOB-A-THON - February 2022
## Score
- Private LB Rank:  
- Private LB Score:  
- Public LB Rank: 55th
- Public LB Score:  0.4256

## Key points for top score achievement

1. Understanding of high cardinality of user_id field
2. Test and Train similarity differences from EDA
3. Target Encoding of the user_id field
4. Tuning parameters of Gradient Boosting algorithm XGBoost
5. Test Predictions using full train data set with the iterations 

## Feature Engineering

### Simple Feature Engineering

- Label encoding is performed on the categorical data `gender`
- Feature `profession` is ordinal data type and its labels are encoded using ordered set of values

### Advanced Feature Engineering

#### Target Encoding
- The `user_id` categorical field has high cardinality (more than 20000 unique values in train) and if this categorical field is converted using one hot encoding, it would generate around 20000 features and this would consume huge memory and cpu. The optimaly techinique to utilize this field is to apply target encoding on this field.
- Target Encoding should not be performed before cross validation and it has been performed during cross validation.
> **Note**: If Target Encoding is performed before cross validation (i.e) performed for entire train set and then if the cross validation is performed using such encoding, then it would result in target leak which means that the partial train set of each fold has got the target leak of the validation set. This would result in very good validation score and poor test score.
- As pycaret package is used for model training and cross validation, the target encoding is mentioned as custom_pipeline parameter for the pycaret setup.
> **Note**: `ColumnTransformer()` is used as pipeline transformer on the columns to transform target encoding and `TargetEncoder()` is used as the actual transformer for target encoding.
- Features `video_id` and `category_id` are also converted using target encoding

#### Feature Aggregations

- User Groupings: Count of videos and Count of categories grouped per user are generated. 
- Category Groupings: Count of users and Count of videos grouped per category are generated. 

List of feature aggregations are as below

| Feature Name |  Grouping By | Description
|----------------------|-------------------------------|-------------------------------|
| user_video_count   | user_id | Count of videos grouped per user
| user_category_count      | user_id |Count of categories grouped per user
| category_user_count      | category_id |Count of users grouped per category
| category_video_count      | category_id |Count of videos grouped per category

#### Usage of original user_id

= Original user_id field is also used as a feature besides the target encoded value of the user_id field

## Model Build - Train - Predict

### Process

1. Model is evaluated using 10 fold cross validation and `KFold` technique is used. Here fixed number of estimators are used.
2. Evaluated model is tuned for optimal parameters for which the validation score yields better results
3. Then Model is trained using full training set with the tuned model parameters and number of estimators. This model is called `Final Model`
> **Note**: In the `Final Model`, full training set is used which is different from cross validation where only partial training set is used for each fold.
5. Predictions of the Test set are performed using the trained `Final Model`.





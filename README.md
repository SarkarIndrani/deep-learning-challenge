## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

  - EIN and NAME—Identification columns
  - APPLICATION_TYPE—Alphabet Soup application type
  - AFFILIATION—Affiliated sector of industry
  - CLASSIFICATION—Government organization classification
  - USE_CASE—Use case for funding
  - ORGANIZATION—Organization type
  - STATUS—Active status
  - INCOME_AMT—Income classification
  - SPECIAL_CONSIDERATIONS—Special considerations for application
  - ASK_AMT—Funding amount requested
  - IS_SUCCESSFUL—Was the money used effectively

### Overview of the analysis: Explain the purpose of this analysis

Step1: 

  - loaded the charity_csv file into a Pandas DataFrame and dropped unnecessary columns.
  - determined the number of unique values in each column.
  - To reduce the number of unique values we performed binning on the "APPLICATION_TYPE" and "CLASSIFICATION" columns.
  - To encode the categorical variables, utilized the pd.get_dummies() function.
  - Split the preprocessed data into a features array, X, and a target array, y, and used the train_test_split function to divide the data into training and testing datasets.
  - Finally, scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, and then applying the transform function.

Step2:

  - I complied, trained, and evaluated the binary classification model to calculate the model's loss and accuracy. This involved creating a neural network model by assinging the number of input features and nodes for each layer using TensorFlow and Keras. First I created one hidden layer then added second hidden layer with an appropriate activation function. 

Step3: 

  Goal: To optimize the model's performance to achieve a predictive target accuracy of more than 75%

  To achieve this goal, I made several attempts. Initially I started with three hidden layers, and the model yielded 72% accuracy score. On second attempt, with the same three layers, I just added nodes, and it yielded 73% accuracy. Lastly I added a forth layer, but I got the same 73%  accuracy. I experimented farther with adding more nodes and layers but the accuracy score stayed between 72% to 73%.

### Results

#### Data Preprocessing

#### Question: What variable(s) are the target(s) for your model?

Answer: The IS_SUCCESSFUL variable serves as a binary classifier with two potential outcomes: either the decision to provide funding for an organization is successful or not.


#### Question: What variable(s) are the features for your model?

Answer: After determining the unique value counts for each column, we decided to narrow down the available features for our model. We focused on those columns that had more than 10 unique values, namely NAME, APPLICATION_TYPE, and CLASSIFICATION. These features possess ideal qualitative properties for training the model.

#### Question: What variable(s) should be removed from the input data because they are neither targets nor features?

Answer: The variables "EIN" and "NAME" were identified as irrelevant for our model and were removed from the input data as they neither served as targets nor features.

### Compiling, Training, and Evaluating the Model

#### Question: How many neurons, layers, and activation functions did you select for your neural network model, and why?

Answer: Number of hidden layers: two (80,30). The "relu" activation function was used in the first and second hidden layers, and the "sigmoid" activation function for the output layer. Sigmoid is well-suited for binary classification in neural networks. Trainable Params: 5,981.

#### Question: Were you able to achieve the target model performance?

Answer: Accuracy measures how well the model predicts in terms of percentage. On the other hand, loss value predicts the percentage of errors. After adding several hidden layers the accuracy  improved from 72% to 73%.

### Summary

I achieved a 73% accuracy by iterating through different models and making various optimization attempts.


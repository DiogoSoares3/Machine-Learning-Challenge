# Machine Learning Challenge

The description of the problem is written in the PDF file in the root of the project. The code to solve this specific problem is in the "notebooks" directory. After some interpretation of the problem and some exploratory data analysis, a machine learning model was created that could predict, with 95% recall, new trucks that could have failures in their air system. The PDF file containing the challenge description has 16 questions that will be answered below.

Since the first 4 questions are related to problem interpretation, they'll be answered below. Questions 5 to 12 will be answered in the notebook file in the "notebooks" directory. Finally, questions 13 to 16 will be answered in this README file with references to the "api" directory, where the best chosen model was put into production..

## Challenge Activities

### Question 1

What steps would you take to solve this problem? Please describe as completely and clearly as possible all the steps that you see as essential for solving the problem.

- First, we need to understand the problem. In the last 3 years, there has been a noticeable increase in expenses related to the maintenance of the air system of vehicles, so our job is to help our client save money by identifying patterns in the features that may end up causing problems in the truck's air system. If we can identify a failure in the truck's air system and send it directly for maintenance, the client would save $475, for example. That being said, it is clear that the best metric for evaluating a model is recall, because if our model predicts a false positive, the client just loses $10, but if the model predicts a false negative, the client loses $475. In other words, our model has to be good at not letting cases of failure in the truck's air system go unnoticed.

- Second, we need to look at the database sent by the client (air_system_previous_years.csv). We can conduct an Exploratory Data Analysis to understand more about the data. Since the feature names are encoded, we can't interpret specific features. We need to preprocess the dataset, such as replacing strings "na" with a real NaN, dropping columns with too many missing values, because we can't just impute them with the mean or median when there are too many missing values. Furthermore, we need to identify possible categorical features, considering variables with more than 10 unique values as categorical. After dividing the categorical and numerical variables, we should cast categorical features to string and numerical features to float. In addition, we should analyze the distribution of the target variable. If it is unbalanced, which is the case, we need to balance it so our model can learn the characteristics of both classes.

- The third step is to train the model. A pipeline was created that contains several preprocessors like imputers, scalers, over-sampling techniques, feature selection algorithms, dimensionality reduction algorithms, and the machine learning algorithms themselves. The idea of the pipeline is to make the code more professional and replicable. Additionally, the pipeline helps us avoid data leakage, so we can put the training dataset into the pipeline to train the model and predict new outputs using the pipeline with the test dataset in an automated way. The pipeline was run with various machine learning models with specific feature selection and dimensionality reduction algorithms. The goal was to find a model that fits the dataset best, and several algorithms were tested for this, including Logistic Regression, Random Forest Classifier, XGBoost Classifier, Gradient Boost Classifier, and others.

- The fourth step is to perform hyperparameter tuning of the best model chosen in the third step. For this, we used BayesSearchCV, an optimizer that can smartly choose the best parameters of a model. The Bayes optimizer was used not only to search for the best parameters of the model but also to choose the best parameters for the feature selection and dimensionality reduction algorithms. The final model returned a recall of 0.953 (this model was cross-validated).

- In the fifth step, we test our model. Some data transformations were performed, such as dropping the same columns that were dropped in the training dataset and converting the numerical features to float and categorical features to string (the exact same transformations were made in the training dataset). After this, we put our test dataset into the pipeline to make predictions. The results were good and showed that only 13 of the trucks that actually had failures in the air system were not sent immediately to maintenance. The recall of the model on the test dataset was 0.952.

- The last step is to put the model into production. For this task, the FastAPI framework was used for an API that has some microservices around the model, such as creating a table in a PostgreSQL database of the test dataset containing the true classes, the predicted classes, and the predicted probabilities calculated by the model. The endpoint that creates this can be useful for the client to look at the probability of a truck having or not having a failure in the air system. These model predictions are there to help the decision-maker. Additionally, we have another endpoint that can predict if completely new data (data without a true class) have failures or not in the air systems, returning its prediction and the predicted probability together. We also have a metrics endpoint that makes an SQL query to obtain the true positives, true negatives, false positives, false negatives, and the total cost if the decision-maker had fully followed the model's predictions for the present year. Lastly, we have the model-info endpoint, which returns the parameters of the pipeline, including the parameters of the feature selection algorithms, the dimensionality reduction algorithm, and the model itself. These last two endpoints could be a good way to identify if a model needs to be retrained.

### Question 2

Which technical data science metric would you use to solve this challenge? Ex: absolute error, rmse, etc.

- If a truck with defects in the air system is not sent directly for maintenance, the company pays $500, and if a truck is sent for maintenance but does not show any defect in this system, the company pays only $10. This indicates that if the model predicts a false positive, the company will pay just $10, but if the model predicts a false negative, the company will have to pay $500. So, with the aim of reducing the company's expenses, the model should minimize false negative predictions, and the metric that fits this problem is recall. Essentially, every false negative is equivalent to 50 false positives in cost to the company.

### Question 3

Which business metric would you use to solve the challenge?

- The business metric that should be used is the total cost that the company would incur compared to last year. On the test dataset (air_system_present_year.csv), we have to multiply the false negatives by 500, sum this result with the false positives multiplied by 10, and sum it with the true positives multiplied by 25. By doing this, we achieve a total cost of $25,140 for the present year. If we assume that they will have the same total expenditure ($37,000) as last year (2020) and that they follow all the model's predictions, they would save $11,860!

### Question 4

How do technical metrics relate to the business metrics?

- The technical metrics (recall, precision, F1-score) and the business metrics (total cost and saved money) are related when you calculate the cost that a true positive, a false positive, and a false negative incur. The precision recall metric tries to minimize the false negatives of the model; in other words, it tries to minimize the $500 expenditure. The precision metric tries to minimize the false positives of the model; in other words, it tries to minimize the $10 expenditure. Finally, the F1-score metric tries to get a harmonic mean of the two metrics, so if one of the two metrics is too low, the F1-score will decrease significantly. In the context of the problem, we should use the recall metric to train and evaluate the model because we are trying to minimize the false negatives; in other words, we are trying to minimize the $500 expenditure.

### Questions 5 until 12

As mentioned above, these questions are related to more technical concepts, so they will be answered in code. The code that contains these answers is located in the "notebooks" directory inside the "model_training_and_validation.ipynb" file.

### Question 13

What risks or precautions would you present to the customer before putting this model into production?

- New Data: The model's performance can degrade if the input data in production differs significantly from the training data in terms of quality or distribution.

- Feature Drift: Changes in the statistical properties of features over time (feature drift) can lead to decreased model performance.

- Scalability and Performance: The model may not scale well or perform efficiently under production workloads. It's important to know customer demand for the model over time.

### Question 14

If your predictive model is approved, how would you put it into production?

- If the model is approved, I would export the model using the "joblib" or "pickle" package. After this, I would create an API and a database around this model based on the customer demands.

- The main endpoints of the API would be:

    - A 'create-tables' endpoint that could create a PostgreSQL database containing all the test data sent by the customer (air_system_present_year.csv) with the true labels, the predicted labels, and the predicted probabilities. This table would be important for the decision-maker to make graphical analyses in a dashboard such as PowerBI, Qlik Sense, or Tableau.

    - A 'predict' endpoint that predicts if new data is 'neg' or 'pos' with the predicted probability.

    - A 'metrics' endpoint that returns the amount of true positives, true negatives, false positives, false negatives, and the total cost related to these predictions.

    - A 'model-info' endpoint that could return information about the model, like its recall and all the best parameters that went through the hyperparameter optimization.

    - Note: All these endpoints are located in the 'api/v1' directory.

- Create a 'requirements.txt' containing all package dependencies.

- Containerize the API. Make a Dockerfile to build an image for the web service (API) and run a docker-compose file to orchestrate the images related to the PostgreSQL database and the web service.

### Question 15

If the model is in production, how would you monitor it?

- Performance Metrics Monitoring: Use monitoring dashboards (e.g., Grafana, Kibana) to visualize performance metrics. Set up alerts to notify when performance metrics fall below a certain threshold.

- Data Drift and Concept Drift: Use libraries such as Evidently AI or River to detect and monitor drift.

- Real-time Monitoring: Use Application Performance Monitoring (APM) tools like Prometheus to monitor latency and throughput.

- Model Explainability: Monitor changes in feature importance scores over time. Tools like InterpretML provide explanations for model predictions.

### Question 16

If the model is in production, how would you know when to retrain it?

- Performance Degradation: The endpoints 'metrics' and 'model-info' could be a good way to analyze possible degradation of the model. If the model's performance is decreasing.

- User feedback about the model's predictions and predicted probabilities.

- Changes in Data Distribution: New data samples differ significantly from the training data distribution, indicating that the current model may not generalize well to new inputs.

- Data Drift and Concept Drift: Changes in the statistical properties of input features like mean or variance over time and changes in the relationship between input features and the target variable.

- Introduction of new features or changes in business requirements.

## How to run the API

First of all, you need to create an '.env' file on the root of the project based on the '.env.example' file.

### Docker option

To run it with Docker, you need to have the Docker instaled on your machine.

Then run:

```
docker compose up --build
```

When the container start, you will be able to make requests for the containerizated API!

To shut down the container, run:

```
docker compose down
```

### Run locally

First you need to create new a database on your PostgreSQl. After that put the database informations on your '.env' file that you created using the '.env.example' as example.

It's advisable that you create an virtual enviroment to avoid conflicts, so on the root of the project, run:

```
python3 -m venv .venv
```

After that, activate your enviroment:

```
source .venv/bin/activate
```

Then execute

```
pip install --no-cache-dir -r requirements.txt
```

Go to the right directory:

```
cd api/v1
```

And lastly, to run the API locally, execute:

```
python3 server.py
```

### Test the API

The Postman collection 'Machine_learning_challenge.postman_collection.json' are on the root of the project. Import the collection and try it out!


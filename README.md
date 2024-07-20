# Machine Learning Challenge

The description of the problem is write on the PDF file on the root of the project. The code to solve this specific problem was made on the "notebooks" directory. After some interpretation of the problem and some exploratory data analysys, it was made a machine learnig model that could predict, with 95% of recall, new trucks that could have failure on their air system. The PDF file containing the challenge description have 16 question that will be answer below.

Since the first 4 questions are related to the problem interpretation, they'll be answered below. The question 5 until 12 will be answered in the notebook file that is into the "notebooks" directory. And finally, questions 13 until 16 will be answered in this README file with mentions to the "api" directory, where the best model chose was put in to production.

## Challenge Activities

### Question 1

What steps would you take to solve this problem? Please describe as completely and clearly as possible all the steps that you see as essential for solving the problem.

- First we need to undestand the problem. In the last 3 years it has been noticing a large increase in the expenses related to the maintenance of the air system of its vehicles, so our job is to help our client to save money by identifying patterns on the features that may end up becoming a problem on the truck's air system. If we could identify a failure on the truck's air system and send directly for maintenance, the client would save $475, for example. That been said, it is clear that the best metric for evaluating a model is the recall, beacause if our model predicts a false positive, the client just looses $10, but if the model predicts false negative, the client looses $475, in other words, our model has to be good at not letting cases of failure in the trucks's air system go unnoticed.

- Second we need to take a look on the database sent by the client (air_system_previous_years.csv). We can make a Exploratory Data Analysis to understand more the about the data that we will be into it. Since the features' name are encoded, we can't do much interpretaion of specifics features it self. We need to do a preprocessing on the dataset, like replacing strings "na" with a real NaN, drop columns that have to much missing values, because we can't just do an input with the mean or median of the column when we have so much missing values. Furthermore we need to identify possible categorical features, it was consider categorical variables those that had more than 10 unique values. After the division of categorical variables and numerical variables, we should cast categorical features into to string and numerical features to float. In addition, we should analyse the distribuition of the target variable, if the this variabe is unbalanced, witch is the case, we need to balance it to our model learn all the caracteristics of the two classes.

- Third step is to train the model. It was made a pipeline that contains several pre-processors like inputers, scalers, over-sampling tecniques, feature selection algorithms, dimensionality reduction algorithms and the machine learning algorithms their selves. The idea of the pipeline is to make the code more professional and replicable, in addiction, the pipeline help us to avoid data leakege, so we can put the train dataset in to the pipeline to train the model and predict new outputs using the pipeline with the test dataset with in an automated way. The pipeline was run with various machine learning models with specific feature selection and dimensionality reduction algorithms. The goal was to find a model that fit more properly in the dataset, and for this was tested several algorithms, like Logistic Regression, Random Forest Classifier, XGBoost Classifier, Gradient Boost Classifier, and others.

- The fourth step is to do a hyperparameter tuning of the best model chose on the third step. For this, it was used the BayesSearchCV, an optimizator that can smartly choose the best parameters of a model. The Bayes optimizator was not only used to search the best parameters of the model, but also to choose the best parameters to the feature selection and dimensionality reduction algorithms. The final model returned a recall of 0.953 (this model was cross-validated).

- On fifth step we test our model. It was made some data tranformation like to drop the same columns that were droped on the training dataset and convert the numerical features to float and categorical features to string (the exacly same tranformation was made in the traning dataset). After this, we put our test dataset into the pipeline to make the predictions. The results were good and show that only 13 of trucks that actually had failure of the air system were not send imediatly to maintenence. The recall of the model on the test dataset was 0.952.

- The last step is to put the model in to production. For this task it was used the FastAPI framework for an API that have some microsservices around the model, like to create a table on a PostgreSQL database of the test dataset containing the true classes, the predicted classes and the predicted probabilitys calculated by the model. The endpoint that creates this can be useful for the client to look to the probability of a truck have or not a failure on the air system. This model's predictions are there to help the decision maker. In addiction, we have another enpoint that can predict if a complete new data (a data without true class) have failure or not on the air systems, returning its prediction and the predicted proba together. We also have a metrics enpoint that make a SQL query to obtain the true positives, true negatives, false positives, false negatives and the total cost if the decision maker had fully followed the model's predictions on the present year. And lastly, we have the model-info endpoint, this microsservice returns the parameters of the pipeline, wicth contains the parameters of the feature selection algorithms, the dimensionality reduction algorithm and the model itself. These two last enpoints could be a good way to identify if a model needs to be retrained.

### Question 2

Which technical data science metric would you use to solve this challenge? Ex: absolute error, rmse, etc.

- If a truck with defects in the air system is not sent directly for maintenance, the company pays $500, and if a truck is sent for maintenance, but it does not show any defect in this system, the company pays only $10. That indicates us that if the model predicts a false positive, the company will pay just $10, but if the model predicts a false negative, the company will have to pay $500. So with the aim of reducing the company's expenses, the model should minimize the false negatives predictions, and the metric that fits to this problem is the recall. Basically every false negative is equivalent to 50 false positives in cost to the company.

### Question 3

Which business metric would you use to solve the challenge?

- The business metric that should be used is the total cost that the company would have compared to the last year. On the test dataset (air_system_present_year.csv) we have to multiply the false negatives by 500, sum this result with the false positives multiplied by 10 and sum it with the true positives multiplied by 25. Doing this, we achive a total cost of $25140 for the present year. If we assume that they will have the same total expendure ($37000) of the last year (2020) and that they follow all the model's predictions, they would save $11860!

### Question 4

How do technical metrics relate to the business metrics?

- The technical metrics (recall, precison, f1-score) and the business metrics (total cost and saved money) are related when you calculate the cost that a true positive, a false positive and a false negative have. The precison recall metric tries to minimize the false negatives of the model, in other words, try to minimize the $500 expendure. The precison metric trys to minimize the false positives of the model, in other words, try to minimize the $10 expendure. And finally, the f1-score metric trys to get a harmonic mean of the two metrics, so if one of the two metrics are to low, the f1-score will decrease by a lot. On the problem's context, we should use the recall metric to train and avaliate the model, because we are trying to minimize the false negatives, in other words, we are trying to minimize the $500 expendure.

### Questions 5 until 12

As mentioned above, these questions are related with more tecnical concepts, so they will be answered on code. The code that contains these answers are located on the "notebooks" directory and inside of the "model_training_and_validation.ipynb" file.

### Question 13

What risks or precautions would you present to the customer before putting this model into production?

- New Data: The model's performance can degrade if the input data in production differs significantly from the training data in terms of quality or distribution.

- Feature Drift: Changes in the statistical properties of features over time (feature drift) can lead to decreased model performance.

- Scalability and Performance: The model may not scale well or perform efficiently under production workloads. It's important to know customer demand for the model over time.

### Question 14

If your predictive model is approved, how would you put it into production?

- If the model is approved, I would export the model using the "joblib" or "pickle" package. After this, I would make an API and a database around this model based on the costumer demands. 

- The main endpoints of the API would be:

    - A 'create-tables' endpoint that could create a PostgreSQl database containing all the test data sent by the costumer (air_system_present_year.csv) with the true labels, the predicted labels and the predicted probabilities. This table would be important for the decision-maker to make graphical analyses in a dashboard such as PowerBI, Qlik Sense or Tableau.

    - A 'predict' enpoint that predicts if a new data is 'neg' or 'pos' with the predicted proba.

    - A 'metrics' enpoint that returns the amount of true positives, true negatives, false positives, false negatives and the total cost related to these predictions.

    - A 'model-info' that could return informations about the model, like its recall and all the best parametersthat went go through the hyperparameter optimization.

    - Note: All these enpoints are located in the 'api/v1' directory.

- Create a 'requirements.txt' containig all of package dependecies

- Containerizate the API. Make a Dockerfile to build an image for the web service (API) and run a docker-compose file to orquestrate the images related to the PostgreSQL database and the web service.

### Question 15

If the model is in production, how would you monitor it?

- Performance Metrics Monitoring: Monitoring dashboards (e.g., Grafana, Kibana) to visualize performance metrics. Set up alerts to notify when performance metrics fall below a certain threshold.

- Data Drift and Concept Drift: Libraries such as Evidently AI or River to detect and monitor drift.

- Real-time Monitoring: Application Performance Monitoring (APM) tools like Prometheus can be used to monitor latency and throughput. 

- Model Explainability: Changes in feature importance scores over time. Tools like InterpretML provides explanations for model predictions.

### Question 16

If the model is in production, how would you know when to retrain it?

- Performance Degradation: The enpoint 'metrics' and model-info could be a good way to analyse a possible degradation of the model. If the model's performance is decreasing.

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


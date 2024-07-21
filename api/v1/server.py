from fastapi import FastAPI, Depends
import joblib
import numpy as np
from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.deps import get_session
from create_tables import create_tables
from schemas.api_schemas import *
from data_preparation.data_cleaning import clean_dataframe, dict_to_dataframe
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession



MODEL = joblib.load('../../aproved_models/model.pk')

app = FastAPI(title='Machine Learning Model in Production')


@app.post('/create-tables', description="Enpoint that creates a PostgreSQl database based on the air_system_present_year.csv file with predictions and predicted probas")
async def create_tables() -> dict:
    await create_tables()
    return {'message': "Test dataset with predictions are now in the database!"}


@app.get('/predict', response_model=OutputData, description="It receives a JSON representing a sample. This sample can come from a front-end application, spreadsheet, etc..")
async def predict(data: List[InputData]) -> OutputData:
    df = dict_to_dataframe(data)

    df_clean = clean_dataframe(df)
    y = None

    prediction = MODEL.predict(df_clean)
    predicted_probas = MODEL.predict_proba(df_clean)
    pos_proba = predicted_probas[:,1]  ## Predict proba will show us the probability of been 'pos'. Values above 0.5 were laballed 'pos' and below 0.5, 'neg'
    prediction = np.where(prediction == 0, 'neg', 'pos')

    return {"prediction": prediction.tolist(), "predict_proba": pos_proba.tolist()}


@app.get('/metrics', response_model=MetricsSchema, description="Return the metrics of the model. Return the predicted True positives, True negatives, False positives and False negatives, and in addiction, returns the total cost.")
async def metrics(db: AsyncSession = Depends(get_session)) -> MetricsSchema:
    response = (await db.execute(text('''
                                      SELECT COUNT(*), predicted_class, true_class 
                                      FROM air_system_present_year 
                                      GROUP BY predicted_class, true_class
                                      '''))).all()

    results = {}
    for tuple in response:
        if tuple[1] == 'pos' and tuple[2] == 'pos':
            results['true_positives'] = tuple[0]
        elif tuple[1] == 'neg' and tuple[2] == 'neg':
            results['true_negatives'] = tuple[0]
        elif tuple[1] == 'neg' and tuple[2] == 'pos':
            results['false_negatives'] = tuple[0]
        elif tuple[1] == 'pos' and tuple[2] == 'neg':
            results['false_positives'] = tuple[0]

    results['total_cost'] = calculate_all_cost(results)

    return results


@app.get('/model-info', response_model=ModelInfoSchema, description="Return the model parameters and the recall metric")
async def model_info() -> ModelInfoSchema:
    recall = MODEL.best_score_  # Recall because the model was trained to achieve the best recall
    best_params = dict(MODEL.best_params_)

    return {'best_recall': recall, 'best_pipeline_params': best_params}


def calculate_all_cost(confusion_matrix: dict) -> float:
    return (confusion_matrix['false_negatives'] * 500) + (confusion_matrix['false_positives'] * 10) + (confusion_matrix['true_positives'] * 25)
    


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1",
                port=8000, log_level='info',
                reload=True)

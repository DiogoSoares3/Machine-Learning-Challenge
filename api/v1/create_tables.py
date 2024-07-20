from core.configs import settings
from core.database import engine
from sqlalchemy import Table, Column, String, Float, text
from data_preparation.data_cleaning import clean_dataframe, CATEGORICAL_COLUMNS
import joblib
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


TABELA = None
MODEL = joblib.load('../../aproved_models/model.pk')

async def create_tables() -> None:
    print("Creating the table in the database!")

    async with engine.begin() as conn:
        await conn.run_sync(settings.DBBaseModel.metadata.drop_all)
        
        await conn.run_sync(
            lambda conn: conn.execute(text(f"DROP TABLE IF EXISTS air_system_present_year"))
        )

        import pandas as pd
        df = pd.read_csv("../../data/air_system_present_year.csv")
        x = df.drop(columns="class", axis=1)
        y = df['class']
        df_clean = clean_dataframe(x)
        
        prediction = MODEL.predict(df_clean)
        predicted_probas = MODEL.predict_proba(df_clean)
        
        df_clean['predicted_class'] = prediction
        df_clean['predicted_class'] = np.where(df_clean['predicted_class'] == 0, 'neg', 'pos')
                
        pos_proba = predicted_probas[:,1] # Predict proba will show us the probability of been 'pos', or 1
        df_clean["predicted_probas"] = pos_proba
        
        df_clean['true_class'] = y
        
        columns = []
        for col in df_clean.columns:
            if col in CATEGORICAL_COLUMNS:
                columns.append(Column(col, String, nullable=True))
            else:
                columns.append(Column(col, Float, nullable=True))
                
        tabela = Table("air_system_present_year", settings.DBBaseModel.metadata, *columns, extend_existing=True)
        
        await conn.run_sync(settings.DBBaseModel.metadata.create_all)
        
        for i, row in df_clean.iterrows():
            data = {col: None if pd.isna(row[col]) else row[col] for col in df_clean.columns}
            await conn.execute(tabela.insert().values(data))
        await conn.commit()

    print("Tables created and test data were inserted with success")


if __name__ == '__main__':
    import asyncio

    asyncio.run(create_tables())

from typing import ClassVar
from pydantic_settings import BaseSettings
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

class Settings(BaseSettings):

    """
    General configs used for the the aplication
    """

    DBBaseModel: ClassVar = declarative_base()
    
    load_dotenv()
    db_url:  ClassVar[str] = os.getenv("DB_URL")
    DB_URL: str = db_url

    class Config:
        case_sensitive = True


settings: Settings = Settings()

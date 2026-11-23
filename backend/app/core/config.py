import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    TOPICLENS_DB_URL: str
    DOCUSENSE_DB_URL: str

    class Config:
        # Look for .env in the root project folder (one level above backend/app/core)
        env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
        env_file_encoding = "utf-8"

settings = Settings()
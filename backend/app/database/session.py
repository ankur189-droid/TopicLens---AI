from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# --- TopicLens DB Setup ---
topiclens_engine = create_engine(settings.TOPICLENS_DB_URL)
SessionTopicLens = sessionmaker(autocommit=False, autoflush=False, bind=topiclens_engine)

# --- DocuSense DB Setup ---
docusense_engine = create_engine(settings.DOCUSENSE_DB_URL)
SessionDocuSense = sessionmaker(autocommit=False, autoflush=False, bind=docusense_engine)

# Dependencies for FastAPI routes
def get_topiclens_db():
    db = SessionTopicLens()
    try:
        yield db
    finally:
        db.close()

def get_docusense_db():
    db = SessionDocuSense()
    try:
        yield db
    finally:
        db.close()
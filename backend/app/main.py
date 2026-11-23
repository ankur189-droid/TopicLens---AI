from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.database.session import topiclens_engine
from app.services.model_loader import load_models
from app.api.routes.prediction import router as prediction_router
from app.api.routes import topics

from app.database.base import TopicLensBase
from app.models.prediction import PredictionHistory


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nStarting TopicLens AI backend...")

    print("\nCreating TopicLens database tables...")
    TopicLensBase.metadata.create_all(bind=topiclens_engine)
    print("TopicLens database tables ready.")

    app.state.models = load_models()

    print("\nTopicLens AI backend is ready.")
    yield

    print("\nShutting down TopicLens AI backend...")


app = FastAPI(
    title="TopicLens AI API",
    description=(
        "AI-powered text topic modelling using "
        "LDA, NMF and BERTopic."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register prediction and topic endpoints
app.include_router(prediction_router)
app.include_router(topics.router)


@app.get("/")
def root():
    return {
        "message": "Welcome to TopicLens AI API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "TopicLens AI"
    }


@app.get("/health/database")
def database_health_check():

    try:
        with topiclens_engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "connected",
            "database": "TopicLens DB",
            "message": "PostgreSQL connection successful"
        }

    except SQLAlchemyError:
        return {
            "status": "error",
            "database": "TopicLens DB",
            "message": "Could not connect to PostgreSQL"
        }
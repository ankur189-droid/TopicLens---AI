
from app.services.model_loader import load_models
from app.services.prediction_service import predict_topics


if __name__ == "__main__":

    print("Loading models...")

    models = load_models()

    sample_text = """
    NASA has announced a new space mission to study
    planets and explore the solar system. Scientists
    are researching satellite technology and space
    exploration to understand the universe.
    """

    print("\nTesting topic prediction...")

    result = predict_topics(sample_text, models)

    print("\nPREDICTION RESULTS")
    print("=" * 50)

    for model_name, prediction in result["predictions"].items():

        print(f"\n{model_name.upper()}")

        for key, value in prediction.items():
            print(f"{key}: {value}")

    print("\nPrediction completed successfully.")
    
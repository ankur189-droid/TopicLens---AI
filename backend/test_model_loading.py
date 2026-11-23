
from app.services.model_loader import load_models


if __name__ == "__main__":

    models = load_models()

    print("\nMODEL VERIFICATION")
    print("-" * 40)

    print(
        "LDA topics:",
        models["lda"]["model"].n_components
    )

    print(
        "NMF topics:",
        models["nmf"]["model"].n_components
    )

    print(
        "BERTopic topics:",
        len([
            topic
            for topic in models["bertopic"]["model"].get_topics()
            if topic != -1
        ])
    )

    print("\nAll model verification checks completed.")

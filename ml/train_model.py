# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pandas as pd
import pickle
from data import process_data
from model import train_model, inference, compute_model_metrics, \
    compute_slice_metrics


if __name__ == "__main__":

    # Add code to load in the data.
    data = pd.read_csv("../data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test
    # split.
    train, test = train_test_split(data, random_state=42, test_size=0.20,
                                   stratify=data['salary'])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
                                        train,
                                        categorical_features=cat_features,
                                        label="salary",
                                        training=True
                                        )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
                                    test,
                                    categorical_features=cat_features,
                                    label="salary",
                                    training=False,
                                    encoder=encoder,
                                    lb=lb
                                    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    # Save artifacts
    pickle.dump(model, open("../model/model.pkl", "wb"))
    pickle.dump(encoder, open("../model/encoder.pkl", 'wb'))
    pickle.dump(lb, open("../model/lb.pkl", 'wb'))

    # Scoring
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    print("precision:", precision)
    print("recall:", recall)
    print("fbeta:", fbeta)

    # on individual slices
    if os.path.exists('./slice_output.txt'):
        os.remove('./slice_output.txt')

    metrics_collection = pd.DataFrame(
                            columns=['feature', 'feature value', 'n_samples',
                                     'precision', 'recall', 'f1']
                            )
    for feat in cat_features:
        slice_metrics = compute_slice_metrics(test, feat, y_test, y_pred)
        metrics_collection = pd.concat(
                                [metrics_collection, slice_metrics],
                                axis=0
                                )

    metrics_collection.to_csv('./slice_output.txt', index=False)

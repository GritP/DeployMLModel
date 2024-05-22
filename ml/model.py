from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    algo = HistGradientBoostingClassifier(
        max_leaf_nodes=25, random_state=0, early_stopping=False
    )
    model = algo.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Runs model inferences and returns the predictions.

    Inputs
    ------
    model : sklearn object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred


def compute_slice_metrics(df, feature, y_test, y_pred):
    """
    Computes performance metrics on slices for a given categorical feature.
    A slice corresponds to one value of the categorical feature analyzed.

    Inputs
    ------
    df: pandas dataframe
        Test dataframe obtained from train-test split.
    feature: string
        Feature to analyze.
    y_test : np.array
        Known labels of test dataset.
    y_pred : np.array
        Predicted labels for test dataset.

    Returns
    ------
    metrics_df: pandas dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        f1 : float
    Each row corresponds one unique value of the feature.
    """
    slice_values = df[feature].unique().tolist()
    metrics_df = pd.DataFrame(index=slice_values,
                              columns=['feature', 'n_samples', 'precision',
                                       'recall', 'f1'])

    for val in slice_values:
        slice_mask = df[feature] == val

        slice_y = y_test[slice_mask]
        slice_preds = y_pred[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)

        metrics_df.at[val, 'feature'] = feature
        metrics_df.at[val, 'n_samples'] = len(slice_y)
        metrics_df.at[val, 'precision'] = precision
        metrics_df.at[val, 'recall'] = recall
        metrics_df.at[val, 'f1'] = fbeta

    # reorder first two columns to have feature before feature value
    metrics_df.reset_index(names='feature value', inplace=True)
    colList = list(metrics_df.columns)
    colList[0], colList[1] = colList[1], colList[0]
    metrics_df = metrics_df[colList]

    return metrics_df

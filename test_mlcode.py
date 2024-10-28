import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sample data for testing
train_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [5, 6, 7, 8],
    'quality': [3, 5, 7, 9]
})

test_data = pd.DataFrame({
    'feature1': [2],
    'feature2': [6]
})


# Evaluation metrics function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Test cases
def test_eval_metrics():
    actual = [3, 5, 7]
    predicted = [2.5, 5.5, 6.5]

    rmse, mae, r2 = eval_metrics(actual, predicted)

    assert rmse < 1.5  # Example threshold for RMSE
    assert mae < 1.0  # Example threshold for MAE
    assert r2 > 0.5  # Example threshold for R^2


def test_elasticnet_model():
    train_x = train_data.drop("quality", axis=1)
    train_y = train_data["quality"]

    model = ElasticNet(alpha=0.7, l1_ratio=0.7)
    model.fit(train_x, train_y)

    predicted_quality = model.predict(test_data)

    assert len(predicted_quality) == 1  # Ensure we get one prediction
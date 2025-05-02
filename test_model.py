import unittest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class TestModel(unittest.TestCase):
    def test_model_training(self):
        data = fetch_california_housing()
        X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.assertIsNotNone(model.coef_)

if __name__ == '__main__':
    unittest.main()
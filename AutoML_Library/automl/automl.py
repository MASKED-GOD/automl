import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib  # For saving models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename="automl_log.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Dataset:
    def __init__(self, path, split_ratio=0.2):
        try:
            self.df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.split_ratio = split_ratio
            self.preprocess_data()
            self.train, self.test = train_test_split(self.df, test_size=split_ratio, random_state=42)
        except Exception as e:
            logging.error(f"Dataset error: {e}")
            print(f"Error loading dataset: {e}")

    def preprocess_data(self):
        """Converts datetime columns into numerical features and handles missing values"""
        try:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':  # Check for text-based columns
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])  # Convert to datetime
                        self.df[f'{col}_year'] = self.df[col].dt.year
                        self.df[f'{col}_month'] = self.df[col].dt.month
                        self.df[f'{col}_day'] = self.df[col].dt.day
                        self.df.drop(columns=[col], inplace=True)  # Remove original datetime column
                    except ValueError:
                        # Handle non-datetime object columns
                        self.df[col] = self.df[col].astype('category').cat.codes
        except Exception as e:
            logging.error(f"Preprocessing error: {e}")
            print(f"Error in preprocessing: {e}")

    def visualize(self, method="hist"):
        """Generates different types of plots"""
        try:
            if method == "hist":
                self.df.hist(figsize=(10, 6))
            elif method == "heatmap":
                sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
            elif method == "pairplot":
                sns.pairplot(self.df)
            elif method == "scatter":
                for col in self.df.columns[1:]:
                    plt.scatter(self.df.iloc[:, 0], self.df[col])
                    plt.xlabel(self.df.columns[0])
                    plt.ylabel(col)
                    plt.title(f"Scatter Plot: {col}")
                    plt.show()
            else:
                print("Invalid visualization method")
                return
            plt.show()
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            print(f"Error in visualization: {e}")

class AutoML:
    def __init__(self, model_type="linear_regression", save_name="model.pkl"):
        self.model_type = model_type
        self.save_name = save_name
        self.model = None

    def train(self, dataset):
        """Trains the model and saves it"""
        X_train, y_train = dataset.train.iloc[:, :-1], dataset.train.iloc[:, -1]
        X_test, y_test = dataset.test.iloc[:, :-1], dataset.test.iloc[:, -1]

        try:
            if y_train.nunique() > 10 or y_train.dtype in [np.float64, np.float32]:
                task_type = "regression"
            else:
                task_type = "classification"

            if task_type == "regression":
                if self.model_type == "linear_regression":
                    self.model = LinearRegression()
                elif self.model_type == "ridge":
                    self.model = Ridge()
                elif self.model_type == "lasso":
                    self.model = Lasso()
                elif self.model_type == "random_forest_regressor":
                    self.model = RandomForestRegressor(n_estimators=100)
                elif self.model_type == "svr":
                    self.model = SVR()
                elif self.model_type == "gradient_boosting":
                    self.model = GradientBoostingRegressor()
                elif self.model_type == "decision_tree":
                    self.model = DecisionTreeRegressor()
                elif self.model_type == "neural_network":
                    self.model = Sequential([
                        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                        Dense(32, activation='relu'),
                        Dense(1, activation='linear')
                    ])
                    self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    self.model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
                    self.model.save(self.save_name.replace(".pkl", ".h5"))
                    print(f"Neural network model saved as {self.save_name.replace('.pkl', '.h5')}")
                    return
                else:
                    print("Invalid regression model type")
                    return

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Model MSE: {mse:.4f}")
                print(f"Model R² Score: {r2:.4f}")

                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted")
                plt.show()

            elif task_type == "classification":
                if self.model_type == "random_forest":
                    self.model = RandomForestClassifier(n_estimators=100)
                else:
                    print("Invalid classification model")
                    return

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                print(f"Model Accuracy: {acc:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d")
                plt.show()
                print(classification_report(y_test, y_pred))

            joblib.dump(self.model, self.save_name)
            print(f"Model saved as {self.save_name}")

        except Exception as e:
            logging.error(f"Model training error: {e}")
            print(f"Error training model: {e}")
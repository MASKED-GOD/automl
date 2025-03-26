import automl

if __name__ == "__main__":
    dataset = Dataset("D:\python\ML projects\AutoML_Library\automl\ITC_NS.csv")  # Replace with your dataset path
    dataset.visualize(method="scatter")

    automl = AutoML(model_type="random_forest_regressor", save_name="stock_model.pkl")
    automl.train(dataset)
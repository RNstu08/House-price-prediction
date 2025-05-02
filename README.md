```markdown
# House Price Prediction Project

## Overview

This project focuses on predicting house prices in Ames, Iowa, using the dataset from Kaggle's "House Prices: Advanced Regression Techniques" competition. It demonstrates a typical machine learning workflow, including data exploration, cleaning, feature engineering, model building (Linear Regression, Ridge, Lasso), evaluation, and deployment via an interactive Streamlit web application.

## Features

* **Exploratory Data Analysis (EDA):** Comprehensive analysis of 79 explanatory variables to understand relationships, distributions, and identify potential issues like missing values and outliers.
* **Data Cleaning:** Strategies for handling missing values based on feature type and meaning (e.g., using 'None' for missing categorical amenities, median for numerical). Outlier detection and handling.
* **Feature Engineering:** Creation of new, potentially more predictive features (e.g., `TotalSF`, `HouseAge`, `IsRemodeled`).
* **Log Transformations:** Applied to the skewed target variable (`SalePrice`) and skewed numerical features to improve model performance.
* **Categorical Encoding:** One-hot encoding applied to categorical features for model compatibility.
* **Feature Scaling:** Standardization/Robust scaling applied to numerical features.
* **Regression Modeling:** Training and evaluation of Linear Regression, Ridge (L2 Regularization), and Lasso (L1 Regularization) models.
* **Model Evaluation:** Using Root Mean Squared Error (RMSE) on the log-transformed target variable, along with cross-validation for robust assessment.
* **Model Persistence:** Saving the best-performing model, the scaler, and the feature columns using `pickle`.
* **Interactive Web App:** A Streamlit application (`app/streamlit_app.py`) for users to input house features and receive a price prediction.
* **Version Control:** Project managed using Git and hosted on GitHub.

## Project Structure

```text
house-price-prediction/
├── app/                  # Contains the Streamlit application code
│   └── streamlit_app.py  # The main Streamlit app script
├── data/                 # Raw data files (needs to be downloaded from Kaggle)
│   ├── train.csv         # Training dataset
│   └── test.csv          # Test dataset
├── models/               # Contains the saved model, scaler, and columns
│   ├── house_price_model.pkl # Example: Saved Ridge model
│   ├── scaler.pkl            # Saved fitted scaler (e.g., RobustScaler)
│   └── feature_columns.pkl   # Saved list of feature column names model expects
├── notebooks/            # Jupyter notebooks for exploration and modeling
│   └── data_analysis_and_modeling.ipynb # Main notebook
├── utils/                  # Optional: Utility scripts (e.g., for preprocessing functions)
│   └── preprocessing.py
├── venv/                 # Virtual environment files (ignored by Git)
├── .gitignore            # Specifies intentionally untracked files
├── README.md             # This file: project explanation and structure
└── requirements.txt      # Project dependencies (needs to be created/updated)
```
*(Note: The `data/` folder will be empty initially in the repository; need to download the data separately as instructed below).*

## Workflow & Methodology

This project was developed following these stages:

1.  **Project Setup:**
    * Created project directory (`house-price-prediction`).
    * Set up a Python virtual environment (`venv`) to isolate dependencies.
    * Installed necessary libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`, `streamlit`) using `pip`.
    * Generated `requirements.txt` using `pip freeze`.

2.  **Dataset Acquisition:** Downloaded `train.csv` and `test.csv` from the [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and placed them in the `data/` folder.

3.  **Exploratory Data Analysis (EDA):** Conducted in the Jupyter Notebook (`notebooks/01_...ipynb`):
    * Loaded data with pandas.
    * Inspected shape, data types, basic statistics (`.info()`, `.describe()`).
    * Analyzed the target variable `SalePrice`, noted its right skew, and applied a log transformation (`np.log1p`) creating `SalePriceLog`.
    * Identified and visualized missing data patterns.
    * Explored correlations between numerical features and the log-transformed target using `.corr()` and heatmaps/bar plots.
    * Investigated potential outliers using scatter plots (e.g., `GrLivArea` vs `SalePriceLog`).
    * Examined relationships between key categorical features (e.g., `OverallQual`) and the target using boxplots.

4.  **Data Cleaning & Feature Engineering:** Also performed in the notebook:
    * Handled potential outliers identified during EDA (e.g., removing extreme `GrLivArea` points).
    * Imputed missing values using appropriate strategies (e.g., 'None' for missing categorical amenities like `PoolQC`, 0 for missing basement/garage numericals indicating absence, median/neighborhood median for others like `LotFrontage`).
    * Engineered new features like `TotalSF`, `HouseAge`, `IsRemodeled`, potentially others like `TotalBath`, `HasPool`, etc.
    * Applied log transformation (`np.log1p`) to highly skewed numerical features identified via `.skew()`.
    * Dropped the original `SalePrice` and `Id` columns.
    * Performed one-hot encoding on categorical features using `pd.get_dummies()`.
    * Stored the final list of feature column names derived after all preprocessing.

5.  **Model Building & Evaluation:**
    * Split the processed data into training and validation sets (`train_test_split`).
    * Applied feature scaling (`StandardScaler` or `RobustScaler`), fitting *only* on the training data and transforming both train and validation sets.
    * Instantiated and trained Linear Regression, Ridge, and Lasso models on the scaled training data (`.fit(X_train_scaled, y_train)` where `y_train` is the log-transformed target).
    * Evaluated models on the scaled validation set using Root Mean Squared Error (RMSE) calculated on the log-transformed predictions.
    * Used k-fold cross-validation (`cross_val_score` with a pipeline including scaling) for a more robust estimate of model performance.

6.  **Model Saving:**
    * Selected the best model based on evaluation (e.g., Ridge).
    * Saved the trained model object, the scaler object (refitted on the entire training set `X`), and the list of feature column names to the `models/` directory using `pickle`.

7.  **Streamlit Application Development:**
    * Created `app/streamlit_app.py`.
    * Loaded the saved model, scaler, and feature columns list.
    * Designed a user interface with Streamlit widgets (`st.slider`, `st.number_input`, `st.selectbox`) to collect input for a subset of key house features.
    * Implemented logic to:
        * Construct a dictionary/DataFrame from user inputs.
        * Perform the *exact same* feature engineering (e.g., calculating `TotalSF`, `HouseAge`) and preprocessing (log-transforming specific inputs, handling categoricals to match one-hot encoding structure) as done in the notebook.
        * Create a final input DataFrame containing *all* expected feature columns (matching the saved list), filling missing columns appropriately.
        * Scale the processed input DataFrame using the loaded scaler's `.transform()` method.
        * Make a prediction using the loaded model's `.predict()` method (which outputs on the log scale).
        * Inverse-transform the prediction (`np.expm1()`) to get the estimated price in original dollar terms.
        * Display the formatted prediction to the user.
        * Included debugging for potential errors during setup and prediction (e.g., the `.astype` error fix).

## Technologies & Libraries Used

* Python 3.x
* pandas
* NumPy
* scikit-learn (for preprocessing, modeling, evaluation)
* Matplotlib & Seaborn (for visualization)
* Streamlit (for the web app)
* Jupyter Notebook/Lab (for development)
* Pickle (for saving/loading model artifacts)
* Git & GitHub (for version control)

## Setup and Installation

Follow these steps to set up and run the project locally:

1.  **Clone the Repository:**
    ```bash
    # Replace YOUR_USERNAME with the actual GitHub username
    git clone [https://github.com/RNstu08/House-price-prediction.git](https://github.com/RNstu08/House-price-prediction.git)
    cd house-price-prediction
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create (use python3 if needed)
    python -m venv venv

    # Activate
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    * **(Ensure `requirements.txt` is present and up-to-date in the repository)**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Dataset:**
    * Download `train.csv` and `test.csv` from the [Kaggle House Prices Competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
    * Place the downloaded CSV files into the `data/` directory within the project folder.

## Usage

1.  **Run the Streamlit Web Application:**
    * Activate your virtual environment (`venv`).
    * From the project root directory, run:
        ```bash
        streamlit run app/streamlit_app.py
        ```
    * Access the app via the local URL provided in your terminal. Use the sidebar to input features and click "Predict House Price".

2.  **Explore the Jupyter Notebook:**
    * Activate your virtual environment (`venv`).
    * Ensure `jupyter` is installed (`pip install jupyterlab` or `pip install notebook`).
    * From the project root directory, run:
        ```bash
        jupyter lab
        # or
        jupyter notebook
        ```
    * Navigate into the `notebooks/` folder in the Jupyter interface and open `data_analysis_and_modeling.ipynb`.

## Model Artifacts

The `models/` directory contains:

* `house_price_model.pkl`: The pre-trained regression model (e.g., Ridge).
* `scaler.pkl`: The pre-fitted scaler (e.g., RobustScaler) used on the training data.
* `feature_columns.pkl`: A Python list containing the exact column names (features) the model and scaler expect, in the correct order.

## Future Improvements (Optional)

* Implement more advanced feature engineering (interactions, polynomials).
* Experiment with Gradient Boosting models (XGBoost, LightGBM).
* Perform systematic hyperparameter tuning (e.g., `GridSearchCV` for `alpha` in Ridge/Lasso).
* Refactor preprocessing logic into reusable functions (e.g., in `utils/preprocessing.py`).
* Improve the Streamlit app UI/UX, possibly allowing input for more features or showing feature importance.
* Add more robust error handling and input validation to the Streamlit app.

**Why did you choose Ridge/Lasso over Linear Regression?**

Ridge and Lasso add regularization terms to the loss function, which helps prevent overfitting by penalizing large coefficients. Lasso can also perform feature selection by shrinking some coefficients to zero.

**How did you handle multicollinearity?**

By using regularization techniques like Ridge and Lasso, which can mitigate the effects of multicollinearity by penalizing large coefficients.

**Why did you apply log transformation to the target variable?**

The SalePrice distribution was right-skewed. Applying a log transformation helps normalize the distribution, leading to better model performance.

**How did you handle missing values?**

For numerical features, missing values were filled with the median. For categorical features, missing values were filled with 'None' or the mode, depending on the context.

**How did you select features for the model?**

Features were selected based on correlation analysis, domain knowledge, and their impact on model performance during experimentation.

```
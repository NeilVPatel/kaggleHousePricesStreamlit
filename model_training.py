import streamlit as st

def display_model_training():
    st.title('Model Training and Evaluation')

    # Explanation of Model Training and Evaluation Steps
    st.subheader('Model Training and Evaluation Steps')
    st.markdown("""
    In this section, I outline the steps taken to train and evaluate the model. The following steps were performed:
    
    1. **Split and Scale Data:**
        - Split the data into training and test sets.
        - Scale the features using `StandardScaler`.
        - Log-transform the target variable to reduce skewness.
    
    2. **Hyperparameter Tuning with Optuna:**
        - Define an objective function to optimize the CatBoost model's hyperparameters using Optuna.
        - Optimize the hyperparameters based on RMSE.

    3. **Train the Model with Best Parameters:**
        - Retrieve the best hyperparameters from Optuna.
        - Train a new CatBoost model using the best hyperparameters.
    
    """)

    # Code for Model Training and Evaluation in an expandable field
    with st.expander("Show Model Training and Evaluation Code"):
        st.code("""
# Split and Scale Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create Training and test set for modelling
X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, y_train,
                                                    test_size=0.20,
                                                    random_state=314)

# Define a scaler
names = X_train.columns
X_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(X_scaler.fit_transform(X_train), columns=names)
X_test_scaled = pd.DataFrame(X_scaler.transform(X_test), columns=names)

# Log Target
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# Hyperparameter Tuning with Optuna
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'random_strength': trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10.0),
        'loss_function': 'RMSE',
    }
    model = CatBoostRegressor(**params, verbose=0, random_state=314)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_rmse = study.best_trial.value
print(f"Best RMSE: {best_rmse:.4f}")
print('Best trial parameters:', study.best_trial.params)

# Retrieve the best parameters
best_params = study.best_trial.params

# Specify 'loss_function'
if 'loss_function' not in best_params:
    best_params['loss_function'] = 'RMSE'

# Create a new model instance with the best parameters
model = CatBoostRegressor(**best_params, verbose=0, random_state=314)
        """, language='python')
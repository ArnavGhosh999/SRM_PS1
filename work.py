import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
import shap
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

def load_and_preprocess_data(train_files, test_file):
    """Load and preprocess training and testing data"""
    print(f"Loading {len(train_files)} training files and 1 test file...")

    # Load training data
    train_dfs = []
    for file in train_files:
        df = pd.read_csv(file)
        print(f"Loaded {file} with shape: {df.shape}")
        train_dfs.append(df)

    # Combine training data
    train_data = pd.concat(train_dfs, ignore_index=True)
    print(f"Combined training data shape: {train_data.shape}")

    # Load test data
    test_data = pd.read_csv(test_file)
    print(f"Test data shape: {test_data.shape}")

    # Ensure consistent columns between train and test
    common_cols = set(train_data.columns).intersection(set(test_data.columns))
    print(f"Number of common columns: {len(common_cols)}")

    # Take only common columns
    train_data = train_data[list(common_cols)]
    test_data = test_data[list(common_cols)]

    # Identify target column - assuming last column is target
    target_column = train_data.columns[-1]
    print(f"Target column identified as: {target_column}")

    # Separate features and target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Handle categorical features
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns: {list(categorical_columns)}")

    # Apply label encoding for categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        # Get all unique values from both train and test
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
        # Fit the encoder on all possible values
        le.fit(all_values)
        # Transform both train and test data
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Handle numeric features
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numeric columns: {list(numeric_columns)[:5]}...")

    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    # Handle missing values
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    # Encode target variable
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)
    y_test = target_encoder.transform(y_test)
    print(f"Target classes: {target_encoder.classes_}")

    # Check if classification or regression
    if len(np.unique(y_train)) < 10:
        task = 'classification'
        print(f"Task identified as classification with {len(np.unique(y_train))} classes")
        if len(np.unique(y_train)) == 2:
            objective = 'binary:logistic'
            print("Using binary:logistic objective")
        else:
            objective = 'multi:softprob'
            print("Using multi:softprob objective")
    else:
        task = 'regression'
        objective = 'reg:squarederror'
        print("Task identified as regression")

    return X_train, X_test, y_train, y_test, task, objective, target_column

def create_dataset_visualizations(X_train, y_train, X_test, y_test, target_column):
    """Create visualizations for the datasets"""
    print("Creating dataset visualizations...")
    visualizations = []

    # 1. Feature distributions in training data
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    feature_cols = X_train.columns[:6]  # Take first 6 features for visualization
    for i, col in enumerate(feature_cols):
        sns.histplot(X_train[col], kde=True, ax=axs[i//3, i%3])
        axs[i//3, i%3].set_title(f'Distribution of {col}')
    plt.tight_layout()
    visualizations.append(('feature_distributions', fig))

    # 2. Target distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(np.unique(y_train)) < 10:  # For classification
        sns.countplot(x=y_train, ax=ax)
        ax.set_title(f'Distribution of Target Classes: {target_column}')
    else:  # For regression
        sns.histplot(y_train, kde=True, ax=ax)
        ax.set_title(f'Distribution of Target Variable: {target_column}')
    plt.tight_layout()
    visualizations.append(('target_distribution', fig))

    # 3. Compare train and test distributions for a few features
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    for i, col in enumerate(X_train.columns[:4]):  # First 4 features
        sns.kdeplot(X_train[col], ax=axs[i//2, i%2], label='Train')
        sns.kdeplot(X_test[col], ax=axs[i//2, i%2], label='Test')
        axs[i//2, i%2].set_title(f'Train vs Test: {col}')
        axs[i//2, i%2].legend()
    plt.tight_layout()
    visualizations.append(('train_test_comparison', fig))

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = X_train.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                vmax=1, vmin=-1, center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    visualizations.append(('correlation_heatmap', fig))

    # 5. Pairplot for a subset of features
    if X_train.shape[1] > 5:
        # If we have many features, just take a subset for pairplot
        subset_features = list(X_train.columns[:4])
        plot_data = X_train[subset_features].copy()
        plot_data['target'] = y_train  # y_train is already a numpy array

        fig = sns.pairplot(plot_data, hue='target', height=2.5, diag_kind='kde')
        fig.fig.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16)
        visualizations.append(('pairplot', fig.fig))

    return visualizations

def train_xgboost_model(X_train, y_train, task, objective):
    """Train XGBoost model with hyperparameter tuning"""
    print("Training XGBoost model with hyperparameter tuning...")

    # Define parameters for hyperparameter search
    if task == 'classification':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
    else:  # regression
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.5, 1, 1.5]
        }

    xgb_model = xgb.XGBClassifier(objective=objective, random_state=42) if task == 'classification' \
                else xgb.XGBRegressor(objective=objective, random_state=42)

    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=10,  # Try 10 parameter combinations
        scoring='accuracy' if task == 'classification' else 'neg_mean_squared_error',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_}")

    # Train final model with best parameters
    best_model = search.best_estimator_
    
    # Create evaluation sets for final training
    eval_set = [(X_train, y_train)]
    best_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Return the trained model
    return best_model, search.best_params_

def evaluate_model(model, X_train, y_train, X_test, y_test, task):
    """Evaluate the model and create performance visualizations"""
    print("Evaluating model and creating performance visualizations...")
    visualizations = []

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # For classification, also get prediction probabilities
    if task == 'classification':
        if len(np.unique(y_train)) == 2:  # Binary classification
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:  # Multi-class
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)

    # Print evaluation metrics
    if task == 'classification':
        print("\nClassification Report (Train):")
        print(classification_report(y_train, y_train_pred))

        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_test_pred))

        # 6. Confusion Matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix (Train)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix (Test)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        plt.tight_layout()
        visualizations.append(('confusion_matrices', fig))

        # 7. ROC Curve (only for binary classification)
        if len(np.unique(y_train)) == 2:
            fig, ax = plt.subplots(figsize=(10, 6))

            # ROC curve for training data
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
            roc_auc_train = auc(fpr_train, tpr_train)
            ax.plot(fpr_train, tpr_train, lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')

            # ROC curve for test data
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
            roc_auc_test = auc(fpr_test, tpr_test)
            ax.plot(fpr_test, tpr_test, lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')

            # Plot the diagonal line
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc="lower right")
            plt.tight_layout()
            visualizations.append(('roc_curve', fig))

            # 8. Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(10, 6))

            # PR curve for training data
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
            ax.plot(recall_train, precision_train, lw=2, label=f'Train PR Curve')

            # PR curve for test data
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
            ax.plot(recall_test, precision_test, lw=2, label=f'Test PR Curve')

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="best")
            plt.tight_layout()
            visualizations.append(('precision_recall_curve', fig))

    else:  # Regression
        # Calculate regression metrics
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        print(f"\nRegression Metrics (Train): MSE = {mse_train:.4f}, R² = {r2_train:.4f}")
        print(f"Regression Metrics (Test): MSE = {mse_test:.4f}, R² = {r2_test:.4f}")

        # 6. Actual vs Predicted Values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.scatter(y_train, y_train_pred, alpha=0.5)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Actual vs Predicted (Train)\nMSE = {mse_train:.4f}, R² = {r2_train:.4f}')

        ax2.scatter(y_test, y_test_pred, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title(f'Actual vs Predicted (Test)\nMSE = {mse_test:.4f}, R² = {r2_test:.4f}')

        plt.tight_layout()
        visualizations.append(('actual_vs_predicted', fig))

        # 7. Residual Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        train_residuals = y_train - y_train_pred
        ax1.scatter(y_train_pred, train_residuals, alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted (Train)')

        test_residuals = y_test - y_test_pred
        ax2.scatter(y_test_pred, test_residuals, alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted (Test)')

        plt.tight_layout()
        visualizations.append(('residual_plots', fig))

        # 8. Residual Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        sns.histplot(train_residuals, kde=True, ax=ax1)
        ax1.set_title('Residual Distribution (Train)')
        ax1.set_xlabel('Residual Value')

        sns.histplot(test_residuals, kde=True, ax=ax2)
        ax2.set_title('Residual Distribution (Test)')
        ax2.set_xlabel('Residual Value')

        plt.tight_layout()
        visualizations.append(('residual_distribution', fig))

    # 9. Feature Importance
    fig, ax = plt.subplots(figsize=(14, 10))
    xgb.plot_importance(model, ax=ax, height=0.8)
    ax.set_title('Feature Importance (Weight)')
    plt.tight_layout()
    visualizations.append(('feature_importance_weight', fig))

    # 10. Feature Importance (Gain)
    fig, ax = plt.subplots(figsize=(14, 10))
    xgb.plot_importance(model, ax=ax, height=0.8, importance_type='gain')
    ax.set_title('Feature Importance (Gain)')
    plt.tight_layout()
    visualizations.append(('feature_importance_gain', fig))

    # 11. Learning Curves from Cross-Validation
    try:
        results = model.evals_result()
        if results:
            epochs = len(results['validation_0']['logloss' if task == 'classification' else 'rmse'])
            x_axis = range(0, epochs)

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot log loss or RMSE for training and validation
            metric = 'logloss' if task == 'classification' else 'rmse'

            ax.plot(x_axis, results['validation_0'][metric], label='Train')
            if 'validation_1' in results:
                ax.plot(x_axis, results['validation_1'][metric], label='Test')
            ax.legend()
            ax.set_ylabel(f'{"Log Loss" if task == "classification" else "RMSE"}')
            ax.set_xlabel('Number of Boosting Rounds')
            ax.set_title(f'XGBoost {"Log Loss" if task == "classification" else "RMSE"}')
            plt.tight_layout()
            visualizations.append(('learning_curves', fig))
    except Exception as e:
        print(f"Could not create learning curves: {e}")

    # 12. SHAP Values Summary Plot
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for a sample of training data
        sample_size = min(500, X_train.shape[0])  # Use at most 500 samples to keep it manageable
        X_sample = X_train.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)

        if task == 'classification' and len(np.unique(y_train)) > 2:
            # For multi-class, pick the first class for visualization
            shap_values = shap_values[0]

        # Create the SHAP summary plot
        fig = plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        visualizations.append(('shap_feature_importance', fig))

        # 13. SHAP Dependence Plots for top features
        if task == 'classification' and len(np.unique(y_train)) <= 2:
            feature_importance = np.abs(shap_values).mean(0)
            indices = feature_importance.argsort()[-3:]  # Top 3 features

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            for i, idx in enumerate(indices):
                feature_name = X_train.columns[idx]
                shap.dependence_plot(idx, shap_values, X_sample, ax=axs[i], show=False)
                axs[i].set_title(f'SHAP Dependence: {feature_name}')
            plt.tight_layout()
            visualizations.append(('shap_dependence_plots', fig))
    except Exception as e:
        print(f"SHAP visualization failed: {e}")

    return visualizations

def save_model_and_visualizations(model, visualizations, model_params, target_column):
    """Save model and visualizations to disk"""
    print("Saving model and visualizations...")

    # Create a results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"xgboost_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model
    model.save_model(f"{results_dir}/xgboost_model.json")

    # Save model parameters
    with open(f"{results_dir}/model_parameters.txt", "w") as f:
        f.write(f"Target: {target_column}\n\n")
        f.write("Model Parameters:\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")

    # Save visualizations
    for name, fig in visualizations:
        fig.savefig(f"{results_dir}/{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Results saved to directory: {results_dir}")
    return results_dir

def run_xgboost_pipeline(train_files, test_file):
    """Run the entire XGBoost pipeline"""
    print("Starting XGBoost pipeline...")

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, task, objective, target_column = load_and_preprocess_data(train_files, test_file)

    # 2. Create dataset visualizations
    dataset_visualizations = create_dataset_visualizations(X_train, y_train, X_test, y_test, target_column)

    # 3. Train XGBoost model
    model, best_params = train_xgboost_model(X_train, y_train, task, objective)

    # 4. Evaluate model and create performance visualizations
    performance_visualizations = evaluate_model(model, X_train, y_train, X_test, y_test, task)

    # 5. Save model and visualizations
    all_visualizations = dataset_visualizations + performance_visualizations
    results_dir = save_model_and_visualizations(model, all_visualizations, best_params, target_column)

    print(f"\nXGBoost pipeline completed successfully!")
    print(f"Saved {len(all_visualizations)} visualizations and model to {results_dir}")

    return model, results_dir

if __name__ == "__main__":
  
    train_files = [
        "/media/umeshgjh/New Volume/Energy7 Internship/Merge thing n check/merged_data_100Hz.csv"
    ]
    test_file = "Test - Zone.csv"

    model, results_dir = run_xgboost_pipeline(train_files, test_file)
import pandas as pd
import numpy as np
from models.clustering.kmeans import KMeans
from typing import Dict, List, Union, Callable, Tuple, Optional
from models.regression.base import Model


def evaluate_model(model, X_test, y_test, metrics=None, inv_transform_pred=None, y_pred: Optional[pd.Series] = None):
    """
    Evaluates the model performance using specific metrics.
    
    Parameters:
    -----------
    model : Model
        Trained model instance
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    metrics : list, default=None
        List of metric functions to calculate
    inv_transform_pred : callable, default=None
        Function to transform predictions back to original scale
        
    Returns:
    --------
    dict
        Dictionary with metric values
    """
    if metrics is None:
        metrics = [mse_score, r2_score]
    
    
    y_pred_test = model.predict(X_test) if y_pred is None else y_pred

    if inv_transform_pred is not None:
        y_pred_test = inv_transform_pred(y_pred_test)

        y_test = inv_transform_pred(y_test)

    results = {}
    for metric in metrics:
        results[metric.__name__.lower()] = metric(y_pred_test, y_test)
            
    return results


def preprocess_data(df, save_path=None):
    """
    Preprocesses real estate data by converting sqft units to m2.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with the data to preprocess.
    save_path : str, default=None
        Path to save the preprocessed data.
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    # verify that the 'area_units' column exists
    if 'area_units' not in df.columns:
        raise ValueError("The 'area_units' column does not exist in the DataFrame.")
    # convert sqft to m2
    df.loc[df['area_units'] == 'sqft', 'area'] = df['area'] * 0.092903  
    df['area_units'] = 'm2'  
    df = df.drop('area_units', axis=1)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def mse_score(y_pred, y_test, round=False):
    """
    Calculates the mean squared error (MSE) between the prediction and the target.
    If y_test is a DataFrame (for multivariate prediction), it calculates the average MSE of all columns.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted values.
    y_test : array-like or pd.DataFrame
        Actual values.
    round : bool, optional
        Whether to round y_pred before calculating the error.
    
    Returns:
    --------
    float
        MSE value.
    """
    if round:
        y_pred = np.round(y_pred)  # rounds the predicted values if requested
    if isinstance(y_test, pd.DataFrame):
        # convert y_pred to DataFrame for multivariate calculations
        y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)
        mse = ((y_test - y_pred) ** 2).mean().mean()
    else:
        mse = ((y_test - y_pred) ** 2).mean()
    return mse


def r2_score(y_pred, y_test):
    """
    Calculates the coefficient of determination R^2 of the prediction.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted values.
    y_test : array-like
        Actual values.
    
    Returns:
    --------
    float
        R^2 value.
    """
    y_mean = y_test.mean()  # mean of actual values
    ss_total = ((y_test - y_mean) ** 2).sum()  # total sum of squares
    ss_res = ((y_test - y_pred) ** 2).sum()      # residual sum of squares
    r2 = 1 - ss_res / ss_total
    return r2


def round_input(X, columnas):
    """
    Rounds the specified columns of a DataFrame to integers.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input DataFrame.
    columnas : list
        List of column names to round.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with rounded columns.
    """
    X_copy = X.copy()
    for col in columnas:
        if col in X_copy.columns:
            X_copy[col] = X_copy[col].round(0).astype(int)  # round and convert to integer
    return X_copy


def get_nan_features(df: pd.DataFrame):
    """
    Finds columns with NaN values and returns a dictionary with counts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze.
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and NaN counts as values.
    """
    nan_features = df.isnull().sum()  # counts NaNs by column
    nan_features = nan_features[nan_features > 0]  # filters only columns with at least one NaN
    nan_features = nan_features.to_dict()
    return nan_features


def split_test_train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_target: bool = True,
    transform_target: Callable = None,
    normalize: bool = False
) -> tuple:
    """
    Splits the data into training and test sets, with normalization option.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature set.
    y : pd.Series
        Target variable.
    test_size : float
        Test set proportion.
    random_state : int
        Seed for reproducibility.
    drop_target : bool
        Indicates whether to remove the target column from X.
    transform_target : Callable, optional
        Function to transform the target variable.
    normalize : bool
        Indicates whether to normalize the data using X_train mean and std.
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if drop_target:
        # removes the target column from X if present
        if isinstance(y, pd.Series) and y.name in X.columns:
            X = X.drop(columns=[y.name])
        elif isinstance(y, pd.DataFrame):
            for col in y.columns:
                if col in X.columns:
                    X = X.drop(columns=[col])
    
    np.random.seed(random_state)  # sets the seed for reproducibility
    n = len(X)
    n_test = int(n * test_size)
    n_train = n - n_test
    idx = np.random.permutation(n)  # generates a random permutation of indices
    X = X.iloc[idx]
    y = y.iloc[idx]
    
    if transform_target:
        y = transform_target(y)
    
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]
    
    if normalize:
        # calculates mean and standard deviation from X_train
        train_mean = X_train.mean()
        train_std = X_train.std()
        train_std = train_std.replace(0, 1)  # avoids division by zero in constant features
        # normalizes X_train and X_test using training statistics
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        # stores normalization parameters in DataFrame attributes
        X_train.attrs['normalization'] = {'mean': train_mean, 'std': train_std}
        X_test.attrs['normalization'] = {'mean': train_mean, 'std': train_std}
    
    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    target_column,
    df=None,
    data_path=None,
    feature_columns=None,
    transform_target=None
):
    """
    Loads data from a DataFrame or CSV file and prepares features and target variable.
    
    Parameters:
    -----------
    target_column : str
        Name of the target variable column.
    df : pd.DataFrame, default=None
        DataFrame with data.
    data_path : str, default=None
        Path to CSV file if df is not provided.
    feature_columns : list, default=None
        List of columns to use as features. If None, all columns except the target are used.
    transform_target : callable, default=None
        Function to transform the target variable.
        
    Returns:
    --------
    tuple
        (X, y, feature_columns)
    """
    # loads DataFrame from file if not provided
    df = pd.read_csv(data_path) if df is None else df.copy()
    
    if transform_target:
        df[target_column] = transform_target(df[target_column])
    
    if feature_columns is None:
        # selects all columns except the target column
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y, feature_columns


def _calculate_normalization_params(data: pd.DataFrame) -> Dict:
    """
    Calculates normalization parameters (mean and standard deviation) from the data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to calculate normalization parameters.
        
    Returns:
    --------
    dict
        Dictionary with keys 'mean' and 'std'.
    """
    binary_cols = data.apply(lambda x: set(x.unique()) <= {0, 1})
    # selects only non-binary columns
    non_binary_data = data.loc[:, ~binary_cols]
    
    # initializes default parameters for all columns
    params = {
        'mean': pd.Series(0.0, index=data.columns, dtype=float),
        'std': pd.Series(1.0, index=data.columns, dtype=float)
    }
    
    # updates parameters only for non-binary columns
    if not non_binary_data.empty:
        means = non_binary_data.mean()
        stds = non_binary_data.std().replace(0, 1.0)  # avoids division by zero
        for col in non_binary_data.columns:
            params['mean'][col] = means[col]
            params['std'][col] = stds[col]
    
    return params


def _apply_normalization(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Applies normalization to the data using the given parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to normalize.
    params : dict
        Dictionary with keys 'mean' and 'std'.
        
    Returns:
    --------
    pd.DataFrame
        Normalized data.
    """
    # normalizes by subtracting the mean and dividing by the standard deviation
    return (data - params['mean']) / params['std']


def normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Normalizes features using statistics calculated from the training data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training data.
    X_test : pd.DataFrame
        Test data.
        
    Returns:
    --------
    tuple
        (X_train_normalized, X_test_normalized, normalization parameters)
    """
    params = _calculate_normalization_params(X_train)
    X_train_normalized = _apply_normalization(X_train, params)
    X_test_normalized = _apply_normalization(X_test, params)
    return X_train_normalized, X_test_normalized, params


def _handle_feature_engineering(df: pd.DataFrame, operations: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Applies feature engineering operations and collects statistics of the new features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to apply operations on.
    operations : List[Dict]
        List of dictionaries with keys 'name' and 'operation'.
        
    Returns:
    --------
    Tuple
        (new_features_df, feature_stats) where new_features_df is a DataFrame with the new features and feature_stats is a dictionary with statistics.
    """
    new_features = {}
    feature_stats = {}
    
    for op in operations:
        feature_name = op['name']
        operation = op['operation']
        # applies the operation: if callable it executes, otherwise it evaluates the expression
        new_features[feature_name] = operation(df) if callable(operation) else df.eval(operation)
        # collects basic statistics of the new feature
        feature_stats[feature_name] = {
            'mean': new_features[feature_name].mean(),
            'std': new_features[feature_name].std(),
            'min': new_features[feature_name].min(),
            'max': new_features[feature_name].max()
        }
    
    return pd.DataFrame(new_features), feature_stats


def process_dataset(
    df: pd.DataFrame,
    kmeans_model: KMeans,
    feature_engineering_ops: List[Dict[str, Union[str, Callable]]] = None,
    features_to_impute: List[str] = ['age', 'rooms'],
    location_columns: List[str] = ['lat', 'lon'],
    impute_by_zone: bool = False,
    save_path: str = None,
    create_zone_interactions: bool = False,
    zone_interaction_features: List[str] = ['area', 'rooms', 'age', 'has_pool', 'is_house']
) -> Dict:
    """
    Processes a dataset by applying clustering and feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to process.
    kmeans_model : KMeans
        KMeans model to cluster locations.
    feature_engineering_ops : List[Dict], optional
        Operations for feature engineering.
    features_to_impute : List[str]
        Features that require imputation of missing values.
    location_columns : List[str]
        Columns containing location data.
    impute_by_zone : bool
        If True, impute missing values using the mean by zone.
    save_path : str, optional
        Path to save the processed DataFrame.
    create_zone_interactions : bool
        If True, create interaction features between zone and other features.
    zone_interaction_features : List[str]
        Features to interact with the zone.
        
    Returns:
    --------
    Dict
        Dictionary with:
            - 'df': Processed DataFrame without location columns.
            - 'df_pos': DataFrame with location columns and zone.
            - 'zone_stats': Statistics by zone (if locations are used).
            - 'feature_stats': Statistics of the new features.
    """
    df = df.copy()
    
    if location_columns:
        # groups locations using kmeans
        location_data = df[location_columns].to_numpy()  
        kmeans_model.fit(location_data)
        df['location_zone'] = kmeans_model.predict(location_data)
        
        # calculates statistics by zone for possible imputation
        zone_stats = {}
        for zone in range(kmeans_model.n_clusters):
            df_zone = df[df['location_zone'] == zone]
            stats = {feat: {
                'mean': df_zone[feat].mean(),
                'median': df_zone[feat].median(),
                'std': df_zone[feat].std(),
                'missing': df_zone[feat].isna().sum()
            } for feat in features_to_impute}
            stats.update({
                'size': len(df_zone),
                'percentage': len(df_zone) / len(df) * 100,
                'center': kmeans_model.cluster_centers_[zone]  # centroid of the zone
            })
            zone_stats[f'zone_{zone}'] = stats
            
            # imputes missing values by zone if requested
            if impute_by_zone:
                for feature in features_to_impute:
                    df.loc[df['location_zone'] == zone, feature] = df.loc[
                        df['location_zone'] == zone, feature
                    ].fillna(stats[feature]['mean'])
    else:
        zone_stats = None
    
    # global imputation if not imputing by zone
    if not impute_by_zone and features_to_impute:
        for feature in features_to_impute:
            global_mean = df[feature].mean()
            df[feature] = df[feature].fillna(global_mean)
    
    # defines default operations if not provided
    if feature_engineering_ops is None:
        feature_engineering_ops = [
            {'name': 'pool_house', 'operation': lambda df: df['has_pool'] * df['is_house']},
            {'name': 'house_area', 'operation': lambda df: df['area'] * df['is_house']},
            {'name': 'dist_to_cluster_center', 'operation': lambda df: 
                np.array([
                    # calculates the Euclidean distance from each record's location to its zone centroid
                    np.linalg.norm(
                        df.loc[i, location_columns].values - kmeans_model.cluster_centers_[df.loc[i, 'location_zone'].astype(int)]
                    )
                    for i in df.index
                ])
            }
        ]
    
    new_features_df, feature_stats = _handle_feature_engineering(df, feature_engineering_ops)
    df = pd.concat([df, new_features_df], axis=1)
    
    # creates interaction features between zone and other features if requested
    if create_zone_interactions and 'location_zone' in df.columns:
        zone_interactions = {}
        for feature in zone_interaction_features:
            if feature in df.columns:
                for zone in range(kmeans_model.n_clusters):
                    interaction_name = f'{feature}_zone_{zone}'
                    # creates dummy variable: multiplication of the feature by the zone membership condition
                    zone_interactions[interaction_name] = df[feature] * (df['location_zone'] == zone)
        if zone_interactions:
            zone_interactions_df = pd.DataFrame(zone_interactions)
            df = pd.concat([df, zone_interactions_df], axis=1)
            # updates statistics of the new interactions
            for col in zone_interactions:
                feature_stats[col] = {
                    'mean': zone_interactions[col].mean(),
                    'std': zone_interactions[col].std(),
                    'min': zone_interactions[col].min(),
                    'max': zone_interactions[col].max()
                }
    
    # extracts position data if available
    df_pos = df[location_columns + ['location_zone']] if location_columns else None
    # removes location columns from the final DataFrame
    df = df.drop(location_columns, axis=1) if location_columns else df

    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return {
        'df': df,
        'df_pos': df_pos if location_columns else None,
        'zone_stats': zone_stats if location_columns else None,
        'feature_stats': feature_stats
    }


def compare_feature_impact(model_results_dict, property_dfs, feature_name='has_pool'):
    """
    Compares the impact of a specific feature between different models.
    
    Parameters:
    -----------
    model_results_dict : dict
        Dictionary of model results or a single results dictionary.
    property_dfs : dict
        Dictionary of DataFrames or a single DataFrame.
    feature_name : str
        Name of the feature to analyze.
        
    Returns:
    --------
    dict
        Dictionary with impact analysis results for each model.
    """
    # if a single model is provided, encapsulate it in a dictionary
    if not isinstance(model_results_dict, dict) or (isinstance(model_results_dict, dict) and 'model' in model_results_dict):
        model_results_dict = {"Model": model_results_dict}
        property_dfs = {"Model": property_dfs}
    
    if not isinstance(property_dfs, dict):
        raise ValueError("property_dfs must be a dictionary")
    
    if set(model_results_dict.keys()) != set(property_dfs.keys()):
        raise ValueError("model_results_dict and property_dfs must have the same keys")
    
    impacts = {}
    for model_name, model_results in model_results_dict.items():
        # tries to get the model's coefficient dictionary
        coef_dict = None
        if hasattr(model_results["model"], 'get_coef_dict') and callable(model_results["model"].get_coef_dict):
            coef_dict = model_results["model"].get_coef_dict()
        elif hasattr(model_results["model"], 'coef_dict') and model_results["model"].coef_dict:
            coef_dict = model_results["model"].coef_dict
        
        if not coef_dict or feature_name not in coef_dict:
            print(f"Warning: Feature '{feature_name}' not found in model '{model_name}'. Skipping.")
            continue
        
        df = property_dfs[model_name]
        feature_impact = coef_dict[feature_name]
        avg_property_price = df["price"].mean()  # average property price
        percentage_impact = (feature_impact / avg_property_price) * 100
        
        display_name = feature_name.replace('_', ' ').replace('has ', '').title()
        
        impacts[model_name] = {
            "feature_name": feature_name,
            "display_name": display_name,
            "absolute_impact": feature_impact,
            "average_price": avg_property_price,
            "percentage_impact": percentage_impact
        }
        
    return impacts


def cross_validate_lambda(X, y, lambdas, model_class: Model, n_splits=5, 
                          method='pseudo_inverse', regularization='l2', 
                          normalize=True, random_state=None,
                          variable: str = 'penalty',
                          transform_target=None,
                          metrics=None,
                          inv_transform_pred=None):
    """
    Performs cross-validation for different lambda (or degree) values and returns average metrics.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    lambdas : iterable
        Lambda (or degree) values to evaluate.
    model_class : Model
        Regression model class to use.
    n_splits : int
        Number of splits for cross-validation.
    method : str
        Fitting method for the model.
    regularization : str
        Type of regularization ('l1', 'l2').
    normalize : bool
        Whether to normalize the features.
    random_state : int
        Seed for reproducibility.
    variable : str
        Parameter to optimize ('penalty' or 'degree').
    transform_target : callable, optional
        Function to transform the target variable.
    metrics : list, optional
        List of metric functions to evaluate.
    inv_transform_pred : callable, optional
        Function to invert the target variable transformation.
        
    Returns:
    --------
    tuple
        (optimal_lambda, min_cv_metrics, cv_metrics_scores)
    """
    if metrics is None:
        metrics = [mse_score, r2_score]

    if transform_target and inv_transform_pred is None:
        raise ValueError("transform_target and inv_transform_pred cannot both be None. If transform_target is used, inv_transform_pred must be used to invert the transformation to calculate metrics.")
    
    def numpy_kfold(n_samples, n_splits, random_state=None):
        """custom implementation of k-fold cross-validation"""
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # randomly shuffles indices
        fold_size = n_samples // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            yield train_indices, val_indices

    n_samples = len(X)
    folds = list(numpy_kfold(n_samples, n_splits, random_state))
    
    # initializes dictionary to store cross-validation metrics
    cv_metrics_scores = {metric.__name__.lower(): [] for metric in metrics}
    
    # iterates over each lambda (or degree) value
    for lambda_val in lambdas:
        # stores metrics for each split for the current lambda
        fold_metrics_scores = {metric.__name__.lower(): [] for metric in metrics}
        for train_idx, val_idx in folds:
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()
            
            if transform_target:
                y_train = transform_target(y_train)
                y_val = transform_target(y_val)

            if normalize:
                # normalizes using statistics calculated from X_train
                mean = X_train.mean()
                std = X_train.std().replace(0, 1e-8)  # avoids division by zero
                X_train = (X_train - mean) / std
                X_val = (X_val - mean) / std
            
            model = model_class()
            if variable == 'degree':
                # for polynomial regression: changes the degree and fits the model
                model.change_degree(lambda_val)
                model.fit(X_train, y_train, regularization=regularization)
            elif variable == 'penalty':
                # for regularized models: fits using the penalty parameter
                model.fit(X_train, y_train, method=method, alpha=lambda_val, regularization=regularization)
            else:
                model.fit(X_train, y_train, regularization=regularization)
            
            y_pred_test = model.predict(X_val)
          
         
            score = evaluate_model(model, X_val, y_val, metrics, inv_transform_pred, y_pred_test)
           
            for key, value in score.items():
                fold_metrics_scores[key].append(value)

        # averages metrics across all folds for the current lambda
        for metric in metrics:
            key = metric.__name__.lower()
            mean_score = np.mean(fold_metrics_scores[key])
            cv_metrics_scores[key].append(mean_score)
    
    # converts metric lists to NumPy arrays
    cv_metrics_scores = {key: np.array(scores) for key, scores in cv_metrics_scores.items()}
    
    # determines the optimal index for each metric (minimum for errors, maximum for metrics like r2)
    optimal_idx = {}
    for key, scores in cv_metrics_scores.items():
        if any(term in key for term in ['error', 'loss', 'mse', 'mae']):
            optimal_idx[key] = np.argmin(scores)
        else:
            optimal_idx[key] = np.argmax(scores)
    
    optimal_lambda = {key: lambdas[optimal_idx[key]] for key in cv_metrics_scores}
    min_cv_metrics = {key: cv_metrics_scores[key][optimal_idx[key]] for key in cv_metrics_scores}
    
    return optimal_lambda, min_cv_metrics, cv_metrics_scores
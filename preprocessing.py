from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column, task_type, test_size=0.2, random_state=42):
    """
    Preprocess the data including handling missing values, encoding categorical variables,
    and scaling numerical variables.
    
    Returns:
    - X_train, X_test, y_train, y_test: Train and test splits
    - preprocessor: Fitted preprocessor pipeline
    - feature_names: List of feature names after preprocessing
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )
    
    if task_type == 'classification':
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = []
    
    if numerical_cols:
        feature_names.extend(numerical_cols)
    
    if categorical_cols:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

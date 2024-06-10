import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load sample data
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Define parameters
params = {
            'objective': 'regression',
                'metric': 'rmse',
                    'boosting_type': 'gbdt',
                        'num_leaves': 31,
                            'learning_rate': 0.05,
                                'verbose': 0
                                }

# Train the model
bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], early_stopping_rounds=10)

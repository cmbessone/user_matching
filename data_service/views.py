
# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import lightgbm as lgb
from .forms import UploadFileForm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64


def handle_uploaded_file(f):
    with open('uploaded_file.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def preprocess_data(df, target_column, features):
    df = df.dropna()
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    X = df[features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_lightgbm(X_train, y_train):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': 0
    }
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

def visualize_results(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=results)
    plt.xlabel('Actual Compatibility')
    plt.ylabel('Predicted Compatibility')
    plt.title('Actual vs Predicted Compatibility')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png)
    image_base64 = urllib.parse.quote(image_base64)

    return image_base64

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            df = load_csv('uploaded_file.csv')

            if df is not None:
                target_column = 'target_column_name'  # Replace with your actual target column
                features = ['feature1', 'feature2', 'feature3']  # Replace with your actual features
                X_train, X_test, y_train, y_test = preprocess_data(df, target_column, features)
                model = train_lightgbm(X_train, y_train)
                image_base64 = visualize_results(model, X_test, y_test)
                return render(request, 'results.html', {'image_base64': image_base64})
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})
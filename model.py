import csv
import sys
import time

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from circuit import get_chsh


def progress(iterable, length=33):
    count = avg = 0
    total = len(iterable)
    then = time.time()
    for it in iter(iterable):
        yield it
        count += 1
        avg += (time.time() - then - avg) / count
        then = time.time()
        percent = count / total
        filled_len = round(length * percent)
        bar = '█' * filled_len + ' ' * (length - filled_len)
        sys.stdout.write(
            f'▕{bar}▏ {round(100 * percent)}% {round((total - count) * avg)}s \r')
        sys.stdout.flush()
    sys.stdout.write(f'▕{"█" * length}▏ 100% {round(avg * total)}s \n')


def generate_dataset(num_samples=1024):
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['theta', 'prob_1', 'prob_2', 'chsh'])
        for i in progress(range(num_samples)):
            theta_vec, probs, chsh_vec = get_chsh()
            for theta, chsh in zip(theta_vec, chsh_vec):
                writer.writerow([theta, probs[0], probs[1], chsh])


def train(data_path='data.csv'):
    # Load data
    data = pd.read_csv(data_path)
    # Separate features and target
    X = data[['theta', 'prob_1', 'prob_2']]
    y = data['chsh']
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Evaluate on test set
    score = model.score(X_test, y_test)
    pickle.dump(model, open('model.pkl', 'wb'))
    print('R^2 score on test set:', score)

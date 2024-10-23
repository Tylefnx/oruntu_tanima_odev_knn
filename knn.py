import pandas as pd
from math import sqrt

data = pd.read_csv('knn_dumenden_dataset.csv')

x = float(input('Please enter the x value: '))
y = float(input('Please enter the y value: '))
k = 3

distances = []
for index, row in data.iterrows():
    distance = sqrt((row['x1'] - x) ** 2 + (row['y1'] - y) ** 2)
    distances.append((distance, row['label']))

distances.sort(key=lambda x: x[0])

nearest_neighbors = distances[:k]

from collections import Counter
labels = [label for _, label in nearest_neighbors]
most_common_label = Counter(labels).most_common(1)[0][0]

print(f'The predicted class for point ({x}, {y}) is: {most_common_label}')

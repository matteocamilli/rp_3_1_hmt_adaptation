import csv

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    y_test = []
    pred = []

    with open('validation_table_{}.csv'.format(threshold)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue

            y_test.append(int(row[4]))
            pred.append(int(row[5]))

    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('Threshold {}: Precision: {}, Recall: {}, F1: {}'.format(threshold, precision, recall, f1))

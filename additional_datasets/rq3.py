import csv

import matplotlib.pyplot as plt

csv_name = 'execution_time_log_{}.csv'
confs = ['20_20', '20_40', '40_20', '40_40']

vectors_name = 'improved_configurations/configurations_improved_{}.csv'

fig, ax = plt.subplots(ncols=1, figsize=(10, 6))

for i, conf in enumerate(confs):
    v = []

    with open(csv_name.format(conf), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for j, row in enumerate(reader):
            if j == 0:
                continue
            v.append(float(row[0]))

        ax.boxplot([v], positions=[i + 1], widths=0.6)

    tau = []
    with open(vectors_name.format(conf)) as vecfile:
        reader = csv.reader(vecfile, delimiter=',')
        for j, row in enumerate(reader):
            if j == 0:
                continue
            tau.append(float(row[5]))

    v.sort()
    tau.sort()

    for j, val in enumerate(v):
        if val < tau[j]:
            print('{}: {}/{}'.format(conf, j + 1, len(v)))
            break

ax.set_ylim(0, 12)
ax.set_ylabel('Time [s]')
ax.set_xticks([i + 1 for i in range(len(confs))], ['NSGA-II ({})'.format(conf) for conf in confs])

plt.savefig('rq3.png', bbox_inches='tight')
plt.show()
plt.close()

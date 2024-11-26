import csv

csv_file = "comparison_with_log.csv"

with open(csv_file, "r") as csvfile:
    reader = csv.reader(csvfile)
    smc_times = []
    fails = 0
    scs = 0
    for i, row in enumerate(reader):
        if i == 0:
            continue

        smc_times.append(float(row[1]))
        if int(row[1]) > int(row[2]):
            fails += 1
        else:
            scs += 1

    print("Avg. time for verification: {}".format((sum(smc_times)/len(smc_times))/1000/60))
    print("no time in {}/{} of the cases".format(fails, 100))

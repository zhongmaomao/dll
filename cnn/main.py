import csv
import numpy as np
import moudles


with open('acc.csv', 'w') as fp:
    writer = csv.writer(fp)
    for dpo in np.arange(0, 1, 0.2):
        acc, net = moudles.train_net(dpo=dpo)
        writer.writerow(acc[0])
        writer.writerow(acc[1])


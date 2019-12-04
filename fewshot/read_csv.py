import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--N', default="10", type=str)
parser.add_argument('--last', default=80, type=int)

args = parser.parse_args()
last = args.last

loss = []
accuracy = []
stnloss = []
stnaccuracy = []

with open(args.file, 'r') as f:
    lines = f.read().split('\n')
    lines = [line for line in lines if line!=""]
    for i, line in enumerate(lines):
        if i>0:
            accuracy.append(float(line.split(',')[-2].strip()))
            loss.append(float(line.split(',')[-1].strip()))
epochs = np.arange(len(loss))
accuracy = np.array(accuracy)[:last]
loss = np.array(loss)[:last]

try:
    args.file = args.file.replace('.csv', '_stn_{}.csv'.format(args.N))
    with open(args.file, 'r') as f:
        lines = f.read().split('\n')
        lines = [line for line in lines if line!=""]
        for i, line in enumerate(lines):
            if i>0:
                stnaccuracy.append(float(line.split(',')[-2].strip()))
                stnloss.append(float(line.split(',')[-1].strip()))

    stnloss = np.array(stnloss)[:last]
    stnaccuracy = np.array(stnaccuracy)[:last]
    stnepochs = np.arange(len(stnloss))
except:
    stnaccuracy = []
    stnloss = []

print(accuracy)
print(stnaccuracy)
print(np.max(accuracy))
print(np.max(stnaccuracy))
plt.figure()
#fig, ax = plt.subplots(1, 2, sharey=True)
#ax[0].plot(epochs, loss, label="loss")
#ax[1].plot(epochs, accuracy, label="Accuracy")
#ax[0].plot(stnepochs, stnloss, label="stnloss")
#ax[1].plot(stnepochs, stnaccuracy, label="stnAccuracy")
#ax[0].legend()
#ax[1].legend()
plt.plot(accuracy)
plt.plot(stnaccuracy)
plt.legend(['baseline', 'stn'])

#plt.show()
plt.savefig('loss.png')

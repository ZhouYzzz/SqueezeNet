import os

imgs = list()
for i in xrange(1467):
	imgs.append([])

# print imgs

for name in os.listdir('labeled'):
	label = int(name[:4])
	imgs[label-1].append(name)

full10 = list()

for i in xrange(1467):
	if len(imgs[i]) >= 5:
		full10.append(i)

# print len(full10)

import numpy as np

choice = np.random.choice(full10, 1400, False)

f = open('train.txt', 'w')

for i in xrange(1400):
	for img in imgs[choice[i]][:-1]:
		f.write('labeled/%s %d'%(img, i)+os.linesep)

f.close()

f = open('test.txt', 'w')

for i in xrange(1400):
	img = imgs[choice[i]][-1]
	f.write('labeled/%s %d'%(img, i)+os.linesep)

f.close()
import sys
sys.path.insert(0, '/home/gpu/zhouyz/caffe/python')
import caffe
import cv2
import numpy as np
from time import time as t

proto = 'deploy.prototxt'
model = 'squeezenet_v1.0.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(proto, model, caffe.TEST)

print '=========='
print 'Net Loaded.'
# print 'Input :', net.inputs
# print 'Output:', net.outputs

def readimg(filename):
	im = cv2.imread(filename) - np.array([[[104,117,123]]])
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	return im

im = readimg('MotorRolling/img/0001.jpg')
net.blobs['data'].reshape(*im.shape)
# print net.blobs['data'].data.shape


net.forward_all(data=im)

'''
['data',
 'conv1',
 'pool1',
 'fire2/squeeze1x1',
 'fire2/squeeze1x1_fire2/relu_squeeze1x1_0_split_0',
 'fire2/squeeze1x1_fire2/relu_squeeze1x1_0_split_1',
 'fire2/expand1x1',
 'fire2/expand3x3',
 'fire2/concat',
 'fire3/squeeze1x1',
 'fire3/squeeze1x1_fire3/relu_squeeze1x1_0_split_0',
 'fire3/squeeze1x1_fire3/relu_squeeze1x1_0_split_1',
 'fire3/expand1x1',
 'fire3/expand3x3',
 'fire3/concat',
 'fire4/squeeze1x1',
 'fire4/squeeze1x1_fire4/relu_squeeze1x1_0_split_0',
 'fire4/squeeze1x1_fire4/relu_squeeze1x1_0_split_1',
 'fire4/expand1x1',
 'fire4/expand3x3',
 'fire4/concat',
 'pool4',
 'fire5/squeeze1x1',
 'fire5/squeeze1x1_fire5/relu_squeeze1x1_0_split_0',
 'fire5/squeeze1x1_fire5/relu_squeeze1x1_0_split_1',
 'fire5/expand1x1',
 'fire5/expand3x3',
 'fire5/concat',
 'fire6/squeeze1x1',
 'fire6/squeeze1x1_fire6/relu_squeeze1x1_0_split_0',
 'fire6/squeeze1x1_fire6/relu_squeeze1x1_0_split_1',
 'fire6/expand1x1',
 'fire6/expand3x3',
 'fire6/concat',
 'fire7/squeeze1x1',
 'fire7/squeeze1x1_fire7/relu_squeeze1x1_0_split_0',
 'fire7/squeeze1x1_fire7/relu_squeeze1x1_0_split_1',
 'fire7/expand1x1',
 'fire7/expand3x3',
 'fire7/concat',
 'fire8/squeeze1x1',
 'fire8/squeeze1x1_fire8/relu_squeeze1x1_0_split_0',
 'fire8/squeeze1x1_fire8/relu_squeeze1x1_0_split_1',
 'fire8/expand1x1',
 'fire8/expand3x3',
 'fire8/concat',
 'pool8',
 'fire9/squeeze1x1',
 'fire9/squeeze1x1_fire9/relu_squeeze1x1_0_split_0',
 'fire9/squeeze1x1_fire9/relu_squeeze1x1_0_split_1',
 'fire9/expand1x1',
 'fire9/expand3x3',
 'fire9/concat',
 'conv10',
 'pool10',
 'prob']
'''

for (LAYER, BLOB) in net.blobs.iteritems():
	print LAYER, '\n\t|--', BLOB.data.shape
	
feat = net.blobs['fire5/squeeze1x1'].data
display = feat[0,:3,:,:].transpose((1,2,0))
# display = cv2.resize(display, (640,320)).astype(np.int8)
print display.shape, display.max(), display.min()
cv2.imshow('display', display)
cv2.waitKey()
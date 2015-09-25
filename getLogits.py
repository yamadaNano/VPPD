'''Get logits for ImageNet'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import caffe
import numpy
import skimage.io as skio

caffe_root = '/home/daniel/Code/caffe/'

if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    sys.path.append('../scripts/download_model_binary.py')
    sys.path.append('../models/bvlc_reference_caffenet')

'''Initialise network'''
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/test_dropout.prototxt',
        caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
        caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
def getPreactivations(image):
    '''Get feature maps'''
    #print image.shape
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    #print net.blobs['data'].data[...].shape
    net.forward()
    s = net.blobs['fc8'].data
    return s

def getImages(folder, writename):
    '''List all of the image addresses in the folder and return a file'''
    fp = open(writename, "w")
    i = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            fname = root + '/' + f + '\n'
            if 'JPEG' in fname:
                fp.write(fname)
                i += 1
                if i % 100 == 0:
                    sys.stdout.write("Loading image: %i \r" % (i,))
                    sys.stdout.flush()
    fp.close()

def getCategories(folder, writename):
    '''Traverse directory and write subfolder names to file'''
    fp = open(writename, "w")
    i = 0
    for x in os.walk(folder):
        fp.write(x[0] + '\n')
        i += 1
        sys.stdout.write("Writing folder: %i, %s \r" % (i,x[0]))
        sys.stdout.flush()
    fp.close()

def chooseRandomCategories(readname, writename, n=50):
    '''Choose n random categories from folder and write all files'''
    with open(readname, "r") as rp:
        lines = rp.readlines()
    num_lines = len(lines)
    print("Num categories: %i" % (num_lines,))
    idx = numpy.random.choice(num_lines, 50)
    with open(writename, "w") as wp:
        for ind in idx:
            wp.write(lines[ind])

def getImagesFromCategories(readname, writename):
    '''For each of the folders get all the images and write to file'''
    with open(readname, "r") as rp:
        lines = rp.readlines()
    i = 0
    wp = open(writename, "w")
    for line in lines:
        for root, dirs, files in os.walk(str(line.replace('\n',''))):
            for f in files:
                fname = root + '/' + f + '\n'
                if 'JPEG' in fname:
                    wp.write(fname)
                    i += 1
                    if i % 100 == 0:
                        sys.stdout.write("Loading image: %i \r" % (i,))
                        sys.stdout.flush()
    wp.close()

def getLogits(srcfile, dstfolder, num_samples):
    '''Get the logits from the given images'''
    i = 0
    start = time.time()
    pred = (0,0,0)
    with open(srcfile, 'r') as fp:
        lines = fp.readlines()
        num_lines = len(lines)
        for line in lines:
            i += 1
            line = line.rstrip('\n')
            base = os.path.basename(line).replace('.JPEG','')
            sys.stdout.flush()
            sys.stdout.write("NUM: %i/%i TIME: %d:%02d:%02d NAME: %s \r" % (i,num_lines,pred[0],pred[1],pred[2],base))
            image = skio.imread(line)
            image = numpy.dstack((image,)*num_samples)
            logits = getPreactivations(image)
            numpy.savez_compressed(dstfolder + '/' + base, logits)
            if i % 10 == 0:
                m, s = divmod(num_lines*(time.time() - start)/i, 60)
                h, m = divmod(m, 60)
                pred = (h, m, s)
                
            
if __name__ == '__main__':
    folder = "/media/daniel/DATA/ImageNet/ILSVRC2012_img_train"
    writename = "/media/daniel/DATA/ImageNet/train.txt"
    transfercategoriesname = "/media/daniel/DATA/ImageNet/transfercategories.txt"
    transfername = "/media/daniel/DATA/ImageNet/transfer.txt"
    categoryname = "/media/daniel/DATA/ImageNet/categories.txt"
    dstname = "/media/daniel/DATA/ImageNet/logits"
    #getImages(folder, writename)
    #getCategories(folder, categoryname)
    #chooseRandomCategories(categoryname, transfercategoriesname)
    #getImagesFromCategories(transfercategoriesname, transfername)
    print getLogits(transfername, dstname, 25)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import os
import sys
import glob
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

datapath = './Pandora_18k'
mergedDatapath = './Pandora_18k_merged'

def preprocess1():
    d = datapath
    dirsNames = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    dirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    #print dirs

    # move all images from subfolders to a single folder. Ignore some naming conflicts
    for i in xrange(len(dirs)):
        mergedPath = mergedDatapath + '/' + dirsNames[i] + "_merged"
        os.system("mkdir "+mergedPath)
        os.system("find "+dirs[i]+"  -iname '*.jpg' -exec cp -f -t "+mergedPath+"  '{}' +")
        #if i > 1: break

def getfeatures(filename,size):
    #print(filename)
    img =cv2.imread(filename)
    img = cv2.resize(img,size)
    #plt.imshow(img)
    #plt.show()
    #print(img.shape)

    '''
    # HOG
    hf0 = hog(img[:,:,0],block_norm="L2-Hys") # 1d array
    hf1 = hog(img[:,:,1],block_norm="L2-Hys")
    hf2 = hog(img[:,:,2],block_norm="L2-Hys")
    hf = np.concatenate((hf0,hf1,hf2))
    #print(hf.shape," --hf")
    #sys.exit(0)
    return hf
    '''
    chist0 =  np.histogram(img[:,:,0], bins=32)[0]
    chist1 =  np.histogram(img[:,:,1], bins=32)[0]
    chist2 =  np.histogram(img[:,:,2], bins=32)[0]
    chist =   np.concatenate((chist0,chist1,chist2))
    #print("chist1.shape, chist.shape",chist1.shape,chist.shape)
    return chist


def getDataset(perClassCount = 100, size=(500,500)):
    
    pkfilename = mergedDatapath + '/' + 'feature_color_data.sav'

    # load features if we have saved it earlier.
    X,Y = pickle.load(open(pkfilename, 'rb'))
    return X,Y ##########################################################################
    

    d = mergedDatapath
    dirsNames = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    dirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

    np.random.seed(0)
    X,Y= [],[]
    for i in range(len(dirs)):
        d = dirs[i]
        filenames = glob.glob(d+'/*.jpg')
        l = len(filenames)
        f = list(np.random.choice(filenames,replace=False,size=perClassCount))
        #print(f)
        X += f
        Y += [i for j in range(perClassCount)] # label of these images
        

    print("Getting features for the images...")
    XF = []
    j = 0
    for i in range(len(X)):
        XF.append(getfeatures(X[i],size))
        if len(XF)%perClassCount==0:
            print("Extracting features for class "+str(j)+" is done..")
            j+=1
    X = XF
    print("Dumping the features for later use..")
    
    pickle.dump((X,Y), open(pkfilename, 'wb'))

    return X,Y


if __name__ == '__main__':
    
    #preprocess1()
    X,Y = getDataset(perClassCount = 600,size=(300,300))
    trainx,testx, trainy ,testy = train_test_split(X,Y,test_size=0.2)

    print("Traing and testing model..")
    clf = svm.SVC(max_iter=1000000000)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainx,trainy)
    print("Score is:",clf.score(testx,testy))




from matplotlib import pyplot as plt
import random
import numpy as np

def affiche(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def split_base(Base, labels, split_ratio ):


    Nb=Base.shape[0]
    Nbnew=round(Nb*split_ratio)
    l = [i for i in range(Nb)]
    l=random.sample(l, Nbnew)
    BaseApp=np.empty([Nbnew,Base.shape[1],Base.shape[2]])
    labelsApp=np.empty(Nbnew)


    for i in range (0,Nbnew):
        BaseApp[i,:,:]=Base[l[i],:,:]
        labelsApp[i]=labels[l[i]]

    return BaseApp,labelsApp

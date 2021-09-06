# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:59:44 2021

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,  accuracy_score

def plot_gallery(images): 
    
    # Affiche les 12 premières images contenues dans images
    # images est de taille Nb image*Ny*Nx 
    plt.figure(figsize=(7.2, 7.2)) 
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35) 
    for i in range(12): 
        plt.subplot(3, 4, i + 1) 
        plt.imshow(images[i], cmap=plt.cm.gray) 
        plt.xticks(()) 
        plt.yticks(()) 
        plt.show()
[X, y, name]=np.load("TP1.npy",allow_pickle=True )
    
#charger les données
plot_gallery(X) #afficher les images

print(X.shape) #taille de limage 62*47
print(max(y)) #/le nombre de classes
print(name)   #identité des personnes
plt.hist(y,bins=7,range=[0,7])   #histogramme

#partitionnement de la base d'apprentissage

#repartition des données en donnee d'apprentissage et test
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=42) 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Redimensionnement des données 
# pour passer d'une image représenté en 2D à une image 1D
# n=2914=62*47

X_train=np.reshape(X_train,[X_train.shape[0],2914])
X_test=np.reshape(X_test,[X_test.shape[0],2914])


# Mise en forme des données pour la classification
# donnees d'apprentissage
scaler1=StandardScaler()
scaler1.fit(X_train)  # pour la calcul de la moyenne et la variance
print(scaler1.mean_)  # moyenne
print(scaler1.var_)   # variance

X_train=scaler1.transform(X_train)  #normalisation des données

#données de test
scaler2=StandardScaler()
scaler2.fit(X_test)   # pour la calcul de la moyenne et la variance
print(scaler2.mean_)  # moyenne
print(scaler2.var_)   # variance

X_test=scaler2.transform(X_test)  #normalisation des données

# Classifieur 1PPV
#classifieur=KNeighboursClassifier(n_neighbours=...,p=1 ou 2) p=1 dist euclidienne p=2 dist de manhattan

classifieur=KNeighborsClassifier(n_neighbors=1,p=1)
classifieur.fit(X_train,y_train)
y_pred=classifieur.predict(X_test)

mat=confusion_matrix(y_test,y_pred) #matrice de confusion

#taux de reconnaissance à partir des éléments de la mat de confusion  
# accuracy = somme des elements de la diagonnale (trace) / somme de tt les elements
# deux façon de calculer l'accuracy
acc1=np.sum(np.diag(mat))/np.sum(mat)
acc2=accuracy_score(y_test,y_pred)
print(acc1,acc2)
plt.hist(y_pred,bins=7,range=[0,7])   #histogramme permet de voir si les classes sont équilibrées ou nn
# d'apres l'histogramme les classes ne sont pas equilibréesla classe 3 est representée plus que les autres
 
# la matrice de confusion represente comment les donnees ont été predites
# la somme de la matrice de confusion represente le nombre d'images de test

# Classifieur KPPV
for k in range(1,10):
    classifieur=KNeighborsClassifier(n_neighbors=k,p=1)
    classifieur.fit(X_train,y_train)
    y_pred=classifieur.predict(X_test)
    mat=confusion_matrix(y_test,y_pred) 
    print(accuracy_score(y_test,y_pred))
    
# Classifieur KPPV et distance de Manhattan
for k in range(1,10):
    classifieur=KNeighborsClassifier(n_neighbors=k,p=2)
    classifieur.fit(X_train,y_train)
    y_pred=classifieur.predict(X_test)
    mat=confusion_matrix(y_test,y_pred) 
    print(accuracy_score(y_test,y_pred))
     
    
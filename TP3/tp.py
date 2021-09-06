# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:53:44 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,  accuracy_score, roc_curve

def norm1(x, m, s):
    p =1/(np.sqrt(2*np.pi)*s)*np.exp(-(x-m)*(x-m)/(2*s*s)) 
    return p

def norm2(x, m, cov): 
    a = np.dot(np.transpose((x-m)), np.linalg.inv(cov)) 
    a = np.dot(a, (x-m)) 
    p =1/(np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*a) 
    return p

def classifieur (p_test,m_train,y_test):
    y_pred=np.zeros(np.size(y_test))
    for i in  range (len(p_test)):
        if p_test[i]>m_train:
            y_pred[i]=1
    return y_pred

#chargement des données

[X_train, y_train, X_test, y_test] = np.load("TP3.npy",allow_pickle=True)
print(np.shape(X_train))
#Pixel peau
T_train = X_train[np.where(y_train==1),:]
T_train = np.reshape(T_train,(T_train.shape[1],T_train.shape[2] ))
#Pixel non peau
F_train = X_train[np.where(y_train==0),:]
F_train = np.reshape(F_train,(F_train.shape[1],F_train.shape[2] ))

plt.plot(F_train[:,0], F_train[:,1], '.b')
plt.show()
plt.plot(T_train[:,0], T_train[:,1], '.r')
plt.show()

#deux colonnes qui correspondent à Cb et Cr
print(np.shape(T_train)) #nombre de pixels de teinte chaire=639
print(np.shape(F_train)) #nombre de pixels de teinte non chaire=1731

#dimension des données = 2370
print(np.shape(X_train))

#Estimation de la densité de probabilité a priori des pixels de teinte chaire
#moyennes
m_cb=np.mean(T_train[:,0])
m_cr=np.mean(T_train[:,1])

#ecarts_type
sigma_cb=np.std(T_train[:,0])
sigma_cr=np.std(T_train[:,1])

p1_train=norm1(X_train[:,0], m_cb, sigma_cb) * norm1(X_train[:,1], m_cr, sigma_cr)
# dimension de p1_train = vecteur de 639 lignes
print(np.shape(p1_train))

#hypothèse nous permet d’estimer la valeur de la loi normale à partir de l’équation précédente ?   

#Classification
print(np.shape(X_test))

m_p1_train=p1_train.mean() #seuil
print(m_p1_train)

p1_test=norm1(X_test[:,0], m_cb, sigma_cb) * norm1(X_test[:,1], m_cr, sigma_cr) # 
y_pred=classifieur(p1_test,m_p1_train,y_test)

#matrice de confusion
mat=confusion_matrix(y_test,y_pred)
print(mat)

TP=mat[0,0]
TN=mat[1,1]
FP=mat[1,0]
FN=mat[0,1]

#sensibilité et specificité
sens=TP/(TP+FN)
spec=TN/(TN+FP)

#taux de bonne classification
TB=(TP+TN)/(TP+TN+FP+FN)


#Pourquoi avoir choisi ce seuil initial

#Courbe ROC
NB = 20
step = (np.max(p1_train) - np.min(p1_train) ) / NB 
SEUILS = np.arange (np.min(p1_train), np.max(p1_train), step)
sensibilite=np.zeros(NB)
specificite=np.zeros(NB)

for i in range (0,20):
        
    y_pred=classifieur(p1_test,SEUILS[i],y_test)
    mat=confusion_matrix(y_test,y_pred)
    sensibilite[i]=mat[0,0]/(mat[0,0]+mat[0,1])
    specificite[i]=mat[1,1]/(mat[1,1]+mat[1,0])
    
plt.plot(1-specificite,sensibilite,'b')
    

#Modélisation de la densité de probabilité a priori de la teinte chaire par une loi normale 2D
    
covariance= np.cov(np.transpose(X_train)) # X_train ou T_train?
print(np.shape(covariance))

m=np.mean(X_train,axis=0) # ou bien T_train?
p1_test_2=norm2(T_test,m,covariance) #oubien X_test?

for i in range (0,20):
        
    y_pred=classifieur(p1_test_2,SEUILS[i],y_test)
    mat=confusion_matrix(y_test,y_pred)
    sensibilite[i]=mat[0,0]/(mat[0,0]+mat[0,1])
    specificite[i]=mat[1,1]/(mat[1,1]+mat[1,0])
    
plt.plot(1-specificite,sensibilite,'r')



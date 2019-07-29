import time
import math
import numpy as np

from keras.models import load_model
from keras import backend as K

class Encoder:
  
    def __init__(self, model_path):
        self.model=load_model(model_path)
        self.get_encoded = K.function([self.model.layers[0].input], [self.model.layers[5].output])

    def _get_encode(self, X):
        layer_output = self.get_encoded([X])[0]
        return layer_output

    def _get_KL(self, A, B):
        kl=0;ans=0
        A, B=np.ravel(A), np.ravel(B)
        C=np.ones(A.shape) # .copy()
        for i, (a, b) in enumerate(zip(A, B)):
            if a!=0 and b!=0:
                C[i]=a/b
            elif a!=0 and b==0:
                C[i]=(2*a)/(a+b)
            else:
                C[i]=1
        kl=np.sum(np.log(C)*A)
        # for a, b in zip(A, B):
        #     if a!=0 and b!=0:
        #         ans=a*np.log(a/b)
        #     elif a!=0 and b==0:
        #         ans=a*np.log((2*a)/(a+b))
        #     elif a==0 and b==0:
        #         ans=0
        #     else:
        #         ans=0
        #     kl+=ans
        # kl=abs(expert.y-robot.y) 
     
        return kl

    def getKL(self, A, B):
       # A, B=self._get_encode(np.array([A, B]))
       # return self._get_KL(A, B)
       return abs(A[1]-B[1])

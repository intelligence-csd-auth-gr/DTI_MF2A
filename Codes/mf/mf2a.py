import numpy as np
from base.inbase import InductiveModelBase

from mf.nrmfap2 import NRMFAP2
from mf.nrmfauc_2 import NRMFAUC_F3

class MF2A(InductiveModelBase):
    """ensemble of MFAP and MFAUC """
    def __init__(self, mfap, mfauc, w):
        self.mfap = mfap
        self.mfauc = mfauc
        self.w = w        
      
        self.copyable_attrs = ['mfap','mfauc','w']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        
        # self.mfap = NRMFAP2(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=num_factors1, theta=0.1, lambda_d=lambda_d1, lambda_t=lambda_t1, lambda_r=lambda_t1, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
        # self.mfauc = NRMFAUC_F3(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=num_factors2, theta=0.1, lambda_d=lambda_d2, lambda_t=lambda_t2, lambda_r=lambda_r2, max_iter=100, seed=0, sfun=2, is_comLoss=0)
        
        self.mfap.fit(intMat, drugMat, targetMat, cvs)
        self.mfauc.fit(intMat, drugMat, targetMat, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)   
        
        scores_ap = self.mfap.predict(drugMatTe, targetMatTe)
        scores_auc = self.mfauc.predict(drugMatTe, targetMatTe)
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
    
    
class MF2A2(InductiveModelBase):
    """ensemble of MFAP and MFAUC """
    def __init__(self, M=21, num_factors1=50, lambda_d1=0.25, lambda_t1=0.25, lambda_r1=0.25, num_factors2=50, lambda_d2=0.25, lambda_t2=0.25, lambda_r2=0.25, w=0.5):
        # self.mfap = mfap
        # self.mfauc = mfauc
        self.M = M
        self.num_factors1 = num_factors1
        self.lambda_d1 = lambda_d1
        self.lambda_t1 = lambda_t1
        self.lambda_r1 = lambda_r1
        
        self.num_factors2 = num_factors2
        self.lambda_d2 = lambda_d2
        self.lambda_t2 = lambda_t2
        self.lambda_r2 = lambda_r2
        
        self.w = w        
      
        self.copyable_attrs = ['M', 'num_factors1','lambda_d1','lambda_t1','lambda_r1', 'num_factors2','lambda_d2','lambda_t2','lambda_r2', 'w']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        
        self.mfap = NRMFAP2(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=self.num_factors1, theta=0.1, lambda_d=self.lambda_d1, lambda_t=self.lambda_t1, lambda_r=self.lambda_r1, M=self.M, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
        self.mfauc = NRMFAUC_F3(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=self.num_factors2, theta=0.1, lambda_d=self.lambda_d2, lambda_t=self.lambda_t2, lambda_r=self.lambda_r2, max_iter=100, seed=0, sfun=2, is_comLoss=0)
        
        self.mfap.fit(intMat, drugMat, targetMat, cvs)
        self.mfauc.fit(intMat, drugMat, targetMat, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)   
        
        scores_ap = self.mfap.predict(drugMatTe, targetMatTe)
        scores_auc = self.mfauc.predict(drugMatTe, targetMatTe)
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
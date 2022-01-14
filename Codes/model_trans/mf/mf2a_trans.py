import numpy as np
from base.transbase import TransductiveModelBase

from model_trans.mf.nrmfap2_trans import NRMFAP2_TRANS
from model_trans.mf.nrmfauc_f3_trans import NRMFAUC_F3_TRANS

class MF2A_TRANS(TransductiveModelBase):
    """ensemble of MFAP and MFAUC """
    def __init__(self, mfap, mfauc, w):
        self.mfap = mfap
        self.mfauc = mfauc
        self.w = w        
      
        self.copyable_attrs = ['mfap','mfauc','w']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, test_indices, cvs)
        
        # self.mfap = NRMFAP2_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=num_factors1, theta=0.1, lambda_d=lambda_d1, lambda_t=lambda_t1, lambda_r=lambda_t1, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
        # self.mfauc = NRMFAUC_F3_TRANS(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=num_factors2, theta=0.1, lambda_d=lambda_d2, lambda_t=lambda_t2, lambda_r=lambda_r2, max_iter=100, seed=0, sfun=2, is_comLoss=0)
        
        scores_ap = self.mfap.fit(intMat, drugMat, targetMat, test_indices, cvs)
        scores_auc = self.mfauc.fit(intMat, drugMat, targetMat, test_indices, cvs)
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
    
    def _get_prediction_trainset(self):
        scores_ap = self.mfap._get_prediction_trainset()
        scores_auc = self.mfauc._get_prediction_trainset()
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
        

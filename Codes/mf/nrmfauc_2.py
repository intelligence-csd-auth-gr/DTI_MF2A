import numpy as np

from sklearn.neighbors import NearestNeighbors
from mf.nrmfauc import NRMFAUC, NRMFAUC2
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score

"""
nrmfauc using sampling pairs to compute derivatives of AUC loss w.r.t U and V
"""

class NRMFAUC_F(NRMFAUC):
    """
    NRMFAUC_F has some results with NRMFAUC
    Difference with NRMLAUC: NRMFAUC_F has faster running time for computing derivatives of AUC loss, 
                            but needs more menory, so that it cannot handle gpcr, ic and e dataset.                   
    """
    
    def _sampling_pairs(self):
        """ random sampling pairs"""
        # n1 = self._idx1[0].size
        # n0 = self._idx0[0].size
        # n = n0+n1 
        # index1 = prng.choice(np.arange(n1), n)
        # idx1_i = self._idx1[0][index1]
        # idx1_j = self._idx1[1][index1]
        # idx1_s = (idx1_i, idx1_j)
        
        # index0 = np.arange(n0)
        # idx0_i = self._idx0[0][index0]
        # idx0_j = self._idx0[1][index0]
        # idx0_s = (idx0_i, idx0_j)
        
        idx1_1d = np.ravel_multi_index(self._idx1,self._intMat.shape) # array([0,1,4,5])
        idx0_1d = np.ravel_multi_index(self._idx0,self._intMat.shape) # array([2,3])
        i1_ = np.tile(idx1_1d,len(idx0_1d))  # i1_ = array([0,1,4,5,0,1,4,5])
        i0_ = np.repeat(idx0_1d,len(idx1_1d)) # i0_ = array([2,2,2,2,3,3,3,3])
        idx1_s = np.unravel_index(i1_,self._intMat.shape)
        idx0_s = np.unravel_index(i0_,self._intMat.shape)
        
        return idx1_s, idx0_s 
    #----------------------------------------------------------------------------------------    
    
    def _deriv_AUCLoss(self, U, V, idx1_s, idx0_s, Y_pred, idx_flag, func_loss_deriv):
        """compute the derivatives of AUC loss w.r.t U or V, idx_flag = [0,1] for U and [1,0] for V        
        """
        n,m = Y_pred.shape
        deriv_U = np.zeros(U.shape)
        
        if idx_flag[0] == 1:  #deriv of V, exchange the tuple elements in idx1_s, idx0_s
            i1, j1 = idx1_s
            idx1_s = j1, i1
            i0, j0 = idx0_s
            idx0_s = j0, i0
        
        Y1 = Y_pred[idx1_s]
        Y0 = Y_pred[idx0_s]
        diff = Y1-Y0
                
        du1 = func_loss_deriv(diff)[:,None]*V[idx1_s[1]]
        du0 = -func_loss_deriv(diff)[:,None]*V[idx0_s[1]] 
        
        deriv_U1 = np.array([du1[idx1_s[0]==i].sum(axis=0) for i in range(n)]) # 2d array shape=(n, num_factors), deriv_U1[index] is the sum of rows of du1 whose corresponding i=index    
        deriv_U0 = np.array([du0[idx0_s[0]==i].sum(axis=0) for i in range(n)])
        deriv_U = deriv_U1+deriv_U0
        
        # this part of code has same function with above three lines and similar running time 
        # for i1 in range(n):
        #     du = du1[idx1_s[0]==i1]
        #     sdu = np.sum(du, axis=0)
        #     deriv_U[i1] += sdu
            
        #     du = du0[idx0_s[0]==i1]
        #     sdu = np.sum(du, axis=0)
        #     deriv_U[i1] += sdu
            
        return deriv_U
    #----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
        
class NRMFAUC_F1(NRMFAUC_F):
    """
    Difference with NRMLAUC_F: random undersampling pairs
    
    Better than NRMFAUC(_F) when setting maxiter=100
    """   
    def _sampling_pairs(self):
        """ random sampling pairs"""
        n1 = self._idx1[0].size
        n0 = self._idx0[0].size
        n = n0+n1 
        index1 = self._prng.choice(np.arange(n1), n)
        idx1_s = (self._idx1[0][index1], self._idx1[1][index1])
        
        index0 = self._prng.choice(np.arange(n0), n)
        idx0_s = (self._idx0[0][index0], self._idx0[1][index0])
        
        return idx1_s, idx0_s 
    #----------------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------    

class NRMFAUC_F3(NRMFAUC_F1):
    """
    This is the model used in paper
    Prediction is U@V.T
    Difference with NRMLAUC_F1: 
    choose best T for computing latent feature of test drug and/or target
    """   
    def __init__(self, K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=0, is_comLoss=0):
        self.K1 = K1 # used for sparsying similarity matrix
        self.K2 = K2 # used for computing latent features of new drugs/targets
        self.Ts = Ts # candidates decay coefficients
        self.metric= metric # metric to choose T, 0: aupr; 1: aupr+auc
        
        self.num_factors = num_factors # letent feature of drug and target
        self.theta = theta  # learning rate
        self.lambda_d = lambda_d  # coefficient of graph based regularization of U
        self.lambda_t = lambda_t  # coefficient of graph based regularization of V
        self.lambda_r = lambda_r  # coefficient of ||U||_F^2+||V||_F^2 regularization
        self.max_iter = max_iter
        self.seed = seed
        
        self.sfun = sfun # the convex surrogate loss function 
        """(0: square loss; 1: square hinge loss; 2: logistic loss)"""
        self.is_comLoss = is_comLoss # if compute loss for each iteration or not (0: do not compute; 1: compute all Loss; 2 compute AUC only)
        """ compute AUC Loss could lead to out of memory dataset e"""
        
        self.copyable_attrs = ['K1','K2','metric','Ts','num_factors','theta','lambda_d','lambda_t','lambda_r', 'sfun','is_comLoss', 'max_iter','seed']
    #----------------------------------------------------------------------------------------        
    
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        # self.lambda_t = self.lambda_d # ensure self.lambda_t = self.lambda_d 
        self._intMat = intMat
        self._init_f_loss_and_deriv()
        
        self._construct_neighborhood(drugMat, targetMat) # 
        self._AGD_optimization()  
        self._get_optimal_T(drugMat, targetMat)
    #----------------------------------------------------------------------------------------
        
    def _compute_aupr(self, labels_1d, scores_1d):
        aupr_val = average_precision_score(labels_1d,scores_1d)
        if np.isnan(aupr_val):
            aupr_val=0.0
        return aupr_val 
    #----------------------------------------------------------------------------------------
    
    def _get_optimal_T(self, drugMat, targetMat):
        Sd = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        St = targetMat - np.diag(np.diag(targetMat))
        
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(1-Sd, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(1-St, return_distance=False)
        
        best_value = -1; self._best_T = None
        for T in self.Ts:
            etas = T**np.arange(self.K2)
            if self._cvs == 2: 
                U = np.zeros(self._U.shape)
                for d in range(self._n_drugs):
                    ii = knn_d[d]
                    sd = Sd[d,ii]
                    U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                V = self._V
            elif self._cvs == 3:
                V = np.zeros(self._V.shape)
                for t in range(self._n_targets):
                    jj = knn_t[t]
                    st = St[t,jj]
                    V[t,:]= etas*st@self._V[jj, :]/np.sum(st)   
                U = self._U
            elif self._cvs == 4:
                U = np.zeros(self._U.shape)
                for d in range(self._n_drugs):
                    ii = knn_d[d]
                    sd = Sd[d,ii]
                    U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                V = np.zeros(self._V.shape)
                for t in range(self._n_targets):
                    jj = knn_t[t]
                    st = St[t,jj]
                    V[t,:]= etas*st@self._V[jj, :]/np.sum(st)
            Y_pred = U@V.T
            if self.metric == 0:
                auc = self._compute_auc(self._intMat.flatten(), Y_pred.flatten())
                value = auc
            elif self.metric == 1:
                aupr = self._compute_aupr(self._intMat.flatten(), Y_pred.flatten())
                auc = self._compute_auc(self._intMat.flatten(), Y_pred.flatten())
                value = aupr + auc
            if value > best_value:
                best_value = value
                self._best_T = T
        # print(self._best_T)
    #----------------------------------------------------------------------------------------  
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        drug_dis_te = 1 - drugMatTe
        target_dis_te = 1 - targetMatTe
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(drug_dis_te, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(target_dis_te, return_distance=False)

        U_te = np.zeros((self._n_drugs_te,self.num_factors), dtype=float)
        V_te = np.zeros((self._n_targets_te,self.num_factors), dtype=float)
        etas = self._best_T**np.arange(self.K2)
        if self._cvs == 2: 
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= etas*drugMatTe[d, ii]@self._U[ii, :]/np.sum(drugMatTe[d, ii])
            V_te = self._V
        elif self._cvs == 3:
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= etas*targetMatTe[t, jj]@self._V[jj, :]/np.sum(targetMatTe[t, jj])   
            U_te = self._U
        elif self._cvs == 4:
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= etas*drugMatTe[d, ii]@self._U[ii, :]/np.sum(drugMatTe[d, ii])
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= etas*targetMatTe[t, jj]@self._V[jj, :]/np.sum(targetMatTe[t, jj])   
        scores = U_te@V_te.T
        
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------- 

class NRMFAUC_F4(NRMFAUC_F3):
    """
    Prediction is U@V.T
    Difference with NRMLAUC_F3: 
    choose best T and K2 for computing latent feature of test drug and/or target
    """   
    def __init__(self, K1=5, K2s=[5,7,9], metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=0, is_comLoss=0):
        self.K1 = K1 # used for sparsying similarity matrix
        self.K2s = K2s # used for computing latent features of new drugs/targets
        self.Ts = Ts # candidates decay coefficients
        self.metric= metric # metric to choose T, 0: aupr; 1: aupr+auc
        
        self.num_factors = num_factors # letent feature of drug and target
        self.theta = theta  # learning rate
        self.lambda_d = lambda_d  # coefficient of graph based regularization of U
        self.lambda_t = lambda_t  # coefficient of graph based regularization of V
        self.lambda_r = lambda_r  # coefficient of ||U||_F^2+||V||_F^2 regularization
        self.max_iter = max_iter
        self.seed = seed
        
        self.sfun = sfun # the convex surrogate loss function 
        """(0: square loss; 1: square hinge loss; 2: logistic loss)"""
        self.is_comLoss = is_comLoss # if compute loss for each iteration or not (0: do not compute; 1: compute all Loss; 2 compute AUC only)
        """ compute AUC Loss could lead to out of memory dataset e"""
        
        self.copyable_attrs = ['K1','K2s','metric','Ts','num_factors','theta','lambda_d','lambda_t','lambda_r', 'sfun','is_comLoss', 'max_iter','seed']
    #----------------------------------------------------------------------------------------        
    
    def _get_optimal_T(self, drugMat, targetMat):
        Sd = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        St = targetMat - np.diag(np.diag(targetMat))        
        max_K2 = np.max(self.K2s)
        
        if self._cvs == 2 or self._cvs == 4:
            max_K2 = min(max_K2, self._n_drugs)
            neigh_d = NearestNeighbors(n_neighbors=max_K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(1-Sd, return_distance=False)
        if self._cvs == 3 or self._cvs == 4: 
            max_K2 = min(max_K2, self._n_targets)
            neigh_t = NearestNeighbors(n_neighbors=max_K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(1-St, return_distance=False)
        
        best_value = -1; self._best_T = None; self._best_K2 = None
        for K2 in self.K2s:
            if K2 > max_K2:
                continue
            for T in self.Ts:
                etas = T**np.arange(K2)
                if self._cvs == 2: 
                    U = np.zeros(self._U.shape)
                    for d in range(self._n_drugs):
                        ii = knn_d[d][:K2]
                        sd = Sd[d,ii]
                        U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                    V = self._V
                elif self._cvs == 3:
                    V = np.zeros(self._V.shape)
                    for t in range(self._n_targets):
                        jj = knn_t[t][:K2]
                        st = St[t,jj]
                        V[t,:]= etas*st@self._V[jj, :]/np.sum(st)   
                    U = self._U
                elif self._cvs == 4:
                    U = np.zeros(self._U.shape)
                    for d in range(self._n_drugs):
                        ii = knn_d[d][:K2]
                        sd = Sd[d,ii]
                        U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                    V = np.zeros(self._V.shape)
                    for t in range(self._n_targets):
                        jj = knn_t[t][:K2]
                        st = St[t,jj]
                        V[t,:]= etas*st@self._V[jj, :]/np.sum(st)
                UV = U@V.T
                Y_pred = 1/(1+np.exp(-UV))
                if self.metric == 0:
                    aupr = self._compute_aupr(self._intMat.flatten(), Y_pred.flatten())
                    value = aupr
                elif self.metric == 1:
                    aupr = self._compute_aupr(self._intMat.flatten(), Y_pred.flatten())
                    auc = self._compute_auc(self._intMat.flatten(), Y_pred.flatten())
                    value = aupr + auc
                if value > best_value:
                    best_value = value
                    self._best_T = T
                    self._best_K2 = K2
        # print(self._best_K2,' ',round(self._best_T,1))
    #----------------------------------------------------------------------------------------  -  
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        drug_dis_te = 1 - drugMatTe
        target_dis_te = 1 - targetMatTe
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self._best_K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(drug_dis_te, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self._best_K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(target_dis_te, return_distance=False)

        U_te = np.zeros((self._n_drugs_te,self.num_factors), dtype=float)
        V_te = np.zeros((self._n_targets_te,self.num_factors), dtype=float)
        etas = self._best_T**np.arange(self._best_K2)
        if self._cvs == 2: 
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= etas*drugMatTe[d, ii]@self._U[ii, :]/np.sum(drugMatTe[d, ii])
            V_te = self._V
        elif self._cvs == 3:
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= etas*targetMatTe[t, jj]@self._V[jj, :]/np.sum(targetMatTe[t, jj])   
            U_te = self._U
        elif self._cvs == 4:
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= etas*drugMatTe[d, ii]@self._U[ii, :]/np.sum(drugMatTe[d, ii])
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= etas*targetMatTe[t, jj]@self._V[jj, :]/np.sum(targetMatTe[t, jj])   
        scores = U_te@V_te.T
        
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------- 


class NRMFAUC_F2(NRMFAUC_F):
    """
    Difference with NRMLAUC_F1: sampling pairs based on their ranks of preicting socres
    
    Worse than NRMFAUC_F1 (random sampling)
    using rank based sampling for all pairs < using rank based samplings for only "1" or "0" paris < NRMFAUC_F1 (random sampling)
    
    """   
    def _sampling_pairs(self):
        """
        top ranked "0" pairs and bottom ranked "1" pairs have more chance to be selected
        """
        Y_pred = self._get_prediction_trainset()
        idx_1d = np.argsort(-1*Y_pred, axis=None) #axis=None: the flattened array is used
        idx = np.unravel_index(idx_1d,Y_pred.shape)

        n1 = self._idx1[0].size
        n0 = self._idx0[0].size
        n = n0+n1         
        p1 , p0 = [], [] # sampling probobility for each pair
        re_n1, co_n0 = n1, 0
        # sorted index
        idx1_i, idx1_j = [], []
        idx0_i, idx0_j = [], []
        for i in range(n):
            if self._intMat[idx[0][i],idx[1][i]] == 1:
                idx1_i.append(idx[0][i]) 
                idx1_j.append(idx[1][i])  
                p1.append(co_n0) # co_n0: the number "0" paris having higher predicting scores
                re_n1 -= 1
            else:
                idx0_i.append(idx[0][i]) 
                idx0_j.append(idx[1][i])  
                p0.append(re_n1) # co_n0: the number "1" paris having lower predicting scores
                co_n0 += 1
        
        idx1_i, idx1_j = np.array(idx1_i, dtype=int), np.array(idx1_j, dtype=int)
        idx0_i, idx0_j = np.array(idx0_i, dtype=int), np.array(idx0_j, dtype=int)
        p1, p0 = np.array(p1, dtype=float), np.array(p0, dtype=float)
        p1 += 1 # adding smooth factor
        p0 += 1
        p1 = p1/np.sum(p1)
        p0 = p0/np.sum(p0)
        
        index1 = self._prng.choice(np.arange(n1), size=n, p=p1)
        idx1_s = (idx1_i[index1], idx1_j[index1])
        # """ random samplings for "1" s """
        # index1 = self._prng.choice(np.arange(n1), n)
        # idx1_s = (self._idx1[0][index1], self._idx1[1][index1])
        
        index0 = self._prng.choice(np.arange(n0), size=n, p=p0)
        idx0_s = (idx0_i[index0], idx0_j[index0])
        # # """ random samplings for "0" s """
        # index0 = self._prng.choice(np.arange(n0), n)
        # idx0_s = (self._idx0[0][index0], self._idx0[1][index0])
        
        
        return idx1_s, idx0_s 
    #----------------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------



class NRMFAUC_F1_2(NRMFAUC_F1):
    """
    Difference with NRMLAUC_F1: update U and V corresponding to all-zero rows and columns in interaction matrix, respectively
    
    For S2 gpcr, ic and e dataset are worse, For S3 are slightly better, For S4 there is no change.  
    """ 
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        # self.lambda_t = self.lambda_d # ensure self.lambda_t = self.lambda_d 
        self._intMat = intMat
        self._init_f_loss_and_deriv()
        
        x, y = np.where(intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self._Sd, self._St = drugMat, targetMat
        
        self._construct_neighborhood(drugMat, targetMat) # 
        self._AGD_optimization()  
    #----------------------------------------------------------------------------------------
    
    def _construct_neighborhood(self, drugMat, targetMat):
        dsMat = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1, self._knn_d = self._get_nearest_neighbors(dsMat, self.K1)  # S1 is sparsified durgMat A
            self._DL = self._laplacian_matrix(S1)                   # L^d
            S2, self._knn_t = self._get_nearest_neighbors(tsMat, self.K1)  # S2 is sparsified durgMat B
            self._TL = self._laplacian_matrix(S2)                   # L^t
        else:
            self._DL = self._laplacian_matrix(dsMat)
            self._TL = self._laplacian_matrix(tsMat)
    #---------------------------------------------------------------------------------------- 

    def _get_nearest_neighbors(self, S, size=5):
        """ Eq.8, Eq.9, the S is the similarity matrix whose diagonal elements are 0"""
        m, n = S.shape
        X = np.zeros((m, n))
        neigh = NearestNeighbors(n_neighbors=size, metric='precomputed')
        neigh.fit(np.zeros((m,n)))
        knn_indices = neigh.kneighbors(1-S, return_distance=False) # 1-S is the distance matrix whose diagonal elements are 0
        for i in range(m):
            ii = knn_indices[i]
            X[i, ii] = S[i, ii]
        return X, knn_indices
    #---------------------------------------------------------------------------------------- 
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        # update U and V corresponding to all-zero rows and columns in interaction matrix, respectively 
        if self._cvs == 3:
            for d in range(self._n_drugs):
                if d not in self.train_drugs:
                    ii = self._knn_d[d]
                    self._U[d,:] = np.dot(self._Sd[d, ii], self._U[ii, :])/np.sum(self._Sd[d, ii])
        if self._cvs == 2:           
            for t in range(self._n_targets):
                if t not in self.train_targets:
                    jj = self._knn_t[t]
                    self._V[t:] = np.dot(self._St[t, jj], self._V[jj, :])/np.sum(self._St[t, jj])  
        
        
        drug_dis_te = 1 - drugMatTe
        target_dis_te = 1 - targetMatTe
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(drug_dis_te, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(target_dis_te, return_distance=False)

        U_te = np.zeros((self._n_drugs_te,self.num_factors), dtype=float)
        V_te = np.zeros((self._n_targets_te,self.num_factors), dtype=float)
        if self._cvs == 2: 
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= np.dot(drugMatTe[d, ii], self._U[ii, :])/np.sum(drugMatTe[d, ii])
            V_te = self._V
        elif self._cvs == 3:
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= np.dot(targetMatTe[t, jj], self._V[jj, :])/np.sum(targetMatTe[t, jj])   
            U_te = self._U
        elif self._cvs == 4:
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= np.dot(drugMatTe[d, ii], self._U[ii, :])/np.sum(drugMatTe[d, ii])
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= np.dot(targetMatTe[t, jj], self._V[jj, :])/np.sum(targetMatTe[t, jj])   
        scores = U_te@V_te.T
        # exp_s = np.exp(scores)
        # scores =exp_s/(1+exp_s)
        
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------    




    
class NRMFAUC2_F1(NRMFAUC2):
    """
    Difference with NRMLAUC2: use random undersampling pairs and run faster 
    
    """   
    def _sampling_pairs(self):
        """ random sampling pairs"""
        n1 = self._idx1[0].size
        n0 = self._idx0[0].size
        n = n0+n1 
        index1 = self._prng.choice(np.arange(n1), n)
        idx1_s = (self._idx1[0][index1], self._idx1[1][index1])
        
        index0 = self._prng.choice(np.arange(n0), n)
        idx0_s = (self._idx0[0][index0], self._idx0[1][index0])
        
        return idx1_s, idx0_s 
    #----------------------------------------------------------------------------------------

    def _deriv_AUCLoss(self, U, V, idx1_s, idx0_s, Y_pred, idx_flag, func_loss_deriv):
        """compute the derivatives of AUC loss w.r.t U or V, idx_flag = [0,1] for U and [1,0] for V        
        """
        n,m = Y_pred.shape
        deriv_U = np.zeros(U.shape)
        P = Y_pred*(1-Y_pred)
        
        if idx_flag[0] == 1:  #deriv of V, exchange the tuple elements in idx1_s, idx0_s
            i1, j1 = idx1_s
            idx1_s = j1, i1
            i0, j0 = idx0_s
            idx0_s = j0, i0
        
        Y1 = Y_pred[idx1_s]
        Y0 = Y_pred[idx0_s]
        diff = Y1-Y0
        
        
        du1 = (func_loss_deriv(diff)*P[idx1_s])[:,None]*V[idx1_s[1]]
        du0 = (-func_loss_deriv(diff)*P[idx0_s])[:,None]*V[idx0_s[1]] 
        
        deriv_U1 = np.array([du1[idx1_s[0]==i].sum(axis=0) for i in range(n)]) # 2d array shape=(n, num_factors), deriv_U1[index] is the sum of rows of du1 whose corresponding i=index    
        deriv_U0 = np.array([du0[idx0_s[0]==i].sum(axis=0) for i in range(n)])
        deriv_U = deriv_U1+deriv_U0
        
        # this part of code has same function with above three lines and similar running time 
        # for i1 in range(n):
        #     du = du1[idx1_s[0]==i1]
        #     sdu = np.sum(du, axis=0)
        #     deriv_U[i1] += sdu
            
        #     du = du0[idx0_s[0]==i1]
        #     sdu = np.sum(du, axis=0)
        #     deriv_U[i1] += sdu
            
        return deriv_U
    #----------------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------        
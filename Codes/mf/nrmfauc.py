import numpy as np
from sklearn.neighbors import NearestNeighbors
from base.inbase import InductiveModelBase

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score


"""
Neighborhood Regularized Matrix Factorization for optimizting micro AUC

"""
class NRMFAUC(InductiveModelBase):
    def __init__(self, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=0, is_comLoss=0):
        self.K1 = K1 # used for sparsying similarity matrix
        self.K2 = K2 # used for computing latent features of new drugs/targets
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
        
        self.copyable_attrs = ['K1','K2','num_factors','theta','lambda_d','lambda_t','lambda_r', 'sfun','is_comLoss', 'max_iter','seed']
    #----------------------------------------------------------------------------------------        
    
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        # self.lambda_t = self.lambda_d # ensure self.lambda_t = self.lambda_d 
        self._intMat = intMat
        self._init_f_loss_and_deriv()
        
        self._construct_neighborhood(drugMat, targetMat) # 
        self._AGD_optimization()  
    #----------------------------------------------------------------------------------------


    def _AGD_optimization(self):
        self._prng = np.random.RandomState(self.seed)
        self._U = np.sqrt(1/float(self.num_factors))*self._prng.normal(size=(self._n_drugs, self.num_factors))
        self._V = np.sqrt(1/float(self.num_factors))*self._prng.normal(size=(self._n_targets, self.num_factors))
        
        
        self._idx1 = np.where(self._intMat==1) # (array([0, 1], dtype=int64), array([2, 0], dtype=int64))
        self._idx0 = np.where(self._intMat==0) # (array([0, 0, 1, 1], dtype=int64), array([0, 1, 1, 2], dtype=int64))
        # if intMat = np.array([[1,1,0],[0,1,1]])
        # self._idx1 = (array([0, 1], dtype=int64), array([2, 0], dtype=int64))
        # self._idx0 = (array([0, 0, 1, 1], dtype=int64), array([0, 1, 1, 2], dtype=int64))    
        
        
        if self.is_comLoss == 1:
            # the 1d index is for compute AUC loss
            idx1_1d = np.ravel_multi_index(self._idx1,self._intMat.shape) # array([0,1,4,5])
            idx0_1d = np.ravel_multi_index(self._idx0,self._intMat.shape) # array([2,3])
            i1_ = np.tile(idx1_1d,len(idx0_1d))  # i1_ = array([0,1,4,5,0,1,4,5])
            i0_ = np.repeat(idx0_1d,len(idx1_1d)) # i0_ = array([2,2,2,2,3,3,3,3])
            pair_idx_1d = (i0_,i1_) # index of negative and positive paris 
        

        du_sum = np.zeros(self._U.shape)
        dv_sum = np.zeros(self._V.shape)
        if self.is_comLoss == 1:
            last_loss, lauc, r_r, r_d, r_t , auc_val= self._compute_loss(pair_idx_1d, self._f_loss)  
            print('\t',round(last_loss,6),'\t', round(lauc,6),'\t',round(r_r,6),'\t',round(r_d,6),'\t',round(r_t,6),'\t', round(auc_val,6))
        
        for iteration in range(self.max_iter):            
            idx1_s, idx0_s = self._sampling_pairs()
            
            Y_pred = self._get_prediction_trainset()
            deriv_U = self.lambda_r*self._U #/self._n_drugs
            deriv_U += self.lambda_d*self._DL@self._U #/self._n_drugs
            du = self._deriv_AUCLoss(self._U, self._V, idx1_s, idx0_s, Y_pred, [0,1], self._f_loss_deriv)
            du += deriv_U
            
            du_sum += np.square(du)
            vec_step_size_d = self.theta / np.sqrt(du_sum) 
            self._U -= vec_step_size_d * du
            # self._U -= self.theta*np.power(0.7,iteration) * du
            
            Y_pred = self._get_prediction_trainset()
            deriv_V = self.lambda_r*self._V #/self._n_targets
            deriv_V += self.lambda_t*self._TL@self._V #/self._n_targets
            dv = self._deriv_AUCLoss(self._V, self._U, idx1_s, idx0_s, Y_pred.T, [1,0], self._f_loss_deriv)
            dv += deriv_V
            
            dv_sum += np.square(dv)
            vec_step_size = self.theta / np.sqrt(dv_sum)
            self._V -= vec_step_size * dv
            # self._V -= self.theta*np.power(0.7,iteration) * dv
            
            if self.is_comLoss == 1:
                curr_loss, lauc, r_r, r_d, r_t, auc_val = self._compute_loss(pair_idx_1d, self._f_loss)  
                delta_loss = (curr_loss-last_loss)/abs(last_loss)
                print(iteration,'\t',round(curr_loss,6),'\t', round(lauc,6),'\t',round(r_r,6),'\t',round(r_d,6),'\t',round(r_t,6),'\t',round(auc_val,6))
                if abs(delta_loss) < 1e-5: #or delta_loss>0:
                    break
                last_loss = curr_loss
            elif self.is_comLoss ==2:
                Y_pred = self._get_prediction_trainset()
                y_pred = Y_pred.flatten()
                auc_val = self._compute_auc(self._intMat.flatten(), y_pred)
                print(iteration,'\t','\t',round(auc_val,6))
            # print(iteration)        
    #----------------------------------------------------------------------------------------
    
    def _init_f_loss_and_deriv(self):
        if self.sfun == 0:
            self._f_loss = self.fSL
            self._f_loss_deriv = self.fSL_deriv
        elif self.sfun == 1:
            self._f_loss = self.fSHL
            self._f_loss_deriv = self.fSHL_deriv
        elif self.sfun == 2:
            self._f_loss = self.fLL
            self._f_loss_deriv = self.fLL_deriv
        else:
            self._f_loss = None
            self._f_loss_deriv = None          
    #----------------------------------------------------------------------------------------  
    
    def fSL(self, x):
        """ square loss function"""
        return np.power(1-x,2)/2
    #----------------------------------------------------------------------------------------
    
    def fSL_deriv(self, x):
        """ derivative of square loss function"""
        return x-1
    #---------------------------------------------------------------------------------------- 
    
    def fSHL(self, x):
        """ square hinge loss function"""
        y = 1-x
        y[y<0] = 0
        return np.power(y,2)         
    #----------------------------------------------------------------------------------------
    
    def fSHL_deriv(self, x):
        """ derivative of square hinge loss function"""
        y=x-1
        if isinstance(y, np.ndarray):    
            y[y>0] = 0
        else:
            y = min(y,0)
        return y
    #---------------------------------------------------------------------------------------- 
    
    def fLL(self, x):
        """ logistic loss function"""
        return np.log(1+np.exp(-x))
    #----------------------------------------------------------------------------------------
    
    def fLL_deriv(self, x):
        """ derivative of logistic loss function"""
        return -1/(1+np.exp(x))
    #----------------------------------------------------------------------------------------  
    
    def _get_prediction_trainset(self):
        return self._U@self._V.T
    
    def _compute_loss(self, pair_idx_1d, func_loss):
        Y_pred = self._get_prediction_trainset()
        
        y_pred = Y_pred.flatten() # transform Y_pred to 1d vector
        # print(np.max(y_pred), '\t', np.min(y_pred))
        diff = y_pred[pair_idx_1d[1]] - y_pred[pair_idx_1d[0]] # the u_i*v_j - u_h*v_l for all pairs where Y_ij=1 and Y_hl=0
        diff1 = func_loss(diff)
        lauc = diff1.sum() #diff1.mean()

        r_r = 0.5*self.lambda_r*np.square(np.linalg.norm(self._U,'fro'))#/self._n_drugs 
        r_r += 0.5*self.lambda_r*np.square(np.linalg.norm(self._V,'fro'))#/self._n_targets
        r_d = 0.5*self.lambda_d*np.trace(self._U.T@self._DL@self._U)#/self._n_drugs
        r_t = 0.5*self.lambda_t*np.trace(self._V.T@self._TL@self._V)#/self._n_targets
        
        loss = lauc + r_r + r_d + r_t
        
        auc_val = self._compute_auc(self._intMat.flatten(), y_pred)

        return loss, lauc, r_r, r_d, r_t, auc_val
    #----------------------------------------------------------------------------------------
    
    def _compute_auc(self, labels_1d, scores_1d):
        fpr, tpr, thr = roc_curve(labels_1d, scores_1d)
        auc_val = auc(fpr, tpr)
        return auc_val
    #----------------------------------------------------------------------------------------
    

    def _sampling_pairs(self):
        """ using all pairs"""
        return self._idx1, self._idx0 
    #----------------------------------------------------------------------------------------        

    def _deriv_AUCLoss(self, U, V, idx1_s, idx0_s, Y_pred, idx_flag, func_loss_deriv):
        """compute the derivatives of AUC loss w.r.t U or V, idx_flag = [0,1] for U and [1,0] for V        
        """
        n,m = Y_pred.shape
        deriv_U = np.zeros(U.shape)
        n1 = len(idx1_s[0]) # number of "1"s in Y
        n0 = len(idx0_s[0]) # number of "0"s in Y
        
        for i1 in range(n1):
            i,j = idx1_s[idx_flag[0]][i1], idx1_s[idx_flag[1]][i1]
            for i0 in range(n0):
                h,l = idx0_s[idx_flag[0]][i0], idx0_s[idx_flag[1]][i0]
                diff = Y_pred[i,j]-Y_pred[h,l]
                if i!=h:
                    deriv_U[i] += func_loss_deriv(diff)*V[j]
                    deriv_U[h] -= func_loss_deriv(diff)*V[l]
                else:
                    deriv_U[i] += func_loss_deriv(diff)*(V[j]-V[l])
        # counts of update in each row of deriv_U[i]
        # c1 = self._intMat.sum(axis=idx_flag[1]) 
        # c0 = m-c1
        # z =  c1*n0+c0*n1-c1*c0
        # deriv_U /= z[:,None]
        return deriv_U
    #----------------------------------------------------------------------------------------
    
    def _deriv_AUCLoss2(self, U, V, idx1_s, idx0_s, Y_pred, idx_flag, func_loss_deriv):
        """compute the derivatives of AUC loss w.r.t U or V, idx_flag = [0,1] for U and [1,0] for V        
        """
        n,m = Y_pred.shape
        deriv_U1 = np.zeros(U.shape) # 1st line in Eq. derivative of u_i
        deriv_U2 = np.zeros(U.shape) # 2nd line in Eq. derivative of u_i
        deriv_U3 = np.zeros(U.shape) # 3nd line in Eq. derivative of u_i
        n1 = len(idx1_s[0]) # number of "1"s in Y
        n0 = len(idx0_s[0]) # number of "0"s in Y
        
        for i1 in range(n1):
            i,j = idx1_s[idx_flag[0]][i1], idx1_s[idx_flag[1]][i1]
            for i0 in range(n0):
                h,l = idx0_s[idx_flag[0]][i0], idx0_s[idx_flag[1]][i0]
                diff = Y_pred[i,j]-Y_pred[h,l]
                if i!=h:
                    deriv_U1[i] += func_loss_deriv(diff)*V[j]
                    deriv_U2[h] -= func_loss_deriv(diff)*V[l]
                else:
                    deriv_U3[i] += func_loss_deriv(diff)*(V[j]-V[l])
        # counts of update in each row of deriv_U[i]
        c1 = self._intMat.sum(axis=idx_flag[1]) 
        c0 = m - c1
        z1 = c1*(n0-c0)
        z2 = (n1-c1)*c0
        z3 = c1*c0
        deriv_U1 /= z1[:,None]
        deriv_U2 /= z2[:,None]
        deriv_U3 /= z3[:,None]
        
        deriv_U = None
        return deriv_U           
    #----------------------------------------------------------------------------------------


    def _construct_neighborhood(self, drugMat, targetMat):
        # construct the laplocian matrices
        dsMat = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self._get_nearest_neighbors(dsMat, self.K1)  # S1 is sparsified durgMat A
            self._DL = self._laplacian_matrix(S1)                   # L^d
            S2 = self._get_nearest_neighbors(tsMat, self.K1)  # S2 is sparsified durgMat B
            self._TL = self._laplacian_matrix(S2)                   # L^t
        else:
            self._DL = self._laplacian_matrix(dsMat)
            self._TL = self._laplacian_matrix(tsMat)
    #----------------------------------------------------------------------------------------
            
    def _laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L
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
        return X
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
#-------------------------------------------------------------------------------------------------------------------
        
class NRMFAUC2(NRMFAUC):
    """
    Difference with NRMLAUC: the prediction is exp(U@V.T)/(1+exp(U@V.T)) instead of U@V.T
    """
    def _get_prediction_trainset(self):
        Y_pred = np.exp(self._U@self._V.T)
        Y_pred = Y_pred/(1+Y_pred)
        return Y_pred
        
    def _deriv_AUCLoss(self, U, V, idx1_s, idx0_s, Y_pred, idx_flag, func_loss_deriv):
        """compute the derivatives of AUC loss w.r.t U or V, idx_flag = [0,1] for U and [1,0] for V        
        """
        n,m = Y_pred.shape
        deriv_U = np.zeros(U.shape)
        n1 = len(idx1_s[0]) # number of "1"s in Y
        n0 = len(idx0_s[0]) # number of "0"s in Y
        
        P = Y_pred*(1-Y_pred)
        
        for i1 in range(n1):
            i,j = idx1_s[idx_flag[0]][i1], idx1_s[idx_flag[1]][i1]
            for i0 in range(n0):
                h,l = idx0_s[idx_flag[0]][i0], idx0_s[idx_flag[1]][i0]
                diff = Y_pred[i,j]-Y_pred[h,l]
                if i!=h:
                    deriv_U[i] += func_loss_deriv(diff)*P[i,j]*V[j]
                    deriv_U[h] -= func_loss_deriv(diff)*P[h,l]*V[l]
                else:
                    deriv_U[i] += func_loss_deriv(diff)*(P[i,j]*V[j]-P[i,l]*V[l])
        # counts of update in each row of deriv_U[i]
        # c1 = self._intMat.sum(axis=idx_flag[1]) 
        # c0 = m-c1
        # z =  c1*n0+c0*n1-c1*c0
        # deriv_U /= z[:,None]
        return deriv_U
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
        exp_s = np.exp(U_te@V_te.T)
        scores =exp_s/(1+exp_s)
        
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    
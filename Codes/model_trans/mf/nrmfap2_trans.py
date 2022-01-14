import numpy as np
from sklearn.neighbors import NearestNeighbors
from base.transbase import TransductiveModelBase

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score




class NRMFAP2_TRANS(TransductiveModelBase):
    """
    Difference with NRMFAP2:
    1. add mask matrix self._Omega to distingguish the traning and test pairs for compute loss and deriv_AP
    2. For S3, S4 and S4, 
        add '_update_train_knns' function to obtain k neareast training drugs/targets, 
        these training kNNs are used to choose optiaml T and infer U_te/V_te for prediction                   
    """
    def __init__(self, K=5, K2=5, metric=0, Ts=np.arange(0,1.1,0.1), num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=True):
        self.K = K # used for sparsying similarity matrix and computing latent features of new drugs/targets
        self.K2 = K2 # K2 used for compute test drug/target latent feature
        self.Ts = Ts # candidates decay coefficients
        self.metric= metric # metric to choose T, 0: aupr; 1: aupr+auc
        
        self.num_factors = num_factors # letent feature of drug and target
        self.theta = theta  # learning rate
        self.lambda_d = lambda_d  # coefficient of graph based regularization of U
        self.lambda_t = lambda_t  # coefficient of graph based regularization of V
        self.lambda_r = lambda_r  # coefficient of ||U||_F^2+||V||_F^2 regularization
        self.M = M # M-1 is the number of intervals for predicting scores
        self.max_iter = max_iter
        self.seed = seed
        
        self.is_comLoss = is_comLoss # if compute loss for each iteration or not (0: do not compute; 1: compute all Loss; 2 compute AUC only)
        self.isAGD = isAGD
        
        self.copyable_attrs = ['K','K2','metric','Ts','num_factors','theta','lambda_d','lambda_t','lambda_r','M','max_iter','seed','is_comLoss','isAGD']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, test_indices, cvs)
        self._Y = intMat

        self._construct_neighborhood(drugMat, targetMat) # 
        self._n1 = np.sum(self._Y) # the number of "1"s in Y 
        self._d = 1.0/(self.M-1) # Delta: the with of intervals
        self._b = 1-np.arange(self.M)*self._d # center of intervals
        
        """New!!! compute mask Matrix Omega Ω, where the traning pairs are 1 and test paris are 0"""
        self._initial_Omega(self._Y)
        
        self._AGD_optimization() 
        if self._cvs != 1:
            self._knn_d, self._knn_t = self._update_train_knns(drugMat, targetMat)        
            self._get_optimal_T(drugMat, targetMat)
            # self._best_T =1.0
            
        if self._cvs == 1: 
            U_te = self._U
            V_te = self._V
        elif self._cvs  == 2:
            test_d = self._test_indices 
            U_te = self._get_U_te(test_d, drugMat, self._U, self._knn_d)
            V_te = self._V
        elif self._cvs  == 3:
            test_t = self._test_indices      
            U_te = self._U
            V_te = self._get_U_te(test_t, targetMat, self._V, self._knn_t)
        elif self._cvs  == 4:
            test_d,test_t = self._test_indices
            U_te = self._get_U_te(test_d, drugMat, self._U, self._knn_d)
            V_te = self._get_U_te(test_t, targetMat, self._V, self._knn_t)
        scores = U_te@V_te.T
        scores = 1/(1+np.exp(-scores))
        S_te = self._get_test_scores(scores)
        # S_te[S_te==np.inf] = 0.0
        # S_te[np.isnan(S_te)] = 0.0
        return S_te
    #----------------------------------------------------------------------------------------

    def _get_U_te(self, test_row_indices, S, U, knn):
        U_te = np.copy(U)
        etas = self._best_T**np.arange(self.K2)
        for d in test_row_indices:
            ii = knn[d]
            sd = S[d,ii]
            U_te[d,:]= etas*sd@U[ii, :]/np.sum(sd)
        # U_te = U  #!!! using U is better
        return U_te
    #----------------------------------------------------------------------------------------       

    def _update_train_knns(self, drugMat, targetMat):
        """ the knn of each drug/target from training drugs/targets, only used for cvs=2,3,4"""
        Sd = drugMat - np.diag(np.diag(drugMat))
        St = targetMat - np.diag(np.diag(targetMat))
        if self._cvs == 2:
            test_d = self._test_indices # test drug indices 
            Sd[:,test_d] = 0                        
        elif self._cvs == 3:            
            test_t = self._test_indices            
            St[:,test_t] = 0
        elif self._cvs == 4: 
            test_d,test_t = self._test_indices
            Sd[:,test_d] = 0
            St[:,test_t] = 0
        
        neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
        neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
        knn_d = neigh_d.kneighbors(1-Sd, return_distance=False)
        
        neigh_t = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
        neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
        knn_t = neigh_t.kneighbors(1-St, return_distance=False)
        return knn_d, knn_t
    #----------------------------------------------------------------------------------------

    def _get_optimal_T(self, Sd, St):      
        best_value = -1; self._best_T = None
        for T in self.Ts:
            etas = T**np.arange(self.K2)
            if self._cvs == 2: 
                U = np.zeros(self._U.shape)
                for d in range(self._n_drugs):
                    ii = self._knn_d[d]
                    sd = Sd[d,ii]
                    U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                V = self._V
            elif self._cvs == 3:
                V = np.zeros(self._V.shape)
                for t in range(self._n_targets):
                    jj = self._knn_t[t]
                    st = St[t,jj]
                    V[t,:]= etas*st@self._V[jj, :]/np.sum(st)   
                U = self._U
            elif self._cvs == 4:
                U = np.zeros(self._U.shape)
                for d in range(self._n_drugs):
                    ii = self._knn_d[d]
                    sd = Sd[d,ii]
                    U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                V = np.zeros(self._V.shape)
                for t in range(self._n_targets):
                    jj = self._knn_t[t]
                    st = St[t,jj]
                    V[t,:]= etas*st@self._V[jj, :]/np.sum(st)
            UV = U@V.T
            Y_pred = 1/(1+np.exp(-UV))
            Y_pred = self._get_train_scores(Y_pred)
            Y_train = self._get_train_scores(self._Y)
            if self.metric == 0:
                auc = self._compute_auc(Y_train.flatten(), Y_pred.flatten())
                value = auc
            elif self.metric == 1:
                aupr = self._compute_aupr(Y_train.flatten(), Y_pred.flatten())
                auc = self._compute_auc(Y_train.flatten(), Y_pred.flatten())
                value = aupr + auc
            if value > best_value:
                best_value = value
                self._best_T = T
        # print(self._best_T)
    #---------------------------------------------------------------------------------------- 

    def _AGD_optimization(self):
        self._prng = np.random.RandomState(self.seed)
        self._U = np.sqrt(1/float(self.num_factors))*self._prng.normal(size=(self._n_drugs, self.num_factors))
        self._V = np.sqrt(1/float(self.num_factors))*self._prng.normal(size=(self._n_targets, self.num_factors))

        if self.isAGD:
            du_sum = np.zeros(self._U.shape)
            dv_sum = np.zeros(self._V.shape)
        else:
            current_theta = self.theta
        psi, psi_ = None, None
        
        if self.is_comLoss == 1:
            last_loss, l_ap, r_r, r_d, r_t , aupr,psi, psi_ = self._compute_loss()  
            # print('\t',round(last_loss,6),'\t', round(l_ap,6),'\t',round(r_r,6),'\t',round(r_d,6),'\t',round(r_t,6),'\t', round(aupr,6))
        
        for iteration in range(self.max_iter):            
            Y_pred = self._get_prediction_trainset()
            deriv_U = self.lambda_r*self._U #/self._n_drugs
            deriv_U += self.lambda_d*self._DL@self._U #/self._n_drugs
            du = self._deriv_AP(self._Y, Y_pred, self._U, self._V, self._Omega, psi, psi_) 
            du = -du + deriv_U
            if self.isAGD:
                du_sum += np.square(du)
                vec_step_size_d = self.theta / np.sqrt(du_sum) 
                self._U -= vec_step_size_d * du
            else:
                self._U -= current_theta* du
            
            Y_pred = self._get_prediction_trainset()
            deriv_V = self.lambda_r*self._V #/self._n_targets
            deriv_V += self.lambda_t*self._TL@self._V #/self._n_targets
            dv = self._deriv_AP(self._Y.T, Y_pred.T, self._V, self._U, self._Omega.T)
            dv = -dv + deriv_V
            if self.isAGD:
                dv_sum += np.square(dv)
                vec_step_size = self.theta / np.sqrt(dv_sum)
                self._V -= vec_step_size * dv
            else:
                self._V -= current_theta * dv
            
            if self.is_comLoss == 1:
                curr_loss, l_ap, r_r, r_d, r_t, aupr, psi, psi_ = self._compute_loss()  
                delta_loss = (curr_loss-last_loss)/abs(last_loss)
                # print(iteration,'\t',round(curr_loss,6),'\t', round(l_ap,6),'\t',round(r_r,6),'\t',round(r_d,6),'\t',round(r_t,6),'\t',round(aupr,6))
                if self.isAGD:
                    if abs(delta_loss) < 1e-6:
                        break
                else:
                    if delta_loss>0: # abs(delta_loss) < 1e-5: 
                        current_theta *= 0.9
                    if abs(delta_loss) < 1e-6:
                        break
                last_loss = curr_loss
            elif self.is_comLoss ==2:
                Y_pred = self._get_prediction_trainset()
                y_pred = Y_pred.flatten()
                aupr = self._compute_aupr(self._intMat.flatten(), y_pred)
                print(iteration,'\t','\t',round(aupr,6))
            # print(iteration)
    #----------------------------------------------------------------------------------------
    
    def _get_prediction_trainset(self):
        Y_pred = 1/(1+np.exp(-1*self._U@self._V.T))
        return Y_pred
    #----------------------------------------------------------------------------------------
    
    def _compute_loss(self):
        Y_pred = self._get_prediction_trainset()
        yp_max = np.amax(Y_pred)
        yp_min = np.amin(Y_pred)
        h_min = int(self.M-1-int(yp_max/self._d+1)) # yp_max=0.61 ==> h_min=7, self._b[h_min]=0.65 
        h_min = max(h_min,0)
        h_max = int(self.M-1-int(yp_min/self._d-1)) # yp_min=0.36 ==> h_max-1=13, self._b[h_max-1]=0.35
        h_max = min(self.M, h_max)
        h_range = range(h_min,h_max)
        
        """compute ψ_h and bar_ψ_h """
        psi = np.zeros(self.M, dtype=float)
        psi_ = np.zeros(self.M, dtype=float)
        for h in h_range: 
            X = self._f_delta(Y_pred, h) # δ(Y^,h) 
            """!!!New only training pairs are counted"""
            X = X*self._Omega 
            psi[h] = np.sum(X*self._Y) # ψ[δ(Y^,h)⊙Y]
            psi_[h] = np.sum(X) # ψ[δ(Y^,h)]
        sum_psi = sum_psi_= 0
        ap = 0
        for h in h_range:
            if psi_[h] == 0:
                continue            
            else:
                sum_psi += psi[h]
                sum_psi_ += psi_[h]
                prec = sum_psi/sum_psi_
                recall = psi[h]/self._n1 
                ap += prec*recall
        l_ap = (1 - ap)*self._n1  # for the acutual loss times self._n1

        r_r = 0.5*self.lambda_r*np.square(np.linalg.norm(self._U,'fro'))#/self._n_drugs 
        r_r += 0.5*self.lambda_r*np.square(np.linalg.norm(self._V,'fro'))#/self._n_targets
        r_d = 0.5*self.lambda_d*np.trace(self._U.T@self._DL@self._U)#/self._n_drugs
        r_t = 0.5*self.lambda_t*np.trace(self._V.T@self._TL@self._V)#/self._n_targets
        
        loss = l_ap + r_r + r_d + r_t
        aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
        # print(psi.sum(),'\t', psi_.sum()) # psi.sum()=n1, psi_.sum()=n*m
        return loss, l_ap, r_r, r_d, r_t, aupr, psi, psi_
    #----------------------------------------------------------------------------------------
    
    def _compute_auc(self, labels_1d, scores_1d):
        fpr, tpr, thr = roc_curve(labels_1d, scores_1d)
        auc_val = auc(fpr, tpr)
        return auc_val
    #----------------------------------------------------------------------------------------
    def _compute_aupr(self, labels_1d, scores_1d):
        aupr_val = average_precision_score(labels_1d,scores_1d)
        if np.isnan(aupr_val):
            aupr_val=0.0
        return aupr_val    

        
    def _f_delta(self, Y,h):
        # δ(Y^,h)
        temp = 1-np.abs(Y-self._b[h])/self._d
        return np.maximum(temp,0)
    #----------------------------------------------------------------------------------------
    
    def _f_delta_derive(self, Y,h):
        # δ'(Y^,h) = -( sign(Y^-b_{h'}) ⊙[[ |Y^-b_{h'}|<=Δ ]] )/ Δ
        Deriv = np.zeros(Y.shape, dtype=float)
        C = Y - self._b[h] # the conditon matrix
        Deriv[(C>0) & (C<=self._d)] = -1.0/self._d
        Deriv[(C<0) & (C>=-self._d)] = 1.0/self._d
        return Deriv
    #----------------------------------------------------------------------------------------
    
    def _deriv_AP(self, Y, Y_pred, U, V, Omega, psi=None, psi_=None):
        """compute the derivatives of AP w.r.t U or V  
        """      
        yp_max = np.amax(Y_pred)
        yp_min = np.amin(Y_pred)
        h_min = int(self.M-1-int(yp_max/self._d+1)) # yp_max=0.61 ==> h_min=7, self._b[h_min]=0.65 
        h_min = max(h_min,0)
        h_max = int(self.M-1-int(yp_min/self._d-1)) # yp_min=0.36 ==> h_max-1=13, self._b[h_max-1]=0.35
        h_max = min(self.M, h_max)
        h_range = range(h_min,h_max)
        
        """compute ψ_h and bar_ψ_h """
        if psi is None or psi_ is None:
            psi = np.zeros(self.M, dtype=float)
            psi_ = np.zeros(self.M, dtype=float) 
            for h in h_range:
                X = self._f_delta(Y_pred, h) # δ(Y^,h) 
                """!!!New only training pairs are counted"""
                X = X*Omega 
                psi[h] = np.sum(X*Y) # ψ[δ(Y^,h)⊙Y]
                psi_[h] = np.sum(X) # ψ[δ(Y^,h)]
            
        """ compute Z_h, the Z_h.shape = (n,m) """
        Z = np.zeros(shape=(self.M, Y_pred.shape[0], Y_pred.shape[1]) ,dtype=float)
        for h in h_range:
            D_delta = self._f_delta_derive(Y_pred, h)
            """!!!New only training pairs are counted"""
            Z[h] = D_delta * Y_pred *(1-Y_pred)*Omega 

        deriv_U = np.zeros(U.shape, dtype=float)
        SUM_YZV = np.zeros(U.shape, dtype=float)
        SUM_ZV = np.zeros(U.shape, dtype=float)
        sum_psi = sum_psi_= 0
        for h in h_range:
            if psi_[h] == 0:
                continue    
            else:
                YZV = Y*Z[h]@V
                ZV = Z[h]@V
                SUM_YZV += YZV
                SUM_ZV += ZV
                sum_psi += psi[h]
                sum_psi_ += psi_[h]
                
                deriv_U += psi[h] * (SUM_YZV*sum_psi_ - SUM_ZV*sum_psi)/(sum_psi_*sum_psi_)
                deriv_U += sum_psi/sum_psi_*YZV
                # deriv_U /= self._n1
        return deriv_U
    #----------------------------------------------------------------------------------------
    
    def _construct_neighborhood(self, drugMat, targetMat):
        # construct the laplocian matrices
        dsMat = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K > 0:
            S1 = self._get_nearest_neighbors(dsMat, self.K)  # S1 is sparsified durgMat A
            self._DL = self._laplacian_matrix(S1)                   # L^d
            S2 = self._get_nearest_neighbors(tsMat, self.K)  # S2 is sparsified durgMat B
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

#-------------------------------------------------------------------------------------------------------------------
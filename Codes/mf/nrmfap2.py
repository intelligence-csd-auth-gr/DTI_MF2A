import numpy as np
from sklearn.neighbors import NearestNeighbors
from mf.nrmfap import NRMFAP

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score


    
class NRMFAP2(NRMFAP):
    """ 
    Difference with NRMFAP: Choose best T for computing latent feature of test drug and/or target
    better than NRMFAP
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
        """ compute AUC Loss could lead to out of memory dataset e"""
        self.isAGD = isAGD
        
        self.copyable_attrs = ['K','K2','metric','Ts','num_factors','theta','lambda_d','lambda_t','lambda_r','M','max_iter','seed','is_comLoss','isAGD']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self._Y = intMat

        self._construct_neighborhood(drugMat, targetMat) # 
        self._n1 = np.sum(self._Y) # the number of "1"s in Y 
        self._d = 1.0/(self.M-1) # Delta: the with of intervals
        self._b = 1-np.arange(self.M)*self._d # center of intervals
        self._AGD_optimization() 
        
        self._get_optimal_T(drugMat, targetMat)
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
            UV = U@V.T
            Y_pred = 1/(1+np.exp(-UV))
            if self.metric == 0:
                aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                value = aupr
            elif self.metric == 1:
                aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                auc = self._compute_auc(self._Y.flatten(), Y_pred.flatten())
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
        scores = 1/(1+np.exp(-scores))
        
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------


class NRMFAP4(NRMFAP2):
    """ 
    Difference with NRMFAP2: Prediction is UV' instead of 1/[1+exp(-UV')]
    Worse than NRMFAP2
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
        """ compute AUC Loss could lead to out of memory dataset e"""
        self.isAGD = isAGD
        
        self.copyable_attrs = ['K','K2','metric','Ts','num_factors','theta','lambda_d','lambda_t','lambda_r','M','max_iter','seed','is_comLoss','isAGD']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self._Y = intMat

        self._construct_neighborhood(drugMat, targetMat) # 
        self._n1 = np.sum(self._Y) # the number of "1"s in Y 
        # self._d = 1.0/(self.M-1) # Delta: the with of intervals
        # self._b = 1-np.arange(self.M)*self._d # center of intervals
        self._AGD_optimization()
        self._get_optimal_T(drugMat, targetMat)
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
            du = self._deriv_AP(self._Y, Y_pred, self._U, self._V, psi, psi_) 
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
            dv = self._deriv_AP(self._Y.T, Y_pred.T, self._V, self._U)
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


    def _deriv_AP(self, Y, Y_pred, U, V, psi=None, psi_=None):
        """compute the derivatives of AP w.r.t U or V  
        """      
        # yp_max = np.amax(Y_pred)
        # yp_min = np.amin(Y_pred)
        # h_min = int(self.M-1-int(yp_max/self._d+1)) # yp_max=0.61 ==> h_min=7, self._b[h_min]=0.65 
        # h_min = max(h_min,0)
        # h_max = int(self.M-1-int(yp_min/self._d-1)) # yp_min=0.36 ==> h_max-1=13, self._b[h_max-1]=0.35
        # h_max = min(self.M, h_max)
        # h_range = range(h_min,h_max)
        h_range = range(self.M)
        
        """compute ??_h and bar_??_h """
        if psi is None or psi_ is None:
            psi = np.zeros(self.M, dtype=float)
            psi_ = np.zeros(self.M, dtype=float) 
            for h in h_range:
                X = self._f_delta(Y_pred, h) # ??(Y^,h) 
                psi[h] = np.sum(X*Y) # ??[??(Y^,h)???Y]
                psi_[h] = np.sum(X) # ??[??(Y^,h)]
            
        """ compute Z_h, the Z_h.shape = (n,m) """
        Z = np.zeros(shape=(self.M, Y_pred.shape[0], Y_pred.shape[1]) ,dtype=float)
        for h in h_range:
            D_delta = self._f_delta_derive(Y_pred, h)
            Z[h] = D_delta  #* Y_pred *(1-Y_pred)

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

    def _compute_loss(self):
        Y_pred = self._get_prediction_trainset()
        # yp_max = np.amax(Y_pred)
        # yp_min = np.amin(Y_pred)
        # h_min = int(self.M-1-int(yp_max/self._d+1)) # yp_max=0.61 ==> h_min=7, self._b[h_min]=0.65 
        # h_min = max(h_min,0)
        # h_max = int(self.M-1-int(yp_min/self._d-1)) # yp_min=0.36 ==> h_max-1=13, self._b[h_max-1]=0.35
        # h_max = min(self.M, h_max)
        # h_range = range(h_min,h_max)
        h_range = range(self.M)
        
        """compute ??_h and bar_??_h """
        psi = np.zeros(self.M, dtype=float)
        psi_ = np.zeros(self.M, dtype=float)
        for h in h_range: 
            X = self._f_delta(Y_pred, h) # ??(Y^,h) 
            psi[h] = np.sum(X*self._Y) # ??[??(Y^,h)???Y]
            psi_[h] = np.sum(X) # ??[??(Y^,h)]
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
    
    def _get_prediction_trainset(self, is_reset_b_d=True):
        Y_pred = self._U@self._V.T
        if is_reset_b_d:
            yp_max = np.amax(Y_pred)
            yp_min = np.amin(Y_pred)
            yp_len = yp_max-yp_min
            self._d = yp_len/(self.M-1) # Delta: the with of intervals
            self._b = yp_max-np.arange(self.M)*self._d # center of intervals
        return Y_pred
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
                aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                value = aupr
            elif self.metric == 1:
                aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                auc = self._compute_auc(self._Y.flatten(), Y_pred.flatten())
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
#-------------------------------------------------------------------------------------------------------------------


        
class NRMFAP3(NRMFAP):
    """ 
    Difference with NRMFAP: Choose best T and K2 for computing latent feature of test drug and/or target
    better than NRMFAP. but similar with NRMFAP2
    For data1 (new data): better than NRMFAP2 on gpcr in S2 and S3, similar with (or slightly worse than) NRMFAP2 for other datasets 
    """
    def __init__(self, K=5, K2s=[5,7,9], metric=0, Ts=np.arange(0,1.1,0.1), num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=True):
        self.K = K # used for sparsying similarity matrix and computing latent features of new drugs/targets
        self.K2s = K2s # K2 used for compute test drug/target latent feature
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
        """ compute AUC Loss could lead to out of memory dataset e"""
        self.isAGD = isAGD
        
        self.copyable_attrs = ['K','K2s','metric','Ts','num_factors','theta','lambda_d','lambda_t','lambda_r','M','max_iter','seed','is_comLoss','isAGD']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self._Y = intMat

        self._construct_neighborhood(drugMat, targetMat) # 
        self._n1 = np.sum(self._Y) # the number of "1"s in Y 
        self._d = 1.0/(self.M-1) # Delta: the with of intervals
        self._b = 1-np.arange(self.M)*self._d # center of intervals
        self._AGD_optimization() 
        
        self._get_optimal_T(drugMat, targetMat)
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
                    aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                    value = aupr
                elif self.metric == 1:
                    aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                    auc = self._compute_auc(self._Y.flatten(), Y_pred.flatten())
                    value = aupr + auc
                if value > best_value:
                    best_value = value
                    self._best_T = T
                    self._best_K2 = K2
        # print(self._best_K2,' ',round(self._best_T,1))
    #----------------------------------------------------------------------------------------  

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
        scores = 1/(1+np.exp(-scores))
        
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
import os
import time
import sys
import numpy as np

from base.crossvalidation import *
from base.loaddata import * 


from mf.nrlmf import NRLMF
from mf.nrmfap2 import NRMFAP2
from mf.nrmfauc_2 import NRMFAUC_F3
from mf.mf2a import MF2A


from model_trans.mf.nrlmf_trans import NRLMF_TRANS
from model_trans.mf.nrmfap2_trans import NRMFAP2_TRANS
from model_trans.mf.nrmfauc_f3_trans import NRMFAUC_F3_TRANS
from model_trans.mf.mf2a_trans import MF2A_TRANS


from mv.com_sim.combine_sims import *
from mv.com_sim.hsic import HSIC

from mv.mv_model.lcs_sv import LCS_SV
from mv.mv_model_trans.lcs_sv_trans import LCS_SV_TRANS


"""
Inllustartions:
    
Name in Code                      Name in paper

NRMFAP2_GD/NRMFAP2_GD_TRANS       MFAUPR
NRMFAUC_F3/NRMFAUC_F3_TRANS       MFAUC
MF2A/MF2A_TRANS                   MF2A
Combine_Sims_Limb3                LIC 


The method with "_TRANS" suffix is used for prediction setting 1 (S1)
The method without the suffix is used for prediction settings 2,3,4 (S4)
""" 


def initialize_model_mv(method, cvs=2):
    if method == 'LCS_SV':
        model = LCS_SV(cs_model=Combine_Sims_Ave(), sv_model=NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0))
    elif method == 'LCS_SV_TRANS':
        model = LCS_SV_TRANS(cs_model=Combine_Sims_Ave(), sv_model=NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0))
    
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_mv function!!".format(method))
    return model
#----------------------------------------------------------------------------------------
        
def initialize_model_cs(method, cvs =2):
    if method == 'Combine_Sims_Ave':
        model = Combine_Sims_Ave()
    elif method == 'Combine_Sims_KA':
        model = Combine_Sims_KA()
    elif method == 'Combine_Sims_Limb3':  
        model = Combine_Sims_Limb3(k = 5)
    elif method == 'HSIC':
        model = HSIC(v1=2**-1, v2=2**-4, seed=0)
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_cs function!!".format(method))
    return model    
#----------------------------------------------------------------------------------------

def initialize_model_sv(method, cvs=2):
    model = None
    if method == 'NRLMF':
        model = NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)
    elif method == 'NRLMF_TRANS':
        model = NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)
    
    elif method == 'NRMFAUC_F3':
        a = 0.25
        model = NRMFAUC_F3(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=2, is_comLoss=0)
    elif method == 'NRMFAP2_GD':
        a = 0.25
        model = NRMFAP2(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
    elif method == 'NRMFAP2_GD_TRANS':
        a = 0.25
        model = NRMFAP2_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
    elif method == 'NRMFAUC_F3_TRANS':
        a = 0.25
        model = NRMFAUC_F3_TRANS(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=2, is_comLoss=0)

    elif method == 'MF2A':
        a = 0.25
        mfap = NRMFAP2(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
        mfauc = NRMFAUC_F3(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=2, is_comLoss=0)
        model = MF2A(mfap, mfauc, w=0.5)
    elif method == 'MF2A_TRANS':
        a = 0.25
        mfap = NRMFAP2_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, M=21, max_iter=100, seed=0, is_comLoss=1, isAGD=False)
        mfauc = NRMFAUC_F3_TRANS(K1=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, sfun=2, is_comLoss=0)
        model = MF2A_TRANS(mfap, mfauc, w=0.5)
    
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_sv function!!".format(method))
    return model
#----------------------------------------------------------------------------------------
    



if __name__ == "__main__":
    # !!! my_path should be change to the path of the project in your machine   
    my_path = 'F:\envs\GitHub_MF2A_MDMF2A'
    n_jobs = 20 # set the n_jobs = 20 if possible
    

    data_dir =  os.path.join(my_path, 'datasets_mv') #'F:\envs\DPI_Py37\Codes_Py3\datasets' #'data'
    output_dir = os.path.join(my_path, 'output','') #'F:\envs\DPI_Py37\Codes_Py3\output' 
    seeds = [0,1] 
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 
    out_summary_file = os.path.join(output_dir, "summary_result"+ time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()) +".txt")

    
    """ run cross validation on a model with best parameters setting from file"""
    cmd ="Method\tsv_method\tcs_method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime\tparam"
    print(cmd)
    with open(out_summary_file,"w") as f:
        f.write(cmd+"\n")
    for sv_method_ori in ['MF2A']: #  options: 'NRMFAP2_GD_TRANS','NRMFAUC_F3_TRANS' 'NRMFAP2_GD','NRMFAUC_F3'
        for cs_method in ['Combine_Sims_Limb3']:  # options: 'Combine_Sims_Ave','Combine_Sims_KA','HSIC', 'Combine_Sims_Limb3'
            for method_ori in ['LCS_SV']:  
                for cvs in [1,2,3,4]: # options: 1,2,3,4
                    num = 10
                    if cvs == 4:
                        num = 3
                    if cvs == 1 and '_TRANS' not in sv_method_ori:
                        method = method_ori+'_TRANS'
                        sv_method = sv_method_ori+'_TRANS'
                    else:
                        method = method_ori
                        sv_method = sv_method_ori
                    
                    if '_TRANS' in sv_method:
                        isInductive = False
                    else:
                        isInductive = True  


                    # load parameter settings
                    full_method =  method+'_'+sv_method+'_'+cs_method # for output and getting parameters
                    vp_best_param_file = os.path.join(data_dir, 'method_params_VP data1',full_method+'_best_param.txt')
                    dict_params = get_params2(vp_best_param_file, num_key=3) # read parameters from file
                    
                    if 'MF2A' in sv_method:
                        full_method_ap = full_method.replace('MF2A', 'NRMFAP2_GD')
                        vp_best_param_file_ap = os.path.join(data_dir, 'method_params_VP data1',full_method_ap+'_best_param.txt')
                        dict_params_ap = get_params2(vp_best_param_file_ap, num_key=3) # read parameters from file
                    
                        full_method_auc = full_method.replace('MF2A', 'NRMFAUC_F3')
                        vp_best_param_file_auc = os.path.join(data_dir, 'method_params_VP data1',full_method_auc+'_best_param.txt')
                        dict_params_auc = get_params2(vp_best_param_file_auc, num_key=3) # read parameters from file
                
          
                    model = initialize_model_mv(method,cvs)  # parammeters could be changed in "initialize_model" function
                    model.sv_model = initialize_model_sv(sv_method, cvs)
                    model.cs_model = initialize_model_cs(cs_method, cvs)
                    num = 10
                    if cvs == 4:
                        num = 3
                    for dataset in ['nr1']:  # options: 'nr1','gpcr1','ic1','e1', 'luo' 
                                             # datasets with suffix "1" are the updated golden standard datasets
                        if dataset == 'luo':  seeds = [0]
                        else:  seeds = [0,1]     
                        out_file_name= os.path.join(output_dir, "Best_parameters_"+full_method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                        intMat, drugMats, targetMats, Dsim_names, Tsim_names = load_datasets(dataset, data_dir ,'low4Limb') # ,'Original','low4Limb'
                        
                        if 'MF2A' in sv_method:
                            param_ap = dict_params_ap[(full_method_ap, str(cvs), dataset)]
                            model.sv_model.mfap.set_params(**param_ap)
                            
                            param_auc = dict_params_auc[(full_method_auc, str(cvs), dataset)]
                            model.sv_model.mfauc.set_params(**param_auc)
                        
                        param = dict_params[(full_method, str(cvs), dataset)]
                        model.sv_model.set_params(**param)
                        
                        tic = time.time()
                        # auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, isInductive)
                        auprs, aucs, run_times = cross_validate_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, n_jobs, isInductive)
                        cmd = "{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(method,sv_method,cs_method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic)
                        print(cmd)
                        with open(out_summary_file,"a") as f:
                            f.write(cmd+"\n")   

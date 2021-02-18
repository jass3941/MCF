# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:02:25 2021

@author: daham.kim

모듈화 끝나는대
graph.py 파일로 소스코드 이전할 계획
command 부분은 main.py 파일로 이전할 계획
"""
#temp for quick test
import pandas as pd
import numpy as np
import graph as gp
import data_loading as load
from test_main_copy import df_test_result

"""
graph 모둘에 있던 코드 --> 혼합 필요
"""
# later change index from inter to table name


# import psutil
# psutil.cpu_count(logical=False)


# from multiprocessing import Pool
# 

def multi_process(rows) :
    # to_append = pd.DataFrame(columns = df_test_result.columns)
    for j in range(df_test_result.shape[1]):
        aaa = gp.simrank_similarity (SG, source =  df_test_result.index[rows] , target = df_test_result.index[j] )        
        print("calculating similarity of (" , rows , " , ", j , ") ", aaa)
        # temp.append(aaa)
        df_test_result.iloc[rows, j] = aaa
    # to_append.loc[rows] = temp
    # to_append.append(temp)
    # df_test_reuslt.loc[rows] = temp
    return df_test_reuslt


def similarity_row(df_input):
    import graph as gp
    
    def pre_process(df):
        import pandas as pd
        import graph as gp
        import data_loading as load
        
        items = ['CF.Prem_Inc' ,	'CF.Net_IMF_Inc' ,	'CF.GMxB_Net_Inc' ,	'CF.Invest_Inc' ,	'CF.EV_Claim' ,	'CF.Claim_Tot' ,	'CF.Res_At_Dth' ,	'CF.Liv_Ben' ,	'CF.Claim_Anty' ,	'CF.Mat_Ben' ,	'CF.Surr_Ben' ,	'CF.PartWith' ,	'CF.Commission' ,	'CF.Acq_Exp' ,	'CF.Acq_ND_Exp' ,	'CF.Acq_Tot' ,	'CF.Mnt_P_Exp' ,	'CF.Mnt_NP_Exp' ,	'CF.Mnt_Tot' ,	'CF.Coll_Exp' ,	'CF.Oth_Exp' ,	'CF.Trans_To_SA' ,	'CF.Res_Inc' ,	'CF.DAC_Inc' ,	'CF.Loan_Balance' ,	'CF.Loan_Inc' ,	'CF.Loan_Interest' ,	'CF.Reserve' ,	'CF.SA' ,	'CF.Surr_Reserve' ,	'CF.UnEarn_Reserve' ,	'CF.GMDB_Res' ,	'CF.GMAB_Res' ,	'DAC.DAC' ,	'CF.Rep_Coll_Alpha_IF' ,	'CF.sop_Coll_Alpha_IF' ,	'CF.sop_Coll_Beta_IF' ,	'CF.sop_Coll_Gamma_IF' ,	'CF.sop_Sav_Prem_IF' ,	'CF.sop_Risk_Prem_IF' ,	'CF.Res_At_Surr' ,	'CF.IMF_Income' ,	'CF.IMF_Outgo' ,	'CF_RBC.RBC_Cred' ,	'CF_RBC.RBC_Market' ,	'CF_RBC.RBC_Ins' ,	'CF_RBC.RBC_ALM' ,	'CF_RBC.RBC_Oper' ,	'CF_RBC.RBC_Risk_Prem_1Yr' ,	'CF_RBC.Ins_Death' ,	'CF_RBC.Ins_Dis' ,	'CF_RBC.Ins_Hosp' ,	'CF_RBC.Ins_SurgDiag' ,	'CF_RBC.Ins_MedExp' ,	'CF_RBC.Ins_ETC' ,	'CF_Run.Excess_Capital' ,	'CF_Run.Base_RBC' ,	'CF_Run.Solv_Capital' ,	'CF_Run.Rep_Capital' ,	'CF_Run.Capital_Int' ,	'CF_Run.Capital_Inc' ,	'CF_Run.Capital_Outgo' ,	'NIER_Mon' ,	'APE_Non_Single' ,	'APE_Single' ,	'No.Pols_B' ,	'No.Pols_E' ,	'No.Pays' ,	'No.Dths' ,	'No.Surrs' ,	'QxBasic' ,	'QxBasic_Ann' ,	'No.Qmo_Pol' ,	'No.Qmo_Prem' ,	'No.Wx' ,	'No.Wx_Skew' ,	'No.Wx_Add' ,	'No.Surr_Rate' ,	'Prem_Paid' ,	'Add_Prem' ,	'Credit_Prem' ,	'Deduct.AlphaLoad' ,	'CF.NBZ_Alpha' ,	'CF.MN_Alpha' ,	'Deduct.BetaLoad_PP' ,	'Deduct.BetaLoad_PU' ,	'Deduct.VarFA_BetaLoad' ,	'Deduct.GammaLoad' ,	'Deduct.Risk_Prem' ,	'Deduct.Waiv_Prem' ,	'Deduct.GMB_Tot' ,	'AV.e' ,	'AV.Base_e' ,	'AV.Addpr_e' ,	'AV.Bonus_e' ,	'Fund.e' ,	'Fund.Base_e' ,	'Fund.Addpr_e' ,	'Fund.Bonus_e' ,	'Shad_AV.e' ,	'Shad_AV.Base_e' ,	'Shad_AV.Addpr_e' ,	'Shad_AV.Bonus_e' ,	'TV.Mon' ,	'TV.Stat_Mon' ,	'TV.W_Mon' ,	'TV.LW_Mon' ,	'CSV' ,	'CF.Csv_Base' ,	'CF.Alp' ,	'CF.Amort_Alpha' ,	'CF.Std_Amort_Alpha' ,	'CF.ROP' ,	'CF.S' ,	'Cred_Rate' ,	'Cred_R' ,	'RBC_RP(1 -)' ,	'RBC_RP(2 -)' ,	'RBC_RP(3 -)' ,	'RBC_RP(4 -)' ,	'RBC_RP(5 -)' ,	'RBC_RP(6 -)' ,	'RBC_RP_Tot' ,	'Inp_Bonus' ,	'Fee_Bonus' ,	'Bonus_V' ,	'CF_RBC.Liab_Amt' ,	'CF_RBC.Asset_Amt' ,	'CF_RBC.ALM_Min_Amt' ,	'CF.Addpr_Inc' ,	'CF.Claim_GMDB' ,	'CF.GMDB_Inc' ,	'CF.GMDB_Net_Inc' ,	'CF.Claim_GMAB' ,	'CF.GMAB_Inc' ,	'CF.GMAB_Net_Inc' ,	'CF.Waiver_Cost' ,	'Expense.Comm_Schedule' ,	'Expense.Comm_Acq_ConvPrem' ,	'Expense.Comm_Mnt_ConvPrem' ,	'Expense.Comm_ExpAcq' ,	'Expense.Comm_ExpMaint' ,	'Expense.Comm_Pol' ,	'Expense.Comm_PrevComm' ,	'Expense.Comm_Nmm_Prmt' ,	'Expense.Comm_Last_1Yr' ,	'Expense.ACQ_A_ExpAcq' ,	'Expense.ACQ_A_ConvPrem' ,	'Expense.ACQ_A_NewPol' ,	'Expense.ACQ_2_Pol' ,	'Expense.ACQ_M_ExpAcq' ,	'Expense.ACQ_M_ConvPrem' ,	'Expense.MNT_ExpMaint' ,	'Expense.MNT_Prem' ,	'Expense.MNT_ExistPol' ,	'Expense.MNT_2_Pol' ,	'Expense.OTH_Prem' ,	'Expense.OTH_GAPrem' ,	'Expense.OTH_SurrV' ,	'Expense.OTH_GASurrV' ,	'Expense.OTH_GMBFee' ,	'Expense.OTH_VarGuarV' ,	'Expense.OTH_Prem_2' ,	'Expense.OTH_SurrV_2' ,	'CF.Pre_Tax_Profit']
        scenario = load.scenario_loading('Output_Scenario_20210105.xlsx')        
        Graph = gp.graph_building(df, item = items , scen = scenario) 
        df_result = pd.DataFrame(columns = df.index, index = df.index)        
        return Graph, df_result
    G, df_test_result = pre_process(df_input)

    for i in range(df_test_result.shape[0]):
        df_test_result.iloc[i].apply(list(map(lambda j_index : gp.simrank_similarity(G, source = df_test_result.index[i] , target = df_test_result.index[j_index]) , range(df_test_result.shape[1]) )))
    return df_test_result


def parallelize_dataframe(df , func, n_cores):
    df_split =  np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df =  pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df





# import os
# os.cpu_count()

# import psutil
# psutil.cpu_count()





from multiprocessing import Pool

def fff(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(fff, [1, 2, 3]))


# from multiprocessing import Process

# def f(name):
#     print('hello', name)

# if __name__ == '__main__':
#     # p = Process(target=fff, args=('bob',))
#     p.start()
#     p.join()
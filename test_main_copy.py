# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:30:30 2021

@author: daham.kim
"""

import pandas as pd
import numpy as np
import time


def dircetory_change(dir):
    import os
    print("Current working Directory is : ")
    print(os.getcwd())
    os.chdir(dir)
    print("Working Directory changed to : ")
    print(os.getcwd())
    
dircetory_change('/Users/daham.kim/Desktop/cf_code')


import data_loading as load
import similarity_measure as sim
import graph as gp

db_name = "CF_Output_20210111.db"

load.dircetory_change('/Users/daham.kim/Desktop/cf_code')
scenario = load.scenario_loading('Output_Scenario_20210105.xlsx')
items = ['CF.Prem_Inc' ,	'CF.Net_IMF_Inc' ,	'CF.GMxB_Net_Inc' ,	'CF.Invest_Inc' ,	'CF.EV_Claim' ,	'CF.Claim_Tot' ,	'CF.Res_At_Dth' ,	'CF.Liv_Ben' ,	'CF.Claim_Anty' ,	'CF.Mat_Ben' ,	'CF.Surr_Ben' ,	'CF.PartWith' ,	'CF.Commission' ,	'CF.Acq_Exp' ,	'CF.Acq_ND_Exp' ,	'CF.Acq_Tot' ,	'CF.Mnt_P_Exp' ,	'CF.Mnt_NP_Exp' ,	'CF.Mnt_Tot' ,	'CF.Coll_Exp' ,	'CF.Oth_Exp' ,	'CF.Trans_To_SA' ,	'CF.Res_Inc' ,	'CF.DAC_Inc' ,	'CF.Loan_Balance' ,	'CF.Loan_Inc' ,	'CF.Loan_Interest' ,	'CF.Reserve' ,	'CF.SA' ,	'CF.Surr_Reserve' ,	'CF.UnEarn_Reserve' ,	'CF.GMDB_Res' ,	'CF.GMAB_Res' ,	'DAC.DAC' ,	'CF.Rep_Coll_Alpha_IF' ,	'CF.sop_Coll_Alpha_IF' ,	'CF.sop_Coll_Beta_IF' ,	'CF.sop_Coll_Gamma_IF' ,	'CF.sop_Sav_Prem_IF' ,	'CF.sop_Risk_Prem_IF' ,	'CF.Res_At_Surr' ,	'CF.IMF_Income' ,	'CF.IMF_Outgo' ,	'CF_RBC.RBC_Cred' ,	'CF_RBC.RBC_Market' ,	'CF_RBC.RBC_Ins' ,	'CF_RBC.RBC_ALM' ,	'CF_RBC.RBC_Oper' ,	'CF_RBC.RBC_Risk_Prem_1Yr' ,	'CF_RBC.Ins_Death' ,	'CF_RBC.Ins_Dis' ,	'CF_RBC.Ins_Hosp' ,	'CF_RBC.Ins_SurgDiag' ,	'CF_RBC.Ins_MedExp' ,	'CF_RBC.Ins_ETC' ,	'CF_Run.Excess_Capital' ,	'CF_Run.Base_RBC' ,	'CF_Run.Solv_Capital' ,	'CF_Run.Rep_Capital' ,	'CF_Run.Capital_Int' ,	'CF_Run.Capital_Inc' ,	'CF_Run.Capital_Outgo' ,	'NIER_Mon' ,	'APE_Non_Single' ,	'APE_Single' ,	'No.Pols_B' ,	'No.Pols_E' ,	'No.Pays' ,	'No.Dths' ,	'No.Surrs' ,	'QxBasic' ,	'QxBasic_Ann' ,	'No.Qmo_Pol' ,	'No.Qmo_Prem' ,	'No.Wx' ,	'No.Wx_Skew' ,	'No.Wx_Add' ,	'No.Surr_Rate' ,	'Prem_Paid' ,	'Add_Prem' ,	'Credit_Prem' ,	'Deduct.AlphaLoad' ,	'CF.NBZ_Alpha' ,	'CF.MN_Alpha' ,	'Deduct.BetaLoad_PP' ,	'Deduct.BetaLoad_PU' ,	'Deduct.VarFA_BetaLoad' ,	'Deduct.GammaLoad' ,	'Deduct.Risk_Prem' ,	'Deduct.Waiv_Prem' ,	'Deduct.GMB_Tot' ,	'AV.e' ,	'AV.Base_e' ,	'AV.Addpr_e' ,	'AV.Bonus_e' ,	'Fund.e' ,	'Fund.Base_e' ,	'Fund.Addpr_e' ,	'Fund.Bonus_e' ,	'Shad_AV.e' ,	'Shad_AV.Base_e' ,	'Shad_AV.Addpr_e' ,	'Shad_AV.Bonus_e' ,	'TV.Mon' ,	'TV.Stat_Mon' ,	'TV.W_Mon' ,	'TV.LW_Mon' ,	'CSV' ,	'CF.Csv_Base' ,	'CF.Alp' ,	'CF.Amort_Alpha' ,	'CF.Std_Amort_Alpha' ,	'CF.ROP' ,	'CF.S' ,	'Cred_Rate' ,	'Cred_R' ,	'RBC_RP(1 -)' ,	'RBC_RP(2 -)' ,	'RBC_RP(3 -)' ,	'RBC_RP(4 -)' ,	'RBC_RP(5 -)' ,	'RBC_RP(6 -)' ,	'RBC_RP_Tot' ,	'Inp_Bonus' ,	'Fee_Bonus' ,	'Bonus_V' ,	'CF_RBC.Liab_Amt' ,	'CF_RBC.Asset_Amt' ,	'CF_RBC.ALM_Min_Amt' ,	'CF.Addpr_Inc' ,	'CF.Claim_GMDB' ,	'CF.GMDB_Inc' ,	'CF.GMDB_Net_Inc' ,	'CF.Claim_GMAB' ,	'CF.GMAB_Inc' ,	'CF.GMAB_Net_Inc' ,	'CF.Waiver_Cost' ,	'Expense.Comm_Schedule' ,	'Expense.Comm_Acq_ConvPrem' ,	'Expense.Comm_Mnt_ConvPrem' ,	'Expense.Comm_ExpAcq' ,	'Expense.Comm_ExpMaint' ,	'Expense.Comm_Pol' ,	'Expense.Comm_PrevComm' ,	'Expense.Comm_Nmm_Prmt' ,	'Expense.Comm_Last_1Yr' ,	'Expense.ACQ_A_ExpAcq' ,	'Expense.ACQ_A_ConvPrem' ,	'Expense.ACQ_A_NewPol' ,	'Expense.ACQ_2_Pol' ,	'Expense.ACQ_M_ExpAcq' ,	'Expense.ACQ_M_ConvPrem' ,	'Expense.MNT_ExpMaint' ,	'Expense.MNT_Prem' ,	'Expense.MNT_ExistPol' ,	'Expense.MNT_2_Pol' ,	'Expense.OTH_Prem' ,	'Expense.OTH_GAPrem' ,	'Expense.OTH_SurrV' ,	'Expense.OTH_GASurrV' ,	'Expense.OTH_GMBFee' ,	'Expense.OTH_VarGuarV' ,	'Expense.OTH_Prem_2' ,	'Expense.OTH_SurrV_2' ,	'CF.Pre_Tax_Profit']

"""
cf = load.sqlite_to_pandas(db_name, "Output_ISP_Prot_T1_CF_Scn_Base")
cf1 = load.sqlite_to_pandas(db_name, "Output_ISP_Prot_T1_CF_Scn_Base")

df_norm = sim.min_max_sclae(cf)
df_norm1 = sim.min_man_scale(cf1)
#df_diff = sim.diff_normalization(cf)
"""

"""
removing nan values --> uncomment if necessary
cleanedList = [x for x in test if ~np.isnan(x)]
"""

def similarity_btw(db_name, cf1_table, cf2_table):
    cf1 = load.sqlite_to_pandas(db_name, cf1_table)
    cf2 = load.sqlite_to_pandas(db_name, cf2_table)
    similarity1 = sim.cos_similarity(cf1,cf2)
#    similarity1 = [x for x in similarity1 if ~np.isnan(x)]
    return similarity1
#test = similarity_btw(db_name, 'Output_ISP_Prot_T1_CF_Scn_Base','Output_ISP_Prot_T1_CF_Scn_31_ExpAcq1')


def similarity_normal_btw(db_name, cf1_table, cf2_table):
    cf1 = load.sqlite_to_pandas(db_name, cf1_table)
    cf2 = load.sqlite_to_pandas(db_name, cf2_table)
    df_norm = sim.min_max_scale(cf1)
    df_norm1 = sim.min_man_scale(cf2)
    similarity1 = sim.cos_similarity(df_norm,df_norm1)
#    similarity1 = [x for x in similarity1 if ~np.isnan(x)]
    return similarity1
#test_normal = similarity_btw(db_name, 'Output_ISP_Prot_T1_CF_Scn_Base','Output_ISP_Prot_T1_CF_Scn_31_ExpAcq1')


def similarity_size(db_name, cf1_table, cf2_table):
    cf1 = load.sqlite_to_pandas(db_name, cf1_table)
    cf2 = load.sqlite_to_pandas(db_name, cf2_table)
    similarity1 = sim.size_similarity(cf1,cf2)
#    similarity1 = [x for x in similarity1 if ~np.isnan(x)]
    return similarity1
#test_size = similarity_size(db_name, 'Output_ISP_Prot_T1_CF_Scn_Base','Output_ISP_Prot_T1_CF_Scn_31_ExpAcq1')



def item_analysis(base_scenario_index):
    base_matrix = pd.DataFrame(columns = items, index = scenario.iloc[:,6])
    for i in range(scenario.shape[0]):
        test = similarity_btw(db_name, scenario.iloc[base_scenario_index,6] , scenario.iloc[i,6])
        for j in range(len(items)):
            base_matrix.iloc[i,j] = test[j]
#        print(base_matrix.iloc[i,:])
    print(base_scenario_index)
    print("============================================================")
    return base_matrix


"""
df0 =item_analysis(0)

#cf item이 원본- 검증본 두개다 출력이 안되는 경우는 올바르게 산출된 경우이므로 유사도 100% 점수 부여
df0.fillna(1)
df0.to_excel("base_step.xlsx")
#df1.to_excel("사업비_ACQ_0.xlsx")
 """




#temp for quick test
df0 = pd.read_excel("base_step_copy.xlsx")
df0 = pd.DataFrame(df0.values,index = df0.iloc[:,0], columns = df0.columns)
df0.drop(columns = '테이블', inplace = True)
df0.fillna(1)
df_test = df0.copy()
df0 = df_test

"""
graph 모둘에 있던 코드 --> 혼합 필요
"""

G = gp.graph_building(df0, item = items , scen = scenario )
### only for running test purpose   below loop statement takes long time to compute
#simrank_similarity (G, source =  df_test.index[0] ,  target = df_test.index[1] )
#simrank_similarity (G, source =  df_test.index[1] ,  target = df_test.index[2] )
#simrank_similarity (G, source =  df_test.index[2] ,  target = df_test.index[3] )



    
def multi_process(rows) :
    result = np.empty(shape = (df0.shape[0] , df0.shape[1]) )
    to_append = []
    
    for j in range(df0.shape[1]):
        if rows > j:
            aaa = 0
            print("similarity of (" , rows , " , ", j , ") is ", aaa)
        else :    
            aaa = gp.simrank_similarity (G, source =  df0.index[rows] , target = df0.index[j] )        
            print("similarity of (" , rows , " , ", j , ") is ", aaa)
        to_append.append(aaa)
    
    result[rows] = to_append
    return result


"""
    for i in range(df_test_result.shape[0]):
        df_test_result.loc[i].apply(list(map(lambda j_index : gp.simrank_similarity(G, source = df_test_result.index[i] , target = df_test_result.index[j_index]) , range(df_test_result.shape[1]) )))
"""

import os
core = os.cpu_count()
from multiprocessing import Pool

"""
start = time.time()

result_dict = np.empty(shape = (df0.shape[0] , df0.shape[1]) )
num_list = np.arange(scenario.shape[0]).tolist()

if __name__ == '__main__':
    with Pool(core - 4) as p:
        result_dict = p.map(multi_process , num_list)

end =  time.time() - start

result = pd.DataFrame(result_dict)
"""


num_list = np.arange(scenario.shape[0]).tolist()
# df_test_result = pd.DataFrame(index = df0.index, columns = df0.columns)
df_test_result = pd.DataFrame(index = num_list, columns = num_list)


def things_to_do(df):
    for rows in num_list:
        # df[df0.index[rows]] = df0.apply(list(map(lambda j_index : gp.simrank_similarity(G, source = df.index[rows] , target = df.index[j_index]) , range(df0.shape[1]) )))
        # df.loc[rows] = df0.apply(list(map(lambda j_index : gp.simrank_similarity(G, source = df.index[rows] , target = df.index[j_index]) , range(df0.shape[1]) )))
        df_test_result.loc[rows]  = list(map(lambda j_index : gp.simrank_similarity(G, source = df0.index[rows] , target = df0.index[j_index]) , num_list ))
        # df_test_result.loc[rows] = list(map(lambda j_index : j_index+1 , num_list ))
    return df_test_result

# df_test_result.loc[1] = df0.apply(list(map(lambda j_index : gp.simrank_similarity(G, source = df0.index[1] , target = df0.index[j_index]) , range(df0.shape[1]) )))

    



"""
def parallelize_dataframe(df, func, n_cores= core - 4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df        
"""

if __name__ =='__main__':
    
    def parallelize_dataframe(df, func, n_cores= core - 4):
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df  
    
    df_final = parallelize_dataframe(df_test_result ,  things_to_do)





""" 
# for non parallel usage    
for i in range(df0.shape[0]):
    for j in range(df0.shape[1]):
        print("calculating similarity of (" , i , " , ", j , ") " )
        temp = gp.simrank_similarity (G, source =  df_test_result.index[i] ,  target = df_test_result.index[j] )
        df_test_result.iloc[i,j] = temp
        print(temp)

df_test_result.to_excel("sililarity_score.xlsx")

print("total time : ", time.time())
print("similarity matrix time : ", mid)
print("similarity graph time  : ", end)

"""
# -*- coding: utf-8 -*-
import sys
import os

import pandas as pd
import numpy as np


def filepath_change(filepath):
    print("Current working Directory is : ")
    print(os.getcwd())
    os.chdir(filepath)
    print("Working Directory changed to : ")
    print(os.getcwd())


def scenario_loading(title):
    scenario = pd.read_excel(title, sheet_name='Scenario')
    return scenario


"""
# when using excel 
def data_loading(title):
    df = pd.read_csv(title)
    to_delete = [ 'PRDCD',	 'PLYNO',	 'OPEXP_SHAR_CHN_SECD',	 'Proj_YM']
    df.drop(columns = to_delete, inplace = True)
    return df
cf = data_loading("CF_Test_1.csv")
"""


def sqlite_tables(db_name):
    #   db_name = sqlite3 db name
    import sqlite3
    conn = sqlite3.connect('%s' % db_name)
    c = conn.cursor()
    # when using multiple table containing db
    tables_in_db = []
    i = 0
    for row in c.execute("SELECT name FROM sqlite_master"):
        #            print(row[0])
        tables_in_db.append((row[0]))
    return tables_in_db


def sqlite_to_pandas(db_name, table_name):
    #   db_name = sqlite3 db name
    #    table_name = sqlite3 table name
    headers = ['PRDCD', 'PLYNO', 'OPEXP_SHAR_CHN_SECD', 'Gender', 'Age', 'YY_STD_PYPD', 'PYCYC_COD', 'Proj_YM', 'th',
               'CF.Prem_Inc', 'CF.Net_IMF_Inc', 'CF.GMxB_Net_Inc', 'CF.Invest_Inc', 'CF.EV_Claim', 'CF.Claim_Tot',
               'CF.Res_At_Dth', 'CF.Liv_Ben', 'CF.Claim_Anty', 'CF.Mat_Ben', 'CF.Surr_Ben', 'CF.PartWith',
               'CF.Commission', 'CF.Acq_Exp', 'CF.Acq_ND_Exp', 'CF.Acq_Tot', 'CF.Mnt_P_Exp', 'CF.Mnt_NP_Exp',
               'CF.Mnt_Tot', 'CF.Coll_Exp', 'CF.Oth_Exp', 'CF.Trans_To_SA', 'CF.Res_Inc', 'CF.DAC_Inc',
               'CF.Loan_Balance', 'CF.Loan_Inc', 'CF.Loan_Interest', 'CF.Reserve', 'CF.SA', 'CF.Surr_Reserve',
               'CF.UnEarn_Reserve', 'CF.GMDB_Res', 'CF.GMAB_Res', 'DAC.DAC', 'CF.Rep_Coll_Alpha_IF',
               'CF.sop_Coll_Alpha_IF', 'CF.sop_Coll_Beta_IF', 'CF.sop_Coll_Gamma_IF', 'CF.sop_Sav_Prem_IF',
               'CF.sop_Risk_Prem_IF', 'CF.Res_At_Surr', 'CF.IMF_Income', 'CF.IMF_Outgo', 'CF_RBC.RBC_Cred',
               'CF_RBC.RBC_Market', 'CF_RBC.RBC_Ins', 'CF_RBC.RBC_ALM', 'CF_RBC.RBC_Oper', 'CF_RBC.RBC_Risk_Prem_1Yr',
               'CF_RBC.Ins_Death', 'CF_RBC.Ins_Dis', 'CF_RBC.Ins_Hosp', 'CF_RBC.Ins_SurgDiag', 'CF_RBC.Ins_MedExp',
               'CF_RBC.Ins_ETC', 'CF_Run.Excess_Capital', 'CF_Run.Base_RBC', 'CF_Run.Solv_Capital',
               'CF_Run.Rep_Capital', 'CF_Run.Capital_Int', 'CF_Run.Capital_Inc', 'CF_Run.Capital_Outgo', 'NIER_Mon',
               'APE_Non_Single', 'APE_Single', 'No.Pols_B', 'No.Pols_E', 'No.Pays', 'No.Dths', 'No.Surrs', 'QxBasic',
               'QxBasic_Ann', 'No.Qmo_Pol', 'No.Qmo_Prem', 'No.Wx', 'No.Wx_Skew', 'No.Wx_Add', 'No.Surr_Rate',
               'Prem_Paid', 'Add_Prem', 'Credit_Prem', 'Deduct.AlphaLoad', 'CF.NBZ_Alpha', 'CF.MN_Alpha',
               'Deduct.BetaLoad_PP', 'Deduct.BetaLoad_PU', 'Deduct.VarFA_BetaLoad', 'Deduct.GammaLoad',
               'Deduct.Risk_Prem', 'Deduct.Waiv_Prem', 'Deduct.GMB_Tot', 'AV.e', 'AV.Base_e', 'AV.Addpr_e',
               'AV.Bonus_e', 'Fund.e', 'Fund.Base_e', 'Fund.Addpr_e', 'Fund.Bonus_e', 'Shad_AV.e', 'Shad_AV.Base_e',
               'Shad_AV.Addpr_e', 'Shad_AV.Bonus_e', 'TV.Mon', 'TV.Stat_Mon', 'TV.W_Mon', 'TV.LW_Mon', 'CSV',
               'CF.Csv_Base', 'CF.Alp', 'CF.Amort_Alpha', 'CF.Std_Amort_Alpha', 'CF.ROP', 'CF.S', 'Cred_Rate', 'Cred_R',
               'RBC_RP(1 -)', 'RBC_RP(2 -)', 'RBC_RP(3 -)', 'RBC_RP(4 -)', 'RBC_RP(5 -)', 'RBC_RP(6 -)', 'RBC_RP_Tot',
               'Inp_Bonus', 'Fee_Bonus', 'Bonus_V', 'CF_RBC.Liab_Amt', 'CF_RBC.Asset_Amt', 'CF_RBC.ALM_Min_Amt',
               'CF.Addpr_Inc', 'CF.Claim_GMDB', 'CF.GMDB_Inc', 'CF.GMDB_Net_Inc', 'CF.Claim_GMAB', 'CF.GMAB_Inc',
               'CF.GMAB_Net_Inc', 'CF.Waiver_Cost', 'Expense.Comm_Schedule', 'Expense.Comm_Acq_ConvPrem',
               'Expense.Comm_Mnt_ConvPrem', 'Expense.Comm_ExpAcq', 'Expense.Comm_ExpMaint', 'Expense.Comm_Pol',
               'Expense.Comm_PrevComm', 'Expense.Comm_Nmm_Prmt', 'Expense.Comm_Last_1Yr', 'Expense.ACQ_A_ExpAcq',
               'Expense.ACQ_A_ConvPrem', 'Expense.ACQ_A_NewPol', 'Expense.ACQ_2_Pol', 'Expense.ACQ_M_ExpAcq',
               'Expense.ACQ_M_ConvPrem', 'Expense.MNT_ExpMaint', 'Expense.MNT_Prem', 'Expense.MNT_ExistPol',
               'Expense.MNT_2_Pol', 'Expense.OTH_Prem', 'Expense.OTH_GAPrem', 'Expense.OTH_SurrV',
               'Expense.OTH_GASurrV', 'Expense.OTH_GMBFee', 'Expense.OTH_VarGuarV', 'Expense.OTH_Prem_2',
               'Expense.OTH_SurrV_2', 'CF.Pre_Tax_Profit']
    to_delete = ['th', 'PRDCD', 'PLYNO', 'OPEXP_SHAR_CHN_SECD', 'Proj_YM', 'Gender', 'Age', 'YY_STD_PYPD', 'PYCYC_COD']
    df_original = pd.DataFrame(columns=headers)

    tables_in_db = sqlite_tables(db_name)
    if table_name not in tables_in_db:
        print("Table is not in Given DB file, please check again")
        sys.exit(1)

    import sqlite3
    conn = sqlite3.connect('%s' % db_name)
    c = conn.cursor()

    i = 0
    for row in c.execute("SELECT * FROM %s" % table_name):
        row = np.array(row)
        df_original.loc[i] = row
        i = i + 1

    df_original.drop(columns=to_delete, inplace=True)
    for i in range(df_original.shape[0]):
        df_original.loc[i] = pd.to_numeric(df_original.loc[i])

    return df_original

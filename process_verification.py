#!/usr/bin/python
# process_verification.py
# Robert C Fritzen - Dpt. Geographic & Atmospheric Sciences
#
# Python script file which handles verification tasks
#
# This script file should not be called, but initialized from mpirun

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import sys
#import conda
import os
from mpi4py import MPI
import glob
import getopt
import itertools
import xesmf as xe
import scipy.stats as stats
from shapely.geometry import Point, Polygon, shape, mapping
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import geopandas as gpd
import pandas as pd
import geojsoncontour
import matplotlib.pyplot as plt
import json

import calculation
import settings
import tools
import mpipool

"""
CONSTANTS
NOTE: I have updated the script to now automatically compute FSS thresholds for precipitation based on percentiles of verification data, just make sure to adjust the START/END times
"""
FILE_VERSION = 2.1

# These three should be ok to be left constant, but are offered here for quick tuning.
FSS_THRESH_SIMREF = 40 # 40 dBZ
FSS_THRESH_HAILSIZE = 25 # 25mm ~ 1 in
FSS_THRESHS_THETAE = [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350] # Not used, retained for other projects
FSS_THRESHS_PRECIP_PERCENTILES = [1, 5, 10, 25, 50, 75, 80, 90, 95, 99, 99.9, 99.99] # Adjust these values to control how many FSS thresholds are computed for precipitation

FSS_Scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                   110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 
                   500, 600, 700, 800, 900, 1000]

"""
CHAPTER 1
"""
START_ANALYSIS_TIME = datetime.strptime("2012_04_13_12", "%Y_%m_%d_%H")
END_ANALYSIS_TIME = datetime.strptime("2012_04_15_12", "%Y_%m_%d_%H")
#FSS_THRESH_PRECIP = [0.0, 0.0, 0.0, 0.0, 1.6790009707212448, 9.032004565000534, 11.617006814479828, 21.64901165962219, 32.404017734527585, 59.146434583664075, 92.26360198116383, 115.52980533618019]

ENSEMBLE_SAVE_DIR = '/data1/climlab/fritzen-dissertation/Chapter1/Ensemble/'
ENSEMBLE = {
    'Control': '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/NoPhysics/',
    'SPPT':
        {1: 
            {0: '/data1/climlab/fritzen-dissertation/Chapter1/Physics/SPPT1/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT1/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT1/'},
         2:
            {0: '/data1/climlab/fritzen-dissertation/Chapter1/Physics/SPPT2/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT2/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT2/'},         
         3:
            {0: '/data1/climlab/fritzen-dissertation/Chapter1/Physics/SPPT3/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT3/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT3/'},         
         4:
            {0: '/data1/climlab/fritzen-dissertation/Chapter1/Physics/SPPT4/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT4/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT4/'},         
         5:
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SPPT5/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT5/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT5/'},         
         6:
            {0: '/data1/climlab/fritzen-dissertation/Chapter1/Physics/SPPT6/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT6/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT6/'},         
         7:
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SPPT7/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT7/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT7/'},         
         8:
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SPPT8/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SPPT8/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SPPT8/'},         
        },
    'SKEBS':
        {1: 
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SKEBS1/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SKEBS1/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SKEBS1/'},
         2:
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SKEBS2/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SKEBS2/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SKEBS2/'},        
         3:
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SKEBS3/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SKEBS3/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SKEBS3/'},        
         4:
            {0: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SKEBS4/',
             1: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED1/SKEBS4/',
             2: '/data2/climlab/fritzen-dissertation/Chapter1/Ch1Files/SEED2/SKEBS4/'},
        },
}
ENSEMBLE_TESTS = {'SPPT1': ["SPPT_1"],
                  'SPPT2': ["SPPT_2"],
                  'SPPT3': ["SPPT_3"],
                  'SPPT4': ["SPPT_4"],
                  'SPPT5': ["SPPT_5"],
                  'SPPT6': ["SPPT_6"],
                  'SPPT7': ["SPPT_7"],
                  'SPPT8': ["SPPT_8"],
                  'SKEBS1': ["SKEBS_1"],
                  'SKEBS2': ["SKEBS_2"],
                  'SKEBS3': ["SKEBS_3"],
                  'SKEBS4': ["SKEBS_4"],
                  'ALL_SPPT': ["SPPT_1", "SPPT_2", "SPPT_3", "SPPT_4", "SPPT_5", "SPPT_6", "SPPT_7", "SPPT_8"],
                  'ALL_SKEBS': ["SKEBS_1", "SKEBS_2", "SKEBS_3", "SKEBS_4"],
                  'ALL_STOCH': ["SPPT_1", "SPPT_2", "SPPT_3", "SPPT_4", "SPPT_5", "SPPT_6", "SPPT_7", "SPPT_8", "SKEBS_1", "SKEBS_2", "SKEBS_3", "SKEBS_4"],
                 }

"""
HELPER FUNCTIONS FOR COMPUTATIONS
"""

# NOTE: This function is NOT compatible with ensemble mode, for ensemble means of Theta-E, use the WRFOUT file function below.
def handle_theta_e_calc(VERIF_O, V_NARR_AIR, V_NARR_SHUM, narr_regridder, TIMESTR, CALC):
    vDay = int(TIMESTR.strftime("%d"))
    vHr = int(TIMESTR.strftime("%H"))

    vDay_step = (vDay-1) * 8
    vHr_step = int(((vHr / 24) * 8))

    vtStep = vDay_step + vHr_step

    # Compute NARR theta-e
    temp_1000 = V_NARR_AIR["air"][vtStep][0]
    temp_500 = V_NARR_AIR["air"][vtStep][16]
    q_1000 = V_NARR_SHUM["shum"][vtStep][0]
    q_500 = V_NARR_SHUM["shum"][vtStep][16] 

    w_1000 = calculation.q_to_w(q_1000)
    w_500 = calculation.q_to_w(q_500)
    
    th_e_1000 = calculation.calc_thetae(w_1000, temp_1000, 100000)
    th_e_500 = calculation.calc_thetae(w_500, temp_500, 50000)   

    th_e_wrfout_1000 = CALC["THE_E_1000"]
    th_e_wrfout_500 = CALC["THE_E_500"]
    
    # Regrid the WRF output to match NARR
    th_e_1000_regridded = narr_regridder(th_e_wrfout_1000)
    th_e_500_regridded = narr_regridder(th_e_wrfout_500)
    
    # Run verification tests
    V_ThE1000_Diff = th_e_1000_regridded - th_e_1000
    V_ThE500_Diff = th_e_500_regridded - th_e_500     
    
    V_ThE1000_RMSE = np.sqrt(np.mean((th_e_1000_regridded - th_e_1000)**2))
    V_ThE500_RMSE = np.sqrt(np.mean((th_e_500_regridded - th_e_500)**2))         
    
    for TE_T in FSS_THRESHS_THETAE:
        fss1 = []
        fss2 = []
        for f in FSS_Scales:
            fss1.append(calculation.FSS(th_e_1000_regridded, th_e_1000, TE_T, f))
            fss2.append(calculation.FSS(th_e_500_regridded, th_e_500, TE_T, f))
        VERIF_O["V_ThetaE_1000_" + str(TE_T) + "_VerfWindow_FSS"] = (['FSS_Scales'], fss1) 
        VERIF_O["V_ThetaE_500_" + str(TE_T) + "_VerfWindow_FSS"] = (['FSS_Scales'], fss2)
        
    VERIF_O["V_ThE_1000_WRF"] = (['y_narr', 'x_narr'], th_e_1000_regridded.data)
    VERIF_O["V_ThE_500_WRF"] = (['y_narr', 'x_narr'], th_e_500_regridded.data)
    VERIF_O["V_ThE_1000_NARR"] = (['y_narr', 'x_narr'], th_e_1000.data)
    VERIF_O["V_ThE_500_NARR"] = (['y_narr', 'x_narr'], th_e_500.data)            
    VERIF_O["V_ThetaE_1000_VerfWindow_Diff"] = (['y_narr', 'x_narr'], V_ThE1000_Diff.data)
    VERIF_O["V_ThetaE_500_VerfWindow_Diff"] = (['y_narr', 'x_narr'], V_ThE500_Diff.data)
    VERIF_O["V_ThetaE_1000_VerfWindow_RMSE"] = (['TimeFile'], [V_ThE1000_RMSE.data])
    VERIF_O["V_ThetaE_500_VerfWindow_RMSE"] = (['TimeFile'], [V_ThE500_RMSE.data])
    
    return VERIF_O

def handle_theta_e_wrfout(VERIF_O, V_NARR_AIR, V_NARR_SHUM, narr_regridder, TIMESTR, domain, WRF):
    ensembleMode = False
    if WRF is None:
        ensembleMode = True

    # Fetch the matching timestep from NARR
    vDay = int(TIMESTR.strftime("%d"))
    vHr = int(TIMESTR.strftime("%H"))

    vDay_step = (vDay-1) * 8
    vHr_step = int(((vHr / 24) * 8))

    vtStep = vDay_step + vHr_step

    try:
        # Compute NARR theta-e
        temp_1000 = V_NARR_AIR["air"][vtStep][0]
        temp_500 = V_NARR_AIR["air"][vtStep][16]
        q_1000 = V_NARR_SHUM["shum"][vtStep][0]
        q_500 = V_NARR_SHUM["shum"][vtStep][16] 

        w_1000 = calculation.q_to_w(q_1000)
        w_500 = calculation.q_to_w(q_500)
        
        th_e_1000 = calculation.calc_thetae(w_1000, temp_1000, 100000)
        th_e_500 = calculation.calc_thetae(w_500, temp_500, 50000)  
    except KeyError:
        tools.loggedPrint.instance().write("handle_theta_e_wrfout: " + timestep + ": WARN: Missing Theta-E data from NARR, skipping computation at this step.")
        return None             
    
    # Check if ensemble mode is active (WRF set to None)   
    if ensembleMode:
        #ENS_DATA = {}
    
        timestep = TIMESTR.strftime("%Y-%m-%d_%H_%M_%S")
        for test in ENSEMBLE_TESTS:
            tList = ENSEMBLE_TESTS[test]
            
            arrays_1000 = []
            arrays_500 = []
            
            for iTest in tList:
                # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
                testItem = iTest.split("_")
                testType = testItem[0]
                testNumber = int(testItem[1])
                # Fetch the associated object
                eObj = ENSEMBLE[testType][testNumber]
                
                # Process Files
                for idx in eObj:
                    tDir = eObj[idx]
                    
                    th_e_1000_regridded = None
                    th_e_500_regridded = None

                    # Check if we've already grabbed this data...
                    #if "TH_E_1000_" + iTest + "_" + str(idx) in ENS_DATA:                 
                    #    th_e_1000_regridded = ENS_DATA["TH_E_1000_" + iTest + "_" + str(idx)]
                    #    th_e_500_regridded = ENS_DATA["TH_E_500_" + iTest + "_" + str(idx)]
                    #else:
                    # We don't have it, so let's process it. 
                    sFile = tDir + "/output/wrfout_" + domain + "_" + timestep
                    if os.path.exists(sFile):
                        WRF = xr.open_dataset(sFile, engine='netcdf4')
                        
                        p_wrfout = WRF["P"][0] + WRF["PB"][0]
                        theta_wrf = WRF["T"][0] + 300.0
                        q_wrf = WRF["QVAPOR"][0]  

                        WRF.close()

                        temp_wrf = calculation.theta_to_temp(theta_wrf, p_wrfout)                      
                        th_e_wrfout = calculation.calc_thetae(q_wrf, temp_wrf, p_wrfout)     
                                  
                        th_e_wrfout_1000 = calculation.linear_interp(th_e_wrfout.data, p_wrfout.data, 1000)
                        th_e_wrfout_500 = calculation.linear_interp(th_e_wrfout.data, p_wrfout.data, 500)

                        # Regrid the WRF output to match NARR
                        th_e_1000_regridded = narr_regridder(th_e_wrfout_1000)
                        th_e_500_regridded = narr_regridder(th_e_wrfout_500)

                        #ENS_DATA["TH_E_1000_" + iTest + "_" + str(idx)] = th_e_1000_regridded
                        #ENS_DATA["TH_E_500_" + iTest + "_" + str(idx)] = th_e_500_regridded                        
                    else:
                        tools.loggedPrint.instance().write("handle_theta_e_wrfout: " + timestep + ": ENS WARN: Cannot find wrfout file [" + sFile + "].")
                        return None  

                    arrays_1000.append(th_e_1000_regridded)
                    arrays_500.append(th_e_500_regridded)
  
                    if not "V_ThetaE_1000_RMSE_" + iTest + "_" + str(idx) in VERIF_O:                  
                        V_ThE1000_RMSE = np.sqrt(np.mean((th_e_1000_regridded - th_e_1000)**2))
                        V_ThE500_RMSE = np.sqrt(np.mean((th_e_500_regridded - th_e_500)**2))
                        
                        VERIF_O["V_ThetaE_1000_RMSE_" + iTest + "_" + str(idx)] = (['TimeFile'], [V_ThE1000_RMSE.data])
                        VERIF_O["V_ThetaE_500_RMSE_" + iTest + "_" + str(idx)] = (['TimeFile'], [V_ThE500_RMSE.data])

                    # Compute FSS for the individual member if needed
                    if not "V_ThetaE_1000_FSS_" + iTest + "_" + str(idx) in VERIF_O:
                        for TE_T in FSS_THRESHS_THETAE:
                            fss1 = []
                            fss2 = []
                            for f in FSS_Scales:
                                fss1.append(calculation.FSS(th_e_1000_regridded, th_e_1000, TE_T, f))
                                fss2.append(calculation.FSS(th_e_500_regridded, th_e_500, TE_T, f))
                                
                                VERIF_O["V_ThetaE_1000_FSS_" + iTest + "_" + str(idx)] = (['FSS_Scales'], fss1) 
                                VERIF_O["V_ThetaE_500_FSS_" + iTest + "_" + str(idx)] = (['FSS_Scales'], fss2)                    
                    
            # Run ensemble mean test.
            MEAN_TH_E_1000 = np.array(arrays_1000).mean(axis = 0)
            MEAN_TH_E_500 = np.array(arrays_500).mean(axis = 0)           
            
            # Run verification tests
            V_ThE1000_Diff = MEAN_TH_E_1000 - th_e_1000
            V_ThE500_Diff = MEAN_TH_E_500 - th_e_500     
            
            V_ThE1000_RMSE = np.sqrt(np.mean((MEAN_TH_E_1000 - th_e_1000)**2))
            V_ThE500_RMSE = np.sqrt(np.mean((MEAN_TH_E_500 - th_e_500)**2))    

            for TE_T in FSS_THRESHS_THETAE:
                fss1 = []
                fss2 = []
                for f in FSS_Scales:
                    fss1.append(calculation.FSS(MEAN_TH_E_1000, th_e_1000, TE_T, f))
                    fss2.append(calculation.FSS(MEAN_TH_E_500, th_e_500, TE_T, f))
                    
                    VERIF_O["V_ThetaE_1000_FSS_" + test + "_EnsMean"] = (['FSS_Scales'], fss1) 
                    VERIF_O["V_ThetaE_500_FSS_" + test + "_EnsMean"] = (['FSS_Scales'], fss2)  

            VERIF_O["V_ThetaE_1000_" + test + "_EnsMean"] = (['y_narr', 'x_narr'], MEAN_TH_E_1000.data)
            VERIF_O["V_ThetaE_500_" + test + "_EnsMean"] = (['y_narr', 'x_narr'], MEAN_TH_E_500.data)
            if not "V_ThetaE_1000_NARR" in VERIF_O:
                VERIF_O["V_ThetaE_1000_NARR"] = (['y_narr', 'x_narr'], th_e_1000.data)
            if not "V_ThetaE_500_NARR" in VERIF_O:
                VERIF_O["V_ThetaE_500_NARR"] = (['y_narr', 'x_narr'], th_e_500.data)  
            VERIF_O["V_ThetaE_1000_" + test + "_Diff_EnsMean"] = (['y_narr', 'x_narr'], V_ThE1000_Diff.data)
            VERIF_O["V_ThetaE_500_" + test + "_Diff_EnsMean"] = (['y_narr', 'x_narr'], V_ThE500_Diff.data)
            VERIF_O["V_ThetaE_1000_" + test + "_RMSE_EnsMean"] = (['TimeFile'], [V_ThE1000_RMSE.data])
            VERIF_O["V_ThetaE_500_" + test + "_RMSE_EnsMean"] = (['TimeFile'], [V_ThE500_RMSE.data])
                    
    else:
        try:
            p_wrfout = WRF["P"][0] + WRF["PB"][0]
            theta_wrf = WRF["T"][0] + 300.0
            q_wrf = WRF["QVAPOR"][0]
        except KeyError:
            tools.loggedPrint.instance().write("handle_theta_e_wrfout: " + timestep + ": WARN: Missing Theta-E data from NARR, skipping computation at this step.")
            return None                  
        
        temp_wrf = calculation.theta_to_temp(theta_wrf, p_wrfout)                      
        th_e_wrfout = calculation.calc_thetae(q_wrf, temp_wrf, p_wrfout)     
                  
        th_e_wrfout_1000 = calculation.linear_interp(th_e_wrfout.data, p_wrfout.data, 1000)
        th_e_wrfout_500 = calculation.linear_interp(th_e_wrfout.data, p_wrfout.data, 500)
        
        #tools.loggedPrint.instance().write("runV: " + timestep + ": DEBUG: Regrid Output")           
        # Regrid the WRF output to match NARR
        th_e_1000_regridded = narr_regridder(th_e_wrfout_1000)
        th_e_500_regridded = narr_regridder(th_e_wrfout_500)
        
        #tools.loggedPrint.instance().write("runV: " + timestep + ": DEBUG: Verify")           
        # Run verification tests
        V_ThE1000_Diff = th_e_1000_regridded - th_e_1000
        V_ThE500_Diff = th_e_500_regridded - th_e_500     
        
        V_ThE1000_RMSE = np.sqrt(np.mean((th_e_1000_regridded - th_e_1000)**2))
        V_ThE500_RMSE = np.sqrt(np.mean((th_e_500_regridded - th_e_500)**2))
        
        # Save the output
        for TE_T in FSS_THRESHS_THETAE:
            fss1 = []
            fss2 = []
            for f in FSS_Scales:
                fss1.append(calculation.FSS(th_e_1000_regridded, th_e_1000, TE_T, f))
                fss2.append(calculation.FSS(th_e_500_regridded, th_e_500, TE_T, f))
            VERIF_O["V_ThetaE_1000_" + str(TE_T) + "_VerfWindow_FSS"] = (['FSS_Scales'], fss1) 
            VERIF_O["V_ThetaE_500_" + str(TE_T) + "_VerfWindow_FSS"] = (['FSS_Scales'], fss2)

        VERIF_O["V_ThE_1000_WRF"] = (['y_narr', 'x_narr'], th_e_1000_regridded.data)
        VERIF_O["V_ThE_500_WRF"] = (['y_narr', 'x_narr'], th_e_500_regridded.data)
        VERIF_O["V_ThE_1000_NARR"] = (['y_narr', 'x_narr'], th_e_1000.data)
        VERIF_O["V_ThE_500_NARR"] = (['y_narr', 'x_narr'], th_e_500.data)  
        VERIF_O["V_ThetaE_1000_VerfWindow_Diff"] = (['y_narr', 'x_narr'], V_ThE1000_Diff.data)
        VERIF_O["V_ThetaE_500_VerfWindow_Diff"] = (['y_narr', 'x_narr'], V_ThE500_Diff.data)
        VERIF_O["V_ThetaE_1000_VerfWindow_RMSE"] = (['TimeFile'], [V_ThE1000_RMSE.data])
        VERIF_O["V_ThetaE_500_VerfWindow_RMSE"] = (['TimeFile'], [V_ThE500_RMSE.data])
    
    return VERIF_O
        
def handle_surrogate_severe(VERIF_O, TIMESTR, domain, GeoMultiplier, WRF):
    # Surrogate Severe (Sobash et al., 2008)
    ensembleMode = False
    if WRF is None:
        ensembleMode = True

    if ensembleMode:   
        #ENS_DATA = {}
    
        timestep = TIMESTR.strftime("%Y-%m-%d_%H_%M_%S")
        tools.loggedPrint.instance().write("handle_surrogate_severe: LOG: starting ensemble analysis for timestep " + timestep + ".")
        for test in ENSEMBLE_TESTS:
            tList = ENSEMBLE_TESTS[test]
            
            array_u10 = []
            array_v10 = []
            array_wup = []
            array_wdn = []
            array_uh = []
            array_r1km = []
            
            for iTest in tList:
                # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
                testItem = iTest.split("_")
                testType = testItem[0]
                testNumber = int(testItem[1])
                # Fetch the associated object
                eObj = ENSEMBLE[testType][testNumber]
                
                # Process Files
                for idx in eObj:
                    tDir = eObj[idx]       
                    
                    U10 = None
                    V10 = None
                    WUP = None
                    WDN = None
                    UH = None
                    R1KM = None                    

                    sFile = tDir + "/output/wrfout_" + domain + "_" + timestep
                    if os.path.exists(sFile):
                        WRF = xr.open_dataset(sFile, engine='netcdf4')                          

                        U10 = WRF["U10"][0]
                        V10 = WRF["V10"][0]
                        WUP = WRF["W_UP_MAX"][0]
                        WDN = WRF["W_DN_MAX"][0]
                        UH = WRF["UP_HELI_MAX"][0]
                        R1KM = WRF["REFL_10CM_1KM"][0]                        
                        
                        WRF.close()
                    else:
                        tools.loggedPrint.instance().write("handle_surrogate_severe: " + timestep + ": ENS WARN: Cannot find wrfout file [" + sFile + "].")
                        return None 

                    array_u10.append(U10)
                    array_v10.append(V10)
                    array_wup.append(WUP)
                    array_wdn.append(WDN)
                    array_uh.append(UH)
                    array_r1km.append(R1KM)

                    if not "Surrogate_LSR_WND_" + iTest + "_" + str(idx) in VERIF_O:
                        WND10 = np.sqrt(U10**2 + V10**2)
                        THRESH_WND = (((WND10 >= 25)).astype(int)) * GeoMultiplier
                        THRESH_WUP = (((WUP >= 30)).astype(int)) * GeoMultiplier
                        THRESH_WDN = (((WDN <= -6)).astype(int)) * GeoMultiplier
                        THRESH_UH =  (((UH >= 75)).astype(int)) * GeoMultiplier
                        THRESH_REF = (((R1KM >= 58)).astype(int)) * GeoMultiplier
                        THRESH_ANY = (((WND10 >= 25) | (WUP >= 30) | (WDN <= -6) | (UH >= 75) | (R1KM >= 58)).astype(int)) * GeoMultiplier                        
                        
                        VERIF_O["Surrogate_LSR_WND_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], THRESH_WND.data)
                        VERIF_O["Surrogate_LSR_WUP_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], THRESH_WUP.data)
                        VERIF_O["Surrogate_LSR_WDN_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], THRESH_WDN.data)
                        VERIF_O["Surrogate_LSR_UH_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], THRESH_UH.data)
                        VERIF_O["Surrogate_LSR_REF_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], THRESH_REF.data)
                        VERIF_O["Surrogate_LSR_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], THRESH_ANY.data)                      
               
            tools.loggedPrint.instance().write("handle_surrogate_severe: LOG: ensemble analysis for timestep " + timestep + " complete.")
               
            # Run ensemble mean test.
            MEAN_U10 = np.array(array_u10).mean(axis = 0)
            MEAN_V10 = np.array(array_v10).mean(axis = 0)
            MEAN_WUP = np.array(array_wup).mean(axis = 0)
            MEAN_WDN = np.array(array_wdn).mean(axis = 0)
            MEAN_UH = np.array(array_uh).mean(axis = 0)
            MEAN_R1KM = np.array(array_r1km).mean(axis = 0)

            MEAN_WND10 = np.sqrt(MEAN_U10**2 + MEAN_V10**2)
            THRESH_WND = (((MEAN_WND10 >= 25)).astype(int)) * GeoMultiplier
            THRESH_WUP = (((MEAN_WUP >= 30)).astype(int)) * GeoMultiplier
            THRESH_WDN = (((MEAN_WDN <= -6)).astype(int)) * GeoMultiplier
            THRESH_UH =  (((MEAN_UH >= 75)).astype(int)) * GeoMultiplier
            THRESH_REF = (((MEAN_R1KM >= 58)).astype(int)) * GeoMultiplier
            THRESH_ANY = (((MEAN_WND10 >= 25) | (MEAN_WUP >= 30) | (MEAN_WDN <= -6) | (MEAN_UH >= 75) | (MEAN_R1KM >= 58)).astype(int)) * GeoMultiplier 

            VERIF_O["Surrogate_LSR_WND_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], THRESH_WND.data)
            VERIF_O["Surrogate_LSR_WUP_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], THRESH_WUP.data)
            VERIF_O["Surrogate_LSR_WDN_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], THRESH_WDN.data)
            VERIF_O["Surrogate_LSR_UH_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], THRESH_UH.data)
            VERIF_O["Surrogate_LSR_REF_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], THRESH_REF.data)
            VERIF_O["Surrogate_LSR_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], THRESH_ANY.data)
    
    else:  
        U10 = WRF["U10"][0]
        V10 = WRF["V10"][0]
        WUP = WRF["W_UP_MAX"][0]
        WDN = WRF["W_DN_MAX"][0]
        UH = WRF["UP_HELI_MAX"][0]
        R1KM = WRF["REFL_10CM_1KM"][0]

        WND10 = np.sqrt(U10**2 + V10**2)
        
        # Check if any of the thresholds have been passed.
        THRESH_WND = (((WND10 >= 25)).astype(int)) * GeoMultiplier
        THRESH_WUP = (((WUP >= 30)).astype(int)) * GeoMultiplier
        THRESH_WDN = (((WDN <= -6)).astype(int)) * GeoMultiplier
        THRESH_UH =  (((UH >= 75)).astype(int)) * GeoMultiplier
        THRESH_REF = (((R1KM >= 58)).astype(int)) * GeoMultiplier
        THRESH_ANY = (((WND10 >= 25) | (WUP >= 30) | (WDN <= -6) | (UH >= 75) | (R1KM >= 58)).astype(int)) * GeoMultiplier
        # Save to output.
        VERIF_O["Surrogate_LSR_WND"] = (['y_wrf', 'x_wrf'], THRESH_WND.data)
        VERIF_O["Surrogate_LSR_WUP"] = (['y_wrf', 'x_wrf'], THRESH_WUP.data)
        VERIF_O["Surrogate_LSR_WDN"] = (['y_wrf', 'x_wrf'], THRESH_WDN.data)
        VERIF_O["Surrogate_LSR_UH"] = (['y_wrf', 'x_wrf'], THRESH_UH.data)
        VERIF_O["Surrogate_LSR_REF"] = (['y_wrf', 'x_wrf'], THRESH_REF.data)
        VERIF_O["Surrogate_LSR"] = (['y_wrf', 'x_wrf'], THRESH_ANY.data)
    
    return VERIF_O
    
def handle_stage_4(VERIF_O, V_STAGE4, st4_regridder, TIMESTR, domain, prev_6hr_file, WRF):
    ensembleMode = False
    if WRF is None and prev_6hr_file is None:
        ensembleMode = True

    accprec = V_STAGE4["accprc"]
    # Compute FSS Thresholds         
    pct = np.nanpercentile(accprec.data, FSS_THRESHS_PRECIP_PERCENTILES)
    VERIF_O.attrs["Stage4_FSS_Thresholds"] = pct
    #  

    if ensembleMode:
        #ENS_DATA = {}
    
        timestep = TIMESTR.strftime("%Y-%m-%d_%H_%M_%S")
        PREV6HR = TIMESTR - timedelta(hours=6)
        prev_timestep = PREV6HR.strftime("%Y-%m-%d_%H_%M_%S")
        
        for test in ENSEMBLE_TESTS:
            tList = ENSEMBLE_TESTS[test]
            
            array_accp = []
            
            for iTest in tList:
                # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
                testItem = iTest.split("_")
                testType = testItem[0]
                testNumber = int(testItem[1])
                # Fetch the associated object
                eObj = ENSEMBLE[testType][testNumber]
                
                # Process Files
                for idx in eObj:
                    tDir = eObj[idx]
                    
                    six_hr_precip_regridded = None
                    
                    # Check if we've already grabbed this data...
                    #if "Six_Hr_Precip_" + iTest + "_" + str(idx) in ENS_DATA:                 
                    #    six_hr_precip_regridded = ENS_DATA["Six_Hr_Precip_" + iTest + "_" + str(idx)]
                    #else:
                    # We don't have it, so let's process it. 
                    pFile = tDir + "/output/wrfout_" + domain + "_" + prev_timestep
                    if not os.path.exists(pFile):
                        #If the 6-hr prior file is not found, then this is the first timestep and we may pass.
                        tools.loggedPrint.instance().write("handle_stage_4: " + timestep + ": WARN: File (" + pFile + ") not found, ignoring ensemble Stage IV verification for this time step.")  
                        return None
                    sFile = tDir + "/output/wrfout_" + domain + "_" + timestep
                    if os.path.exists(sFile):
                        WRF = xr.open_dataset(sFile, engine='netcdf4')
                        WRF_P = xr.open_dataset(pFile, engine='netcdf4')
                        
                        precip_ending_now = WRF["RAINC"][0] + WRF["RAINNC"][0]
                        precip_ending_before = WRF_P["RAINC"][0] + WRF_P["RAINNC"][0]
                        
                        WRF.close()
                        WRF_P.close()

                        six_hr_precip = precip_ending_now - precip_ending_before
                        six_hr_precip_regridded = st4_regridder(six_hr_precip)

                        #ENS_DATA["Six_Hr_Precip_" + iTest + "_" + str(idx)] = six_hr_precip_regridded                              
                    else:
                        tools.loggedPrint.instance().write("handle_stage_4: " + timestep + ": ENS WARN: Cannot find wrfout file [" + sFile + "].")
                        return None  

                    array_accp.append(six_hr_precip_regridded)
                    
                    if not "V_AccP_RMSE_" + iTest + "_" + str(idx) in VERIF_O:                  
                        V_AccP_RMSE = np.sqrt(np.mean((six_hr_precip_regridded - accprec)**2))                       
                        VERIF_O["V_AccP_RMSE_" + iTest + "_" + str(idx)] = (['TimeFile'], [V_AccP_RMSE.data])
                        VERIF_O["V_AccP_" + iTest] = (['y_stage4', 'x_stage4'], six_hr_precip_regridded.data)

                    # Compute FSS for the individual member if needed
                    for iPC, PC_T in enumerate(pct):
                        if not "V_AccP_FSS_" + str(FSS_THRESHS_PRECIP_PERCENTILES[iPC]) + "_" + iTest + "_" + str(idx) in VERIF_O:
                            fss = []
                            for f in FSS_Scales:
                                fss.append(calculation.FSS(six_hr_precip_regridded, accprec, PC_T, f)) 
                            
                            VERIF_O["V_AccP_FSS_" + str(FSS_THRESHS_PRECIP_PERCENTILES[iPC]) + "_" + iTest + "_" + str(idx)] = (['FSS_Scales'], fss)   
                            
            # Run ensemble mean test.
            MEAN_six_hour_precip = np.array(array_accp).mean(axis = 0)
            
            # Run verification tests
            V_AccP_Diff = MEAN_six_hour_precip - accprec
            V_AccP_RMSE = np.sqrt(np.mean((MEAN_six_hour_precip - accprec)**2))              
            
            for iPC, PC_T in enumerate(pct):
                fss = []
                for f in FSS_Scales:
                    fss.append(calculation.FSS(MEAN_six_hour_precip, accprec, PC_T, f)) 
                
                VERIF_O["V_AccP_" + str(FSS_THRESHS_PRECIP_PERCENTILES[iPC]) + "_FSS_" + test + "_EnsMean"] = (['FSS_Scales'], fss)
            
            VERIF_O["V_AccP_Diff_" + test + "_EnsMean"] = (['y_stage4', 'x_stage4'], V_AccP_Diff.data)
            VERIF_O["V_AccP_RMSE_" + test + "_EnsMean"] = (['TimeFile'], [V_AccP_RMSE.data])
            VERIF_O["V_AccP_" + test + "_EnsMean"] = (['y_stage4', 'x_stage4'], MEAN_six_hour_precip.data)
    else:
        WRF_P = xr.open_dataset(prev_6hr_file, engine='netcdf4')
        precip_ending_now = WRF["RAINC"][0] + WRF["RAINNC"][0]
        precip_ending_before = WRF_P["RAINC"][0] + WRF_P["RAINNC"][0]
        WRF_P.close()
        six_hr_precip = precip_ending_now - precip_ending_before
        six_hr_precip_regridded = st4_regridder(six_hr_precip)
        
        V_AccP_Diff = six_hr_precip_regridded - accprec
        V_AccP_RMSE = np.sqrt(np.mean((six_hr_precip_regridded - accprec)**2))              
        
        for iPC, PC_T in enumerate(pct):
            fss = []
            for f in FSS_Scales:
                fss.append(calculation.FSS(six_hr_precip_regridded, accprec, PC_T, f)) 
            
            VERIF_O["V_AccP_" + str(FSS_THRESHS_PRECIP_PERCENTILES[iPC]) + "_FSS"] = (['FSS_Scales'], fss)
        
        VERIF_O["AccP_6Hr"] = (['y_wrf', 'x_wrf'], six_hr_precip.data)
        VERIF_O["V_AccP_Diff"] = (['y_stage4', 'x_stage4'], V_AccP_Diff.data)
        VERIF_O["V_AccP_RMSE"] = (['TimeFile'], [V_AccP_RMSE.data])
    
    return VERIF_O

def handle_gridrad(VERIF_O, V_GRIDRAD, gridrad_regridder, TIMESTR, domain, SUBHR, V_GRIDRAD_R = None, gridrad_r_regridder = None):
    ensembleMode = False
    if SUBHR is None:
        ensembleMode = True
        
    ref_gridrad = V_GRIDRAD["Reflectivity"][0]
    ref_gridrad_r = None
    if V_GRIDRAD_R is not None:
        ref_gridrad_r = V_GRIDRAD_R["Reflectivity"][0]    
        
    if ensembleMode:
        #ENS_DATA = {}
    
        timestep = TIMESTR.strftime("%Y-%m-%d_%H_%M_%S")
        for test in ENSEMBLE_TESTS:
            tList = ENSEMBLE_TESTS[test]
            
            array_ref = []
            array_refr = None
            if V_GRIDRAD_R:
                array_refr = []
            
            for iTest in tList:
                # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
                testItem = iTest.split("_")
                testType = testItem[0]
                testNumber = int(testItem[1])
                # Fetch the associated object
                eObj = ENSEMBLE[testType][testNumber]
                
                # Process Files
                for idx in eObj:
                    tDir = eObj[idx]
                    
                    ref_wrf_regridded = None
                    ref_wrf_regridded_r = None

                    # Check if we've already grabbed this data...
                    #if "REF_" + iTest + "_" + str(idx) in ENS_DATA:                 
                    #    ref_wrf_regridded = ENS_DATA["REF_" + iTest + "_" + str(idx)]
                    #    if V_GRIDRAD_R:
                    #       ref_wrf_regridded_r = ENS_DATA["REF_R_" + iTest + "_" + str(idx)] 
                    #else:
                    # We don't have it, so let's process it. 
                    sFile = tDir + "/output/subhr_" + domain + "_" + timestep
                    if os.path.exists(sFile):
                        SUBHR = xr.open_dataset(sFile, engine='netcdf4')
                        
                        ref_wrf = SUBHR["REFD"][0]     
                        ref_wrf_regridded = gridrad_regridder(ref_wrf)
                        if V_GRIDRAD_R:
                            ref_wrf_regridded_r = gridrad_r_regridder(ref_wrf)

                        SUBHR.close()
                      
                    else:
                        tools.loggedPrint.instance().write("handle_gridrad: " + timestep + ": ENS WARN: Cannot find subhr file [" + sFile + "].")
                        return None  

                    array_ref.append(ref_wrf_regridded)
                    if ref_wrf_regridded_r is not None:
                        array_refr.append(ref_wrf_regridded_r)

                    # Compute individual member items
                    if not "V_Ref0_" + iTest + "_" + str(idx) + "_Diff" in VERIF_O:
                        V_Ref0_Diff = ref_wrf_regridded - ref_gridrad
                        VERIF_O["V_Ref0_" + iTest + "_" + str(idx) + "_Diff"] = (['y_gridrad', 'x_gridrad'], V_Ref0_Diff.data)
                        
                    if not "V_Ref0_" + iTest + "_" + str(idx) + "_RMSE" in VERIF_O:
                        V_Ref0_RMSE = np.sqrt(np.mean((ref_wrf_regridded - ref_gridrad)**2))
                        VERIF_O["V_Ref0_" + iTest + "_" + str(idx) + "_RMSE"] = (['TimeFile'], [V_Ref0_RMSE.data]) 

                    if not "V_Ref0_" + iTest + "_" + str(idx) + "_FSS" in VERIF_O:
                        fss = []
                        for f in FSS_Scales:
                            fss.append(calculation.FSS(ref_wrf_regridded, ref_gridrad, FSS_THRESH_SIMREF, f)) 
                        VERIF_O["V_Ref0_" + iTest + "_" + str(idx) + "_FSS"] = (['FSS_Scales'], fss) 
   
                    if V_GRIDRAD_R is not None:
                        # These are the same across all GridRAD files, it is safe for overwrite.
                        VERIF_O.coords["Latitude_GRIDRAD_R"] = (["y_gridrad_r"], V_GRIDRAD_R["Latitude"].data)
                        VERIF_O.coords["Longitude_GRIDRAD_R"] = (["x_gridrad_r"], V_GRIDRAD_R["Longitude"].data)                    
                    
                        if not "V_Ref0_R_" + iTest + "_" + str(idx) + "_Diff" in VERIF_O:
                            V_Ref0_R_Diff = ref_wrf_regridded_r - ref_gridrad_r
                            VERIF_O["V_Ref0_R_" + iTest + "_" + str(idx) + "_Diff"] = (['y_gridrad_r', 'x_gridrad_r'], V_Ref0_R_Diff.data)
                            
                        if not "V_Ref0_R_" + iTest + "_" + str(idx) + "_RMSE" in VERIF_O:
                            V_Ref0_R_RMSE = np.sqrt(np.mean((ref_wrf_regridded_r - ref_gridrad_r)**2))
                            VERIF_O["V_Ref0_R_" + iTest + "_" + str(idx) + "_RMSE"] = (['TimeFile'], [V_Ref0_R_RMSE.data]) 

                        if not "V_Ref0_R_" + iTest + "_" + str(idx) + "_FSS" in VERIF_O:
                            fss_r = []
                            for f in FSS_Scales:
                                fss_r.append(calculation.FSS(ref_wrf_regridded_r, ref_gridrad_r, FSS_THRESH_SIMREF, f))
                            VERIF_O["V_Ref0_R_" + iTest + "_" + str(idx) + "_FSS"] = (['FSS_Scales'], fss_r)  
                            
            # Ensemble Mean
            MEAN_REF = np.array(array_ref).mean(axis = 0)

            fss = []
            for f in FSS_Scales:
                fss.append(calculation.FSS(MEAN_REF, ref_gridrad, FSS_THRESH_SIMREF, f))
          
            V_Ref0_Diff = MEAN_REF - ref_gridrad
            V_Ref0_RMSE = np.sqrt(np.mean((MEAN_REF - ref_gridrad)**2))   

            VERIF_O["V_Ref0_" + test + "_EnsMean"] = (['y_gridrad', 'x_gridrad'], MEAN_REF.data)
            VERIF_O["V_Ref0_Diff_" + test + "_EnsMean"] = (['y_gridrad', 'x_gridrad'], V_Ref0_Diff.data)
            VERIF_O["V_Ref0_RMSE_" + test + "_EnsMean"] = (['TimeFile'], [V_Ref0_RMSE.data])
            VERIF_O["V_Ref0_FSS_" + test + "_EnsMean"] = (['FSS_Scales'], fss)  

            if array_refr is not None:
                MEAN_REF_R = np.array(array_refr).mean(axis = 0)   
                fss = []
                for f in FSS_Scales:
                    fss.append(calculation.FSS(MEAN_REF_R, ref_gridrad_r, FSS_THRESH_SIMREF, f))
              
                V_Ref0_Diff = MEAN_REF_R - ref_gridrad_r
                V_Ref0_RMSE = np.sqrt(np.mean((MEAN_REF_R - ref_gridrad_r)**2))   

                VERIF_O["V_Ref0_R_" + test + "_EnsMean"] = (['y_gridrad_r', 'x_gridrad_r'], MEAN_REF_R.data)
                VERIF_O["V_Ref0_R_Diff_" + test + "_EnsMean"] = (['y_gridrad_r', 'x_gridrad_r'], V_Ref0_Diff.data)
                VERIF_O["V_Ref0_R_RMSE_" + test + "_EnsMean"] = (['TimeFile'], [V_Ref0_RMSE.data])
                VERIF_O["V_Ref0_R_FSS_" + test + "_EnsMean"] = (['FSS_Scales'], fss)                                
            
    else:
        ref_wrf = SUBHR["REFD"][0]     
        ref_wrf_regridded = gridrad_regridder(ref_wrf)
      
        fss = []
        for f in FSS_Scales:
            fss.append(calculation.FSS(ref_wrf_regridded, ref_gridrad, FSS_THRESH_SIMREF, f))
      
        V_Ref0_Diff = ref_wrf_regridded - ref_gridrad
        V_Ref0_RMSE = np.sqrt(np.mean((ref_wrf_regridded - ref_gridrad)**2))
        
        VERIF_O["V_Ref0_Diff"] = (['y_gridrad', 'x_gridrad'], V_Ref0_Diff.data)
        VERIF_O["V_Ref0_RMSE"] = (['TimeFile'], [V_Ref0_RMSE.data])
        VERIF_O["V_Ref0_FSS"] = (['FSS_Scales'], fss)  

        if V_GRIDRAD_R is not None:   
            ref_wrf_regridded_r = gridrad_r_regridder(ref_wrf)
          
            fss_r = []
            for f in FSS_Scales:
                fss_r.append(calculation.FSS(ref_wrf_regridded_r, ref_gridrad_r, FSS_THRESH_SIMREF, f))
          
            V_Ref0_DiffR = ref_wrf_regridded_r - ref_gridrad_r
            V_Ref0_RMSER = np.sqrt(np.mean((ref_wrf_regridded_r - ref_gridrad_r)**2))

            VERIF_O.coords["Latitude_GRIDRAD_R"] = (["y_gridrad_r"], V_GRIDRAD_R["Latitude"].data)
            VERIF_O.coords["Longitude_GRIDRAD_R"] = (["x_gridrad_r"], V_GRIDRAD_R["Longitude"].data)

            VERIF_O["V_Ref0_Diff_R"] = (['y_gridrad_r', 'x_gridrad_r'], V_Ref0_DiffR.data)
            VERIF_O["V_Ref0_RMSE_R"] = (['TimeFile'], [V_Ref0_RMSER.data])
            VERIF_O["V_Ref0_FSS_R"] = (['FSS_Scales'], fss_r)  
            
    return VERIF_O

def add_UH_NP(VERIF_O, TIMESTR, domain, SUBHR):
    ensembleMode = False
    if SUBHR is None:
        ensembleMode = True
    
    if ensembleMode:
        #ENS_DATA = {}
    
        timestep = TIMESTR.strftime("%Y-%m-%d_%H_%M_%S")
        for test in ENSEMBLE_TESTS:
            tList = ENSEMBLE_TESTS[test]
            
            array_uh25 = []
            array_uh03 = []
            
            for iTest in tList:
                # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
                testItem = iTest.split("_")
                testType = testItem[0]
                testNumber = int(testItem[1])
                # Fetch the associated object
                eObj = ENSEMBLE[testType][testNumber]
                
                # Process Files
                for idx in eObj:
                    tDir = eObj[idx] 

                    UH25 = None
                    UH03 = None

                    # Check if we've already grabbed this data...
                    #if "UH25_" + iTest + "_" + str(idx) in ENS_DATA:                 
                    #    UH25 = ENS_DATA["UH25_" + iTest + "_" + str(idx)]
                    #    UH03 = ENS_DATA["UH03_" + iTest + "_" + str(idx)] 
                    #else:
                    # We don't have it, so let's process it. 
                    sFile = tDir + "/output/subhr_" + domain + "_" + timestep
                    if os.path.exists(sFile):
                        SUBHR = xr.open_dataset(sFile, engine='netcdf4') 
                        UH25 = SUBHR["UH"][0]
                        UH03 = SUBHR["UH03"][0]
                        SUBHR.close()
                        
                        #ENS_DATA["UH25_" + iTest + "_" + str(idx)] = UH25
                        #ENS_DATA["UH03_" + iTest + "_" + str(idx)] = UH03
                    else:
                        tools.loggedPrint.instance().write("add_UH_NP: " + timestep + ": ENS WARN: Cannot find subhr file [" + sFile + "].")
                        return None                              
                            
                    array_uh25.append(UH25)
                    array_uh03.append(UH03)
                            
                    # Compute NP for individual members, if needed.
                    if not "NP_UH25_" + iTest + "_" + str(idx) in VERIF_O:
                        NP_UH25 = calculation.NProb(UH25, 75, 10)
                        NP_UH03 = calculation.NProb(UH03, 100, 10)
                        
                        VERIF_O["NP_UH25_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], NP_UH25)
                        VERIF_O["NP_UH03_" + iTest + "_" + str(idx)] = (['y_wrf', 'x_wrf'], NP_UH03)
                        
            # Ensemble Mean
            MEAN_UH25 = np.array(array_uh25).mean(axis = 0)
            MEAN_UH03 = np.array(array_uh03).mean(axis = 0)
            
            NP_UH25 = calculation.NProb(MEAN_UH25, 75, 10)
            NP_UH03 = calculation.NProb(MEAN_UH03, 100, 10)
            
            VERIF_O["NP_UH25_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], NP_UH25)
            VERIF_O["NP_UH03_" + test + "_EnsMean"] = (['y_wrf', 'x_wrf'], NP_UH03)            
    else:
        uh_2_5_wrf = SUBHR["UH"][0]
        NP_UH25 = calculation.NProb(uh_2_5_wrf, 75, 10)         
        VERIF_O["V_UH25_NP"] = (['y_wrf', 'x_wrf'], NP_UH25)
      
        # 0 - 3km Updraft Helicity
        uh_0_3_wrf = SUBHR["UH03"][0]
        NP_UH03 = calculation.NProb(uh_0_3_wrf, 100, 10)    
        VERIF_O["V_UH03_NP"] = (['y_wrf', 'x_wrf'], NP_UH03)
                
    return VERIF_O
    
def handle_gridrad_mesh(VERIF_O, V_GRIDRAD_MESH, gridrad_mesh_regridder, TIMESTR, domain, AFWA):
    ensembleMode = False
    if AFWA is None:
        ensembleMode = True
    
    PREVDAY = TIMESTR - timedelta(days=1)
    # The GridRad Severe mesh is on a convective day (12z - 12z), so check the timestep
    if(int(TIMESTR.strftime("%H")) < 12):
        TS = int(((TIMESTR - datetime.strptime(PREVDAY.strftime("%Y%m%d_120000"), '%Y%m%d_%H%M%S')).total_seconds()) / 300)
    else:    
        CUR_START = datetime.strptime(TIMESTR.strftime("%Y%m%d_120000"), '%Y%m%d_%H%M%S')
        TS = int(((TIMESTR - CUR_START).total_seconds()) / 300) 
        
    hail_mesh = V_GRIDRAD_MESH["MESH95"][TS]    
    
    if ensembleMode:
        #ENS_DATA = {}
    
        timestep = TIMESTR.strftime("%Y-%m-%d_%H_%M_%S")
        for test in ENSEMBLE_TESTS:
            tList = ENSEMBLE_TESTS[test]
            
            array_hail = []
            
            for iTest in tList:
                # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
                testItem = iTest.split("_")
                testType = testItem[0]
                testNumber = int(testItem[1])
                # Fetch the associated object
                eObj = ENSEMBLE[testType][testNumber]
                
                # Process Files
                for idx in eObj:
                    tDir = eObj[idx] 

                    hail_regridded = None

                    # Check if we've already grabbed this data...
                    #if "HAIL_" + iTest + "_" + str(idx) in ENS_DATA:                 
                    #    hail_regridded = ENS_DATA["HAIL_" + iTest + "_" + str(idx)]
                    #else:
                    # We don't have it, so let's process it. 
                    aFile = tDir + "/output/AFWA_" + domain + "_" + timestep
                    if os.path.exists(aFile):
                        AFWA = xr.open_dataset(aFile, engine='netcdf4') 
                        hail_wrf = AFWA["AFWA_HAIL"][0]
                        AFWA.close()
                        
                        hail_regridded = gridrad_mesh_regridder(hail_wrf)
                        #ENS_DATA["HAIL_" + iTest + "_" + str(idx)] = hail_regridded
                    else:
                        tools.loggedPrint.instance().write("handle_gridrad_mesh: " + timestep + ": ENS WARN: Cannot find AFWA file [" + aFile + "].")
                        return None                                
                
                    array_hail.append(hail_regridded)
                    
                    # Run tests on individual members, if needed
                    if not "V_HailMesh_" + iTest + "_" + str(idx) + "_Diff" in VERIF_O:
                        V_HailMesh_Diff = hail_regridded - hail_mesh
                        VERIF_O["V_HailMesh_" + iTest + "_" + str(idx) + "_Diff"] = (['y_gridradmesh', 'x_gridradmesh'], V_HailMesh_Diff.data)
                        
                    if not "V_HailMesh_" + iTest + "_" + str(idx) + "_RMSE" in VERIF_O:
                        V_HailMesh_RMSE = np.sqrt(np.mean((hail_regridded - hail_mesh)**2))
                        VERIF_O["V_HailMesh_" + iTest + "_" + str(idx) + "_RMSE"] = (['TimeFile'], [V_HailMesh_RMSE.data]) 

                    if not "V_HailMesh_" + iTest + "_" + str(idx) + "_FSS" in VERIF_O:
                        fss = []
                        for f in FSS_Scales:
                            fss.append(calculation.FSS(hail_regridded, hail_mesh, FSS_THRESH_HAILSIZE, f)) 
                        VERIF_O["V_HailMesh_" + iTest + "_" + str(idx) + "_FSS"] = (['FSS_Scales'], fss)                    
                    
            # Ensemble Mean
            MEAN_Hail = np.array(array_hail).mean(axis = 0)
            fss = []
            for f in FSS_Scales:
                fss.append(calculation.FSS(MEAN_Hail, hail_mesh, FSS_THRESH_HAILSIZE, f)) # 25mm ~ 1 in
          
            V_HailMesh_Diff = MEAN_Hail - hail_mesh
            V_HailMesh_RMSE = np.sqrt(np.mean((MEAN_Hail - hail_mesh)**2))
            
            VERIF_O["V_HailMesh_" + test + "_EnsMean"] = (['y_gridradmesh', 'x_gridradmesh'], MEAN_Hail.data)
            VERIF_O["V_HailMesh_Diff_" + test + "_EnsMean"] = (['y_gridradmesh', 'x_gridradmesh'], V_HailMesh_Diff.data)
            VERIF_O["V_HailMesh_RMSE_" + test + "_EnsMean"] = (['TimeFile'], [V_HailMesh_RMSE.data])
            VERIF_O["V_HailMesh_FSS_" + test + "_EnsMean"] = (['FSS_Scales'], fss)               
            
    else:
        hail_wrf = AFWA["AFWA_HAIL"][0]
        hail_wrf_regridded = gridrad_mesh_regridder(hail_wrf)
        
        fss = []
        for f in FSS_Scales:
            fss.append(calculation.FSS(hail_wrf_regridded, hail_mesh, FSS_THRESH_HAILSIZE, f)) # 25mm ~ 1 in
      
        V_HailMesh_Diff = hail_wrf_regridded - hail_mesh
        V_HailMesh_RMSE = np.sqrt(np.mean((hail_wrf_regridded - hail_mesh)**2))
        
        VERIF_O["WRF_HailMesh_Regridded"] = (['y_gridradmesh', 'x_gridradmesh'], hail_wrf_regridded.data)
        VERIF_O["V_HailMesh_Diff"] = (['y_gridradmesh', 'x_gridradmesh'], V_HailMesh_Diff.data)
        VERIF_O["V_HailMesh_RMSE"] = (['TimeFile'], [V_HailMesh_RMSE.data])
        VERIF_O["V_HailMesh_FSS"] = (['FSS_Scales'], fss)   
        
    return VERIF_O

def handle_lsr_mesh_verification(fVerf, 
                                 mesh, 
                                 vDir,
                                 centroid_x,
                                 centroid_y,
                                 lDateTM,
                                 fSPC_Wnd, 
                                 fSPC_Hail, 
                                 fSPC_Tor, 
                                 THRESH_WND_NAME = "Daily_LSR_WND",
                                 THRESH_WUP_NAME = "Daily_LSR_WUP",
                                 THRESH_WDN_NAME = "Daily_LSR_WDN",
                                 THRESH_UH_NAME = "Daily_LSR_UH",
                                 THRESH_REF_NAME = "Daily_LSR_REF",
                                 THRESH_TOT_NAME = "Daily_LSR_Total"):
    if not os.path.exists(fVerf):
        tools.loggedPrint.instance().write("handle_lsr_mesh_verification: ERROR: Cannot find file [" + fVerf + "]")
        return None
    xrVerf = xr.open_dataset(fVerf, engine='netcdf4')
    HAS_VMESH_COORDS = False
    try:
        HAS_VMESH_COORDS = bool(xrVerf.attrs["HAS_VMESH_COORDS"])
    except KeyError:
        pass

    if(HAS_VMESH_COORDS != True): 
        xrVerf.coords["Longitude_LSRMESH"] = (['x_lsrmesh', 'y_lsrmesh'], centroid_x)
        xrVerf.coords["Latitude_LSRMESH"] = (['x_lsrmesh', 'y_lsrmesh'], centroid_y)    
        xrVerf.to_netcdf(vDir + "/verification_output_" + lDateTM.strftime("%Y%m%d_") + "120000.nc.new")
        xrVerf.close()
        ovFiles = sorted(glob.glob(vDir + "/verification_output_*.nc.new"))
        if(len(ovFiles) > 0):
            tools.loggedPrint.instance().write("Overwritting existing netcdf files.")
            for of in ovFiles:
                oldFile = of[:-4]
                tools.popen("rm " + oldFile)
                tools.popen("mv " + of + " " + oldFile)
        xrVerf = xr.open_dataset(fVerf, engine='netcdf4')
    
    # SPC reports sometimes come out in invalid .csv files (IE: They put commas as part of the "comments" column), we need
    #  to directly specify the header info so pandas doesn't shift the rows to the right in the event of an invalid CSV file.
    dfWnd = pd.read_csv(fSPC_Wnd, 
                        names = ["Time", "Speed", "Location", "County", "State", "Lat", "Lon", "Comments"],
                        usecols = ["Time", "Location", "County", "State", "Lat", "Lon"])
    dfHail = pd.read_csv(fSPC_Hail,
                        names = ["Time", "Size", "Location", "County", "State", "Lat", "Lon", "Comments"],
                        usecols = ["Time", "Location", "County", "State", "Lat", "Lon"])
    dfTor = pd.read_csv(fSPC_Tor,                     
                        names = ["Time", "F-Scale", "Location", "County", "State", "Lat", "Lon", "Comments"],
                        usecols = ["Time", "Location", "County", "State", "Lat", "Lon"])
    # As a result of us defining the header names, the header row is read in as the first row, drop it.            
    dfWnd.drop(index=dfWnd.index[0], axis=0, inplace=True)
    dfHail.drop(index=dfHail.index[0], axis=0, inplace=True)
    dfTor.drop(index=dfTor.index[0], axis=0, inplace=True)
    # Finally, we need to type-cast the data we need from the frame.
    dfWnd["Time"] = dfWnd["Time"].astype(int)
    dfWnd["Lat"] = dfWnd["Lat"].astype(float)
    dfWnd["Lon"] = dfWnd["Lon"].astype(float)     
    dfHail["Time"] = dfHail["Time"].astype(int)
    dfHail["Lat"] = dfHail["Lat"].astype(float)
    dfHail["Lon"] = dfHail["Lon"].astype(float)  
    dfTor["Time"] = dfTor["Time"].astype(int)
    dfTor["Lat"] = dfTor["Lat"].astype(float)
    dfTor["Lon"] = dfTor["Lon"].astype(float)              
    
    gdfSPC = gpd.GeoDataFrame(columns = ["Time", "Location", "County", "State", "Lat", "Lon", "geometry"], geometry="geometry", crs="epsg:4326")        
    
    gdfWnd = gpd.GeoDataFrame(dfWnd, geometry=gpd.points_from_xy(dfWnd.Lon, dfWnd.Lat))
    gdfHail = gpd.GeoDataFrame(dfHail, geometry=gpd.points_from_xy(dfHail.Lon, dfHail.Lat))
    gdfTor = gpd.GeoDataFrame(dfTor, geometry=gpd.points_from_xy(dfTor.Lon, dfTor.Lat))
    
    gdfSPC = gdfSPC.append(gdfWnd, ignore_index=True)
    gdfSPC = gdfSPC.append(gdfHail, ignore_index=True)
    gdfSPC = gdfSPC.append(gdfTor, ignore_index=True)

    tools.loggedPrint.instance().write("Converting thresholds into pandas dataframes.")
    # Load the thresholds we calculated previously into geopandas
    THRESH_WND = xrVerf[THRESH_WND_NAME]
    THRESH_WUP = xrVerf[THRESH_WUP_NAME]
    THRESH_WDN = xrVerf[THRESH_WDN_NAME]
    THRESH_UH = xrVerf[THRESH_UH_NAME]
    THRESH_REF = xrVerf[THRESH_REF_NAME]
    THRESH_TOT = xrVerf[THRESH_TOT_NAME]

    tWndGDF = THRESH_WND.to_dataframe(name = "LSR")
    tWupGDF = THRESH_WUP.to_dataframe(name = "LSR")
    tWdnGDF = THRESH_WDN.to_dataframe(name = "LSR")
    tUHGDF = THRESH_UH.to_dataframe(name = "LSR")
    tRefGDF = THRESH_REF.to_dataframe(name = "LSR")
    tTotGDF = THRESH_TOT.to_dataframe(name = "LSR")
    
    # Find coordinates of LSR values >1 (Where things happened)
    wndLSR = tWndGDF.loc[tWndGDF["LSR"] >= 1]
    wupLSR = tWupGDF.loc[tWupGDF["LSR"] >= 1]
    wdnLSR = tWdnGDF.loc[tWdnGDF["LSR"] >= 1]
    uhLSR = tUHGDF.loc[tUHGDF["LSR"] >= 1]
    refLSR = tRefGDF.loc[tRefGDF["LSR"] >= 1]
    totLSR = tTotGDF.loc[tTotGDF["LSR"] >= 1]
    
    tools.loggedPrint.instance().write("Converting threshold pandas dataframes to shapely point instances.")
    # Convert these locations to shapely Point objects
    wndGeom = [Point(x,y) for x, y in zip(wndLSR['Longitude_WRF'], wndLSR['Latitude_WRF'])]
    wupGeom = [Point(x,y) for x, y in zip(wupLSR['Longitude_WRF'], wupLSR['Latitude_WRF'])]
    wdnGeom = [Point(x,y) for x, y in zip(wdnLSR['Longitude_WRF'], wdnLSR['Latitude_WRF'])]
    uhGeom = [Point(x,y) for x, y in zip(uhLSR['Longitude_WRF'], uhLSR['Latitude_WRF'])]
    refGeom = [Point(x,y) for x, y in zip(refLSR['Longitude_WRF'], refLSR['Latitude_WRF'])]
    totGeom = [Point(x,y) for x, y in zip(totLSR['Longitude_WRF'], totLSR['Latitude_WRF'])]
    
    gdf_wndlsr = gpd.GeoDataFrame({'geometry': wndGeom}, crs="epsg:4326")
    gdf_wuplsr = gpd.GeoDataFrame({'geometry': wupGeom}, crs="epsg:4326")
    gdf_wdnlsr = gpd.GeoDataFrame({'geometry': wdnGeom}, crs="epsg:4326")
    gdf_uhlsr = gpd.GeoDataFrame({'geometry': uhGeom}, crs="epsg:4326")
    gdf_reflsr = gpd.GeoDataFrame({'geometry': refGeom}, crs="epsg:4326")
    gdf_totlsr = gpd.GeoDataFrame({'geometry': totGeom}, crs="epsg:4326")
    
    mesh["countWND"] = 0 
    mesh["countWUP"] = 0 
    mesh["countWDN"] = 0 
    mesh["countUH"] = 0 
    mesh["countREF"] = 0 
    mesh["countTOT"] = 0 
    mesh["countSPC"] = 0 
    
    tools.loggedPrint.instance().write("Running spatial joins.")
    # Merge the points to our mesh instance, then compute the spatial join between the mesh and point objects (Counts per grid box)
    #  NOTE: Shape files cannot have column names longer than 10 characters, so we cut our names to 8-9 here
    pointInPolys = gpd.sjoin(gdf_wndlsr, mesh, how='left')
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countWND')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["WND" + lDateTM.strftime("%y%m%d")] = mesh['countWND_y'].fillna(mesh['countWND_x'])
    
    pointInPolys = gpd.sjoin(gdf_wuplsr, mesh, how='left')
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countWUP')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["WUP" + lDateTM.strftime("%y%m%d")] = mesh['countWUP_y'].fillna(mesh['countWUP_x'])  
   
    pointInPolys = gpd.sjoin(gdf_wdnlsr, mesh, how='left')
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countWDN')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["WDN" + lDateTM.strftime("%y%m%d")] = mesh['countWDN_y'].fillna(mesh['countWDN_x'])            
    
    pointInPolys = gpd.sjoin(gdf_uhlsr, mesh, how='left')
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countUH')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["UH" + lDateTM.strftime("%y%m%d")] = mesh['countUH_y'].fillna(mesh['countUH_x'])              
    
    pointInPolys = gpd.sjoin(gdf_reflsr, mesh, how='left')
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countREF')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["REF" + lDateTM.strftime("%y%m%d")] = mesh['countREF_y'].fillna(mesh['countREF_x'])
    
    pointInPolys = gpd.sjoin(gdf_totlsr, mesh, how='left')
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countTOT')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["TOT" + lDateTM.strftime("%y%m%d")] = mesh['countTOT_y'].fillna(mesh['countTOT_x'])  
    
    pointInPolys = gpd.sjoin(gdfSPC, mesh, how='left')                       
    result = pointInPolys.groupby(['meshIndex']).size().reset_index(name='countSPC')
    mesh = mesh.merge(result, on='meshIndex', how='left')
    mesh["SPC" + lDateTM.strftime("%y%m%d")] = mesh['countSPC_y'].fillna(mesh['countSPC_x'])  

    mesh = mesh.drop(['countWND_x', 'countWUP_x', 'countWDN_x', 'countUH_x', 'countREF_x', 'countTOT_x', 
                      'countSPC_x', 'countWND_y', 'countWUP_y', 'countWDN_y', 'countUH_y', 'countREF_y', 
                      'countTOT_y', 'countSPC_y'], axis=1)
                      
    xrVerf.close()
                      
    return mesh

"""
SCRIPT FUNCTIONS (MPI)
"""  
    
def run_verification(command_tuple):
    command_list = command_tuple[0].split(";")
    domain = command_list[0]
    timestep = command_list[1]
    eMode = command_list[2]
    ensembleMode = False
    if eMode == "E":
        tools.loggedPrint.instance().write("runV: " + timestep + ": LOG: Ensemble Mode Active.")
        ensembleMode = True
    
    #day_list = command_list[2][1:-1].replace(" ", "").split(",") #RF: No longer used.
    
    settings = command_tuple[1]
    
    GeoMultiplier = settings["pointsInStates_" + domain].astype(int)
    
    vFileDir = None
    if ensembleMode:
        vFileDir = ENSEMBLE_SAVE_DIR
    else:
        vFileDir = settings["vOutDir"]
    verificationHead = settings["verificationHead"]
    
    TIMESTR = datetime.strptime(timestep, '%Y-%m-%d_%H_%M_%S')
    # Check if we need verification at this time step.
    if(os.path.exists(vFileDir + "/verification_output_" + TIMESTR.strftime("%Y%m%d_%H%M%S") + ".nc")):
        VERF = xr.open_dataset(vFileDir + "/verification_output_" + TIMESTR.strftime("%Y%m%d_%H%M%S") + ".nc")
        try:
            fVer = float(VERF.attrs["FILE_VERSION"])
            if(fVer >= FILE_VERSION):
                tools.loggedPrint.instance().write("runV: " + timestep + ": LOG: Verification output at same or newer file version already present for this timestep, skipping.")
                VERF.close()
                return True
        except KeyError:
            pass
        VERF.close()
    
    PREVDAY = TIMESTR - timedelta(days=1)
    PREV6HR = TIMESTR - timedelta(hours=6)
    NEXTDAY = TIMESTR + timedelta(days=1)
    #tools.loggedPrint.instance().write("runV: " + timestep + ": Start -- " + str(day_list) + " -- " + command_list[2])      
    V_GRIDRAD = None
    GridRad_IsRegular = False
    if TIMESTR in settings["gridrad_severe_timesteps"]:
        fileTest_gridrad = TIMESTR.strftime(verificationHead + "gridrad_filtered_S_%Y%m%d_%H%M%S.nc")
        if os.path.exists(fileTest_gridrad):
            V_GRIDRAD = xr.open_dataset(fileTest_gridrad, engine='netcdf4')
    else:
        fileTest_gridrad = TIMESTR.strftime(verificationHead + "gridrad_filtered_R_%Y%m%d_%H%M%S.nc")
        if os.path.exists(fileTest_gridrad):
            V_GRIDRAD = xr.open_dataset(fileTest_gridrad, engine='netcdf4')
            GridRad_IsRegular = True
        
    V_GRIDRAD_MESH = None
    fileTest_gridrad_mesh = TIMESTR.strftime(verificationHead + "GridRAD_Severe_Mesh_%Y%m%d.nc")
    if os.path.exists(fileTest_gridrad_mesh):
        if(int(TIMESTR.strftime("%H")) < 12):
            if not os.path.exists(PREVDAY.strftime(verificationHead + "GridRAD_Severe_Mesh_%Y%m%d.nc")):
                V_GRIDRAD_MESH = None
            else:
                V_GRIDRAD_MESH = xr.open_dataset(PREVDAY.strftime(verificationHead + "GridRAD_Severe_Mesh_%Y%m%d.nc"), engine='netcdf4') 
        else:
            V_GRIDRAD_MESH = xr.open_dataset(fileTest_gridrad_mesh, engine='netcdf4')        
        
    #V_PRISM = None
    #if(int(TIMESTR.strftime("%H")) == 12 and int(TIMESTR.strftime("%M")) == 0):
    #    fileTest_prism = TIMESTR.strftime(verificationHead + "PRISM_ppt_%Y%m%d.nc")
    #    V_PRISM = xr.open_dataset(fileTest_prism, engine='netcdf4')
        
    V_STAGE4 = None
    if(int(TIMESTR.strftime("%H")) % 6 == 0 and int(TIMESTR.strftime("%M")) == 0):
        fileTest_stage4 = TIMESTR.strftime(verificationHead + "Stage4_%Y%m%d%H.nc")
        V_STAGE4 = xr.open_dataset(fileTest_stage4, engine='netcdf4')        
    
    """    
    V_NARR_SHUM = None
    V_NARR_AIR = None
    # NARR only processes every 3rd hour (00z, 03z, 06z, etc...)
    if(int(TIMESTR.strftime("%H")) % 3 == 0):
        fileTest_narr_shum = TIMESTR.strftime(verificationHead + "shum.%Y%m.nc")
        fileTest_narr_air = TIMESTR.strftime(verificationHead + "air.%Y%m.nc")
        if os.path.exists(fileTest_narr_shum):
            V_NARR_SHUM = xr.open_dataset(fileTest_narr_shum)
        if os.path.exists(fileTest_narr_air):
            V_NARR_AIR = xr.open_dataset(fileTest_narr_air)        
    """       
       
    if (V_GRIDRAD is None) and (V_STAGE4 is None) and (V_GRIDRAD_MESH is None): #and (V_NARR_SHUM is None or V_NARR_AIR is None):
        # No verification data available.
        tools.loggedPrint.instance().write("runV: " + timestep + ": WARN: No verification data available for this timestep, skipping this timestep.")
        return False
        
    wrf_bounds = settings["wrf_bounds_" + str(domain)]
    # Determine the GridRAD regridder / bounds we need
    gridrad_bounds = None
    gridrad_regridder = None
    gridrad_mesh_regridder = None
    if TIMESTR in settings["gridrad_severe_timesteps"]:
        # Severe Bounds
        dictKey = TIMESTR.strftime("%Y%m%d%H%M%S")
        regridderIDX = settings["gridrad_severe_regridder_numbers"][dictKey]
        gridrad_bounds = settings["gridrad_severe" + str(regridderIDX) + "_bounds"]
        gridrad_regridder = xe.Regridder(wrf_bounds, gridrad_bounds, 'conservative', reuse_weights = True, filename=verificationHead + "conservative_GRIDRAD_severe" + str(regridderIDX) + "_" + str(domain) + ".nc")
    else:
        # Regular (Non-Severe) Bounds
        gridrad_bounds = settings["gridrad_bounds"]
        gridrad_regridder = xe.Regridder(wrf_bounds, gridrad_bounds, 'conservative', reuse_weights = True, filename=verificationHead + "conservative_GRIDRAD_" + str(domain) + ".nc")
        
    if gridrad_bounds is None or gridrad_regridder is None:
        tools.loggedPrint.instance().write("runV: " + timestep + ": ERROR: GridRad Regridder failed to load.")
        return False   

    if V_GRIDRAD_MESH is not None:
        if(int(TIMESTR.strftime("%H")) < 12):
            tFileSvr = verificationHead + PREVDAY.strftime("conservative_GridRad_Severe_Mesh_%Y%m%d") + "_" + str(domain) + ".nc"
        else:
            tFileSvr = verificationHead + TIMESTR.strftime("conservative_GridRad_Severe_Mesh_%Y%m%d") + "_" + str(domain) + ".nc"
        bounds = calculation.get_bounds(V_GRIDRAD_MESH, "Latitude", "Longitude")
        gridrad_mesh_regridder = xe.Regridder(wrf_bounds, bounds, 'conservative', reuse_weights = True, filename=tFileSvr)
        
    # Create our output file, reorder the dimensions appropriately
    VERIF_O = xr.Dataset(coords = {
        #"TimeFile": day_list,
        #"Latitude_ERA": (["y_era"], V_ERA_PRES["latitude"].data),
        #"Longitude_ERA": (["x_era"], V_ERA_PRES["longitude"].data),
        "TimeFile": [0],
        "FSS_Scales": FSS_Scales,
        "Latitude_WRF": (["y_wrf", "x_wrf"], wrf_bounds["lat"].data),
        "Longitude_WRF": (["y_wrf", "x_wrf"], wrf_bounds["lon"].data),
    })
    if(V_GRIDRAD is not None):
        VERIF_O.coords["Latitude_GRIDRAD"] = (["y_gridrad"], V_GRIDRAD["Latitude"].data)
        VERIF_O.coords["Longitude_GRIDRAD"] = (["x_gridrad"], V_GRIDRAD["Longitude"].data)
        VERIF_O.attrs["GRIDRAD_IsRegularOnly"] = int(GridRad_IsRegular)
    if(V_STAGE4 is not None):
        VERIF_O.coords["Latitude_STAGE4"] = (["y_stage4", "x_stage4"], V_STAGE4["latitude"].data)
        VERIF_O.coords["Longitude_STAGE4"] = (["y_stage4", "x_stage4"], V_STAGE4["longitude"].data)    
    """
    if(V_NARR_AIR is not None and V_NARR_SHUM is not None):
        VERIF_O.coords["Latitude_NARR"] = (["y_narr", "x_narr"], V_NARR_AIR["lat"].data)
        VERIF_O.coords["Longitude_NARR"] = (["y_narr", "x_narr"], V_NARR_AIR["lon"].data)        
    """
    if(V_GRIDRAD_MESH is not None):
        VERIF_O.coords["Latitude_GMESH"] = (["y_gridradmesh"], V_GRIDRAD_MESH["Latitude"].data)
        VERIF_O.coords["Longitude_GMESH"] = (["x_gridradmesh"], V_GRIDRAD_MESH["Longitude"].data)    
    VERIF_O.attrs["FILE_VERSION"] = FILE_VERSION

    # Get regridders   
    #prism_regridder = xe.Regridder(wrf_bounds, settings['prism_bounds'], 'conservative', reuse_weights = True, filename=verificationHead + "conservative_PRISM_" + str(domain) + ".nc")
    st4_regridder = xe.Regridder(wrf_bounds, settings['stage4_bounds'], 'conservative', reuse_weights = True, filename=verificationHead + "conservative_STAGE4_" + str(domain) + ".nc")
    #narr_regridder = xe.Regridder(wrf_bounds, settings['narr_bounds'], 'conservative', reuse_weights = True, filename=verificationHead + "conservative_NARR_" + str(domain) + ".nc")    

    # Process files
    #for i, day in enumerate(day_list):   
    if ensembleMode:
        tools.loggedPrint.instance().write("runV: " + timestep + ": LOG: Ensemble Mode - Issuing Verification Instructions")
        
        """
        if(int(TIMESTR.strftime("%H")) % 3 == 0 and int(TIMESTR.strftime("%M")) == 0 and (V_NARR_AIR is not None and V_NARR_SHUM is not None)):
            V_T_THE = handle_theta_e_wrfout(VERIF_O, V_NARR_AIR, V_NARR_SHUM, narr_regridder, TIMESTR, domain, None)
            if V_T_THE:
                VERIF_O = V_T_THE
        """

        if(int(TIMESTR.strftime("%M")) == 0):
            VERIF_O = handle_surrogate_severe(VERIF_O, TIMESTR, domain, GeoMultiplier, None)
            
        if((int(TIMESTR.strftime("%H")) % 6 == 0) and int(TIMESTR.strftime("%M")) == 0 and V_STAGE4 is not None):
            V_T_ST4 = handle_stage_4(VERIF_O, V_STAGE4, st4_regridder, TIMESTR, domain, None, None)
            if V_T_ST4:
                VERIF_O = V_T_ST4    

        if(V_GRIDRAD is not None): 
            V_GRIDRAD_R = None
            gridrad_r_regridder = None
            if GridRad_IsRegular == False:
                # Check for a Regular GridRad file at this timestep.
                fileTest_gridrad_r = TIMESTR.strftime(verificationHead + "gridrad_filtered_R_%Y%m%d_%H%M%S.nc")
                if os.path.exists(fileTest_gridrad_r):
                    V_GRIDRAD_R = xr.open_dataset(fileTest_gridrad_r, engine='netcdf4')  
                    gridrad_r_bounds = settings["gridrad_bounds"]
                    gridrad_r_regridder = xe.Regridder(wrf_bounds, gridrad_r_bounds, 'conservative', reuse_weights = True, filename=verificationHead + "conservative_GRIDRAD_" + str(domain) + ".nc")                                      

            V_T_GRIDRAD = handle_gridrad(VERIF_O, V_GRIDRAD, gridrad_regridder, TIMESTR, domain, None, V_GRIDRAD_R, gridrad_r_regridder)
            if V_T_GRIDRAD:
                VERIF_O = V_T_GRIDRAD
                
            if V_GRIDRAD_R is not None:
                V_GRIDRAD_R.close()

        V_T_UHNP = add_UH_NP(VERIF_O, TIMESTR, domain, None)
        if V_T_UHNP:
            VERIF_O = V_T_UHNP
            
        if V_GRIDRAD_MESH is not None:
            V_T_HAILMESH = handle_gridrad_mesh(VERIF_O, V_GRIDRAD_MESH, gridrad_mesh_regridder, TIMESTR, domain, None)        
            if V_T_HAILMESH:
                VERIF_O = V_T_HAILMESH
                
        tools.loggedPrint.instance().write("runV: " + timestep + ": LOG: Ensemble Instructions Completed")
    
    else: 
        SUBHR = None
        AFWA = None #xr.open_dataset(afwaout, engine='netcdf4')
        WRF = None #xr.open_dataset(wrfout, engine='netcdf4')
        CALC = None 
    
        subhrout = settings["outFileDir"] + "subhr_" + domain + "_" + timestep  
        afwaout = settings["outFileDir"] + "AFWA_" + domain + "_" + timestep
        wrfout = settings["outFileDir"] + "wrfout_" + domain + "_" + timestep    
        calcout = settings["outFileDir"] + "calcs_" + domain + "_" + timestep        
        tools.loggedPrint.instance().write("runV: " + timestep + ": Opening Files")
        
        if os.path.exists(subhrout):
            SUBHR = xr.open_dataset(subhrout, engine='netcdf4') 
        if os.path.exists(afwaout):
            AFWA = xr.open_dataset(afwaout, engine='netcdf4') 
        if os.path.exists(wrfout):
            WRF = xr.open_dataset(wrfout, engine='netcdf4') 
        if os.path.exists(calcout):
            CALC = xr.open_dataset(calcout, engine='netcdf4')         
            
        tools.loggedPrint.instance().write("runV: " + timestep + ": LOG: Present Data [SUBHR: " + str(SUBHR is not None) + ", AFWA: " + str(AFWA is not None) + ", WRF: " + str(WRF is not None) + ", CALC: " + str(CALC is not None) + "]")

        """
        # NARR
        # Calculate Theta-E from the input NARR data to compare to the wrfout file.
        if(int(TIMESTR.strftime("%H")) % 3 == 0 and int(TIMESTR.strftime("%M")) == 0 and (V_NARR_AIR is not None and V_NARR_SHUM is not None)):
            if CALC is not None:
                # Fetch the matching timestep from NARR
                V_T_THE = handle_theta_e_calc(VERIF_O, V_NARR_AIR, V_NARR_SHUM, narr_regridder, TIMESTR, CALC)
                if V_T_THE:
                    VERIF_O = V_T
                
            elif WRF is not None:
                V_T_THE = handle_theta_e_wrfout(VERIF_O, V_NARR_AIR, V_NARR_SHUM, narr_regridder, TIMESTR, domain, WRF)
                if V_T_THE:
                    VERIF_O = V_T
        """

        # Process the individual output files
        # WRF Output (~1 hr)
        if WRF is not None:
            VERIF_O = handle_surrogate_severe(VERIF_O, TIMESTR, domain, GeoMultiplier, WRF)
            
            # Stage IV
            if((int(TIMESTR.strftime("%H")) % 6 == 0) and int(TIMESTR.strftime("%M")) == 0 and V_STAGE4 is not None):            
                prev_6hr_file = settings["outFileDir"] + "wrfout_" + domain + "_" + PREV6HR.strftime("%Y-%m-%d_%H_%M_%S")
                if os.path.exists(prev_6hr_file):
                    V_T_ST4 = handle_stage_4(VERIF_O, V_STAGE4, st4_regridder, TIMESTR, domain, prev_6hr_file, WRF)
                    if V_T_ST4:
                        VERIF_O = V_T_ST4
                else:
                    tools.loggedPrint.instance().write("runV: " + timestep + ": WARN: File (" + prev_6hr_file + ") not found, ignoring Stage IV verification for this time step.")    
                    
        # Subhourly files (~15 min)
        if SUBHR is not None:
            if(V_GRIDRAD is not None): 
                V_GRIDRAD_R = None
                gridrad_r_regridder = None
                if GridRad_IsRegular == False:
                    # Check for a Regular GridRad file at this timestep.
                    fileTest_gridrad_r = TIMESTR.strftime(verificationHead + "gridrad_filtered_R_%Y%m%d_%H%M%S.nc")
                    if os.path.exists(fileTest_gridrad_r):
                        V_GRIDRAD_R = xr.open_dataset(fileTest_gridrad_r, engine='netcdf4')  
                        gridrad_r_bounds = settings["gridrad_bounds"]
                        gridrad_r_regridder = xe.Regridder(wrf_bounds, gridrad_r_bounds, 'conservative', reuse_weights = True, filename=verificationHead + "conservative_GRIDRAD_" + str(domain) + ".nc")                                      

                V_T_GRIDRAD = handle_gridrad(VERIF_O, V_GRIDRAD, gridrad_regridder, TIMESTR, domain, SUBHR, V_GRIDRAD_R, gridrad_r_regridder)
                if V_T_GRIDRAD:
                    VERIF_O = V_T_GRIDRAD
                    
                if V_GRIDRAD_R is not None:
                    V_GRIDRAD_R.close()

            V_T_UHNP = add_UH_NP(VERIF_O, TIMESTR, domain, SUBHR)
            if V_T_UHNP:
                VERIF_O = V_T_UHNP
            
        # AFWA Diagnostics (~15 Minutes)
        if AFWA is not None:
            if V_GRIDRAD_MESH is not None:
                V_T_HAILMESH = handle_gridrad_mesh(VERIF_O, V_GRIDRAD_MESH, gridrad_mesh_regridder, TIMESTR, domain, AFWA)        
                if V_T_HAILMESH:
                    VERIF_O = V_T_HAILMESH

        tools.loggedPrint.instance().write("runV: " + timestep + ": Calculation complete.\n")  

        #tools.loggedPrint.instance().write("runV (DEBUG): " + timestep + ": RMSE: " + str(V_Ref0_RMSE) + ", Type: " + str(type(V_Ref0_RMSE)) + ", As .data: " + str(V_Ref0_RMSE.data))

        if SUBHR is not None:
            SUBHR.close()
        if AFWA is not None:
            AFWA.close()
        if WRF is not None:
            WRF.close()
        if CALC is not None:
            CALC.close()  
    
    tools.loggedPrint.instance().write("runV: " + timestep + ": Save Output")
    #tools.loggedPrint.instance().write("runV: " + timestep + ": Output (In File): " + str(VERIF_O["V_Ref0_Diff"]))
    # Write the output
    VERIF_O.to_netcdf(vFileDir + "/verification_output_" + TIMESTR.strftime("%Y%m%d_%H%M%S") + ".nc")
    
    # Clean
    #V_ERA_PRES.close()
    #V_ERA_SLEV.close()
    if V_GRIDRAD is not None:
        V_GRIDRAD.close()
    if V_GRIDRAD_MESH is not None:
        V_GRIDRAD_MESH.close()
    if V_STAGE4 is not None:
        V_STAGE4.close()
    """
    if V_NARR_AIR is not None:
        V_NARR_AIR.close()
    if V_NARR_SHUM is not None:
        V_NARR_SHUM.close()
    """        
    VERIF_O.close()
    return True
    
def run_lsr_verification(timesteps, sDict, ensembleMode = False):
    tools.loggedPrint.instance().write("Starting LSR Verification...")
    # This is the number of points generated, the number of polygons is nX-1 * nY-1
    #  The bigger these numbers, the finer the scale of the grid that is generated
    nX = 150
    nY = 150
    FSS_Scales_LSR = np.arange(1, min(nX, nY))
    # Get a list of days
    lsrDates = []
    for t in timesteps:
        ydm = t.strftime("%y%m%d")
        if not (ydm in lsrDates):
            lsrDates.append(ydm)
    # Create zero numpy arrays to store the totals from WRF.
    wrf_bounds = sDict["wrf_bounds_d01"]
    shape = wrf_bounds["lat"].shape
    # We split by each threshold type and then the net threshold
    daily_lsr_total = np.zeros(shape)
    daily_lsr_WND = np.zeros(shape)
    daily_lsr_WUP = np.zeros(shape)
    daily_lsr_WDN = np.zeros(shape)
    daily_lsr_UH = np.zeros(shape)
    daily_lsr_REF = np.zeros(shape)
    all_lsr_total = np.zeros(shape)
    all_lsr_WND = np.zeros(shape)
    all_lsr_WUP = np.zeros(shape)
    all_lsr_WDN = np.zeros(shape)
    all_lsr_UH = np.zeros(shape)
    all_lsr_REF = np.zeros(shape)  
    # Loop into the verification files
    tools.loggedPrint.instance().write("Scanning generated verification files for LSR data.")
    
    if ensembleMode:
        lsr_totals = {}
        for test in ENSEMBLE_TESTS:
            lsr_totals[test] = {
                'daily_lsr_total': daily_lsr_total,
                'daily_lsr_WND': daily_lsr_WND,
                'daily_lsr_WUP': daily_lsr_WUP,
                'daily_lsr_WDN': daily_lsr_WDN,
                'daily_lsr_UH': daily_lsr_UH,
                'daily_lsr_REF': daily_lsr_UH,
                'all_lsr_total': all_lsr_total,
                'all_lsr_WND': all_lsr_WND,
                'all_lsr_WUP': all_lsr_WUP,
                'all_lsr_WDN': all_lsr_WDN,
                'all_lsr_UH': all_lsr_UH,
                'all_lsr_REF': all_lsr_REF,                
            }
    
        vFiles = sorted(glob.glob(ENSEMBLE_SAVE_DIR + "/verification_output_*.nc"))
        for vF in vFiles:
            timeString = datetime.strptime(vF[-18:-3], "%Y%m%d_%H%M%S")
            # RF Note: For the future, I may have work where wrfout files are not at the 0s, this should read as a file-check for WRF-Out, but for now, this is all that's needed.
            if(int(timeString.strftime("%M")) == 0 and int(timeString.strftime("%S")) == 0):
                VERF = xr.open_dataset(vF, engine='netcdf4')
                LSRS_PROCESSED = False
                try:
                    LSRS_PROCESSED = bool(VERF.attrs["LSRS_PROCESSED"])
                except KeyError:
                    pass            
                if(LSRS_PROCESSED != True):
                    if(int(timeString.strftime("%H")) == 12):  
                        for test in ENSEMBLE_TESTS:
                            # 12Z is denoted as the "end" of a convective day, save the current arrays, then zero them.
                            VERF["Daily_LSR_Total_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['daily_lsr_total'])
                            VERF["Daily_LSR_WND_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['daily_lsr_WND'])
                            VERF["Daily_LSR_WUP_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['daily_lsr_WUP'])
                            VERF["Daily_LSR_WDN_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['daily_lsr_WDN'])
                            VERF["Daily_LSR_UH_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['daily_lsr_UH'])
                            VERF["Daily_LSR_REF_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['daily_lsr_REF'])
                            lsr_totals[test]['daily_lsr_total'] = np.zeros(shape)
                            lsr_totals[test]['daily_lsr_WND'] = np.zeros(shape)
                            lsr_totals[test]['daily_lsr_WUP'] = np.zeros(shape)
                            lsr_totals[test]['daily_lsr_WDN'] = np.zeros(shape)
                            lsr_totals[test]['daily_lsr_UH'] = np.zeros(shape)
                            lsr_totals[test]['daily_lsr_REF'] = np.zeros(shape)
                        VERF.attrs["LSRS_PROCESSED"] = 1
                        # Save the new file.
                        newFile = ENSEMBLE_SAVE_DIR + "/verification_output_" + timeString.strftime("%Y%m%d_%H%M%S") + ".nc.new"
                        VERF.to_netcdf(newFile)
                        # Overwrite the old file.
                        VERF.close()
                        oldFile = newFile[:-4]
                        tools.popen("rm " + oldFile)
                        tools.popen("mv " + newFile + " " + oldFile)                                
                        # Reopen the file for the daily LSR totals needed below.
                        VERF = xr.open_dataset(vF, engine='netcdf4')
                    for test in ENSEMBLE_TESTS:    
                        # Collect the surrogate LSR values, add them to our arrays
                        lsr_totals[test]['daily_lsr_total'] += VERF["Surrogate_LSR_" + test + "_EnsMean"].data
                        lsr_totals[test]['daily_lsr_WND'] += VERF["Surrogate_LSR_WND_" + test + "_EnsMean"].data
                        lsr_totals[test]['daily_lsr_WUP'] += VERF["Surrogate_LSR_WUP_" + test + "_EnsMean"].data
                        lsr_totals[test]['daily_lsr_WDN'] += VERF["Surrogate_LSR_WDN_" + test + "_EnsMean"].data
                        lsr_totals[test]['daily_lsr_UH'] += VERF["Surrogate_LSR_UH_" + test + "_EnsMean"].data
                        lsr_totals[test]['daily_lsr_REF'] += VERF["Surrogate_LSR_REF_" + test + "_EnsMean"].data  
                        lsr_totals[test]['all_lsr_total'] += VERF["Surrogate_LSR_" + test + "_EnsMean"].data
                        lsr_totals[test]['all_lsr_WND'] += VERF["Surrogate_LSR_WND_" + test + "_EnsMean"].data
                        lsr_totals[test]['all_lsr_WUP'] += VERF["Surrogate_LSR_WUP_" + test + "_EnsMean"].data
                        lsr_totals[test]['all_lsr_WDN'] += VERF["Surrogate_LSR_WDN_" + test + "_EnsMean"].data
                        lsr_totals[test]['all_lsr_UH'] += VERF["Surrogate_LSR_UH_" + test + "_EnsMean"].data
                        lsr_totals[test]['all_lsr_REF'] += VERF["Surrogate_LSR_REF_" + test + "_EnsMean"].data      
                VERF.close()
            # If the file is the last one, save the totals
            if vF == vFiles[-1]:
                VERF = xr.open_dataset(vF, engine='netcdf4')
                ALL_LSRS_PROCESSED = False
                try:
                    ALL_LSRS_PROCESSED = bool(VERF.attrs["ALL_LSRS_PROCESSED"])
                except KeyError:
                    pass 
                if(ALL_LSRS_PROCESSED != True):
                    for test in ENSEMBLE_TESTS:
                        VERF["All_LSR_Total_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['all_lsr_total'])
                        VERF["All_LSR_WND_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['all_lsr_WND'])
                        VERF["All_LSR_WUP_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['all_lsr_WUP'])
                        VERF["All_LSR_WDN_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['all_lsr_WDN'])
                        VERF["All_LSR_UH_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['all_lsr_UH'])
                        VERF["All_LSR_REF_" + test] = (['y_wrf', 'x_wrf'], lsr_totals[test]['all_lsr_REF'])
                    VERF.attrs["ALL_LSRS_PROCESSED"] = 1
                    newFile = ENSEMBLE_SAVE_DIR + "/verification_output_" + timeString.strftime("%Y%m%d_%H%M%S") + ".nc.new"
                    VERF.to_netcdf(newFile)
                    VERF.close()
                    oldFile = newFile[:-4]
                    tools.popen("rm " + oldFile)
                    tools.popen("mv " + newFile + " " + oldFile)                   
                VERF.close()                        
    else:
        vFiles = sorted(glob.glob(sDict["vOutDir"] + "/verification_output_*.nc"))
        for vF in vFiles:
            timeString = datetime.strptime(vF[-18:-3], "%Y%m%d_%H%M%S")
            # RF Note: For the future, I may have work where wrfout files are not at the 0s, this should read as a file-check for WRF-Out, but for now, this is all that's needed.
            if(int(timeString.strftime("%M")) == 0 and int(timeString.strftime("%S")) == 0):
                VERF = xr.open_dataset(vF, engine='netcdf4')
                LSRS_PROCESSED = False
                try:
                    LSRS_PROCESSED = bool(VERF.attrs["LSRS_PROCESSED"])
                except KeyError:
                    pass            
                if(LSRS_PROCESSED != True):              
                    if(int(timeString.strftime("%H")) == 12):
                        tools.loggedPrint.instance().write("LSR Processing required for " + timeString.strftime("%Y%m%d - %H%M%S"))
                        # 12Z is denoted as the "end" of a convective day, save the current arrays, then zero them.
                        VERF["Daily_LSR_Total"] = (['y_wrf', 'x_wrf'], daily_lsr_total)
                        VERF["Daily_LSR_WND"] = (['y_wrf', 'x_wrf'], daily_lsr_WND)
                        VERF["Daily_LSR_WUP"] = (['y_wrf', 'x_wrf'], daily_lsr_WUP)
                        VERF["Daily_LSR_WDN"] = (['y_wrf', 'x_wrf'], daily_lsr_WDN)
                        VERF["Daily_LSR_UH"] = (['y_wrf', 'x_wrf'], daily_lsr_UH)
                        VERF["Daily_LSR_REF"] = (['y_wrf', 'x_wrf'], daily_lsr_REF)
                        daily_lsr_total = np.zeros(shape)
                        daily_lsr_WND = np.zeros(shape)
                        daily_lsr_WUP = np.zeros(shape)
                        daily_lsr_WDN = np.zeros(shape)
                        daily_lsr_UH = np.zeros(shape)
                        daily_lsr_REF = np.zeros(shape)
                        VERF.attrs["LSRS_PROCESSED"] = 1
                        # Save the new file.
                        newFile = sDict["vOutDir"] + "/verification_output_" + timeString.strftime("%Y%m%d_%H%M%S") + ".nc.new"
                        VERF.to_netcdf(newFile)
                        # Overwrite the old file.
                        VERF.close()
                        oldFile = newFile[:-4]
                        tools.popen("rm " + oldFile)
                        tools.popen("mv " + newFile + " " + oldFile)                                
                        # Reopen the file for the daily LSR totals needed below.
                        VERF = xr.open_dataset(vF, engine='netcdf4')
                    # Collect the surrogate LSR values, add them to our arrays
                    daily_lsr_total += VERF["Surrogate_LSR"].data
                    daily_lsr_WND += VERF["Surrogate_LSR_WND"].data
                    daily_lsr_WUP += VERF["Surrogate_LSR_WUP"].data
                    daily_lsr_WDN += VERF["Surrogate_LSR_WDN"].data
                    daily_lsr_UH += VERF["Surrogate_LSR_UH"].data
                    daily_lsr_REF += VERF["Surrogate_LSR_REF"].data  
                    all_lsr_total += VERF["Surrogate_LSR"].data
                    all_lsr_WND += VERF["Surrogate_LSR_WND"].data
                    all_lsr_WUP += VERF["Surrogate_LSR_WUP"].data
                    all_lsr_WDN += VERF["Surrogate_LSR_WDN"].data
                    all_lsr_UH += VERF["Surrogate_LSR_UH"].data
                    all_lsr_REF += VERF["Surrogate_LSR_REF"].data      
                VERF.close()
            # If the file is the last one, save the totals
            if vF == vFiles[-1]:
                VERF = xr.open_dataset(vF, engine='netcdf4')
                ALL_LSRS_PROCESSED = False
                try:
                    ALL_LSRS_PROCESSED = bool(VERF.attrs["ALL_LSRS_PROCESSED"])
                except KeyError:
                    pass 
                if(ALL_LSRS_PROCESSED != True):
                    VERF["All_LSR_Total"] = (['y_wrf', 'x_wrf'], all_lsr_total)
                    VERF["All_LSR_WND"] = (['y_wrf', 'x_wrf'], all_lsr_WND)
                    VERF["All_LSR_WUP"] = (['y_wrf', 'x_wrf'], all_lsr_WUP)
                    VERF["All_LSR_WDN"] = (['y_wrf', 'x_wrf'], all_lsr_WDN)
                    VERF["All_LSR_UH"] = (['y_wrf', 'x_wrf'], all_lsr_UH)
                    VERF["All_LSR_REF"] = (['y_wrf', 'x_wrf'], all_lsr_REF)
                    VERF.attrs["ALL_LSRS_PROCESSED"] = 1
                    newFile = sDict["vOutDir"] + "/verification_output_" + timeString.strftime("%Y%m%d_%H%M%S") + ".nc.new"
                    VERF.to_netcdf(newFile)
                    VERF.close()
                    oldFile = newFile[:-4]
                    tools.popen("rm " + oldFile)
                    tools.popen("mv " + newFile + " " + oldFile)                   
                VERF.close()      
    
    tools.loggedPrint.instance().write("Scanning for LSR mesh file.")
    
    if ensembleMode:
        # Each ensemble has its own shapefile due to the 10-character limit on shapefile variable names
        for test in ENSEMBLE_TESTS:
            if not os.path.exists(ENSEMBLE_SAVE_DIR + "/verification_lsr_mesh_" + test + ".shp"):
                tools.loggedPrint.instance().write("LSR Mesh file for test [" + test + "] not found, generating geopandas mesh object.")
                # Now that we have saved the surroage LSR totals, we can process the data.
                #  Start by generating a grid around the wrf domain to store counts
                xmin, ymin, xmax, ymax = (np.nanmin(wrf_bounds["lon"]), np.nanmin(wrf_bounds["lat"]), np.nanmax(wrf_bounds["lon"]), np.nanmax(wrf_bounds["lat"])) #(-135.9, 17, -56.8, 53)

                dX = (xmax - xmin) / nX
                dY = (ymax - ymin) / nY

                cols = list(np.linspace(xmin, xmax + dX, nX))
                rows = list(np.linspace(ymin, ymax + dY, nY))

                X, Y = np.meshgrid(cols, rows)

                gridBoxes = []

                x0 = 0
                for x1 in range(1, len(X)):
                    y0 = 0
                    for y1 in range(1, len(Y)):
                        newPoly = Polygon([[X[x0, y0], Y[x0, y0]], 
                                     [X[x1, y1], Y[x0, y0]], 
                                     [X[x1, y1], Y[x1, y1]], 
                                     [X[x0, y0], Y[x1, y1]]])
                        gridBoxes.append(newPoly)
                        
                        y0 = y1
                    x0 = x1
                # Load the mesh into geopandas 
                tools.loggedPrint.instance().write("Loading the generated mesh into geopandas.")
                mesh = gpd.GeoDataFrame({"geometry":gridBoxes}, crs="epsg:4326")   
                mesh["meshIndex"] = mesh.index
                # Grab the centroid to be saved to our verification file
                centroid_xy = mesh.centroid
                centroid_x = np.asarray([C.x for C in centroid_xy])
                centroid_y = np.asarray([C.y for C in centroid_xy])

                centroid_x = centroid_x.reshape((nX-1, nY-1))
                centroid_y = centroid_y.reshape((nX-1, nY-1))        
                # loop through our LSR dates (Including the processing above and SPC files)    
                for lDate in lsrDates:
                    lDateTM = datetime.strptime(lDate, "%y%m%d")
                    prevDay = lDateTM - timedelta(days=1)
                    tools.loggedPrint.instance().write("Processing LSR Data for " + lDateTM.strftime("%y%m%d") + ".")
                    # Load the data
                    fVerf = lDateTM.strftime(ENSEMBLE_SAVE_DIR + "/verification_output_%Y%m%d_120000.nc")
                    fSPC_Wnd = prevDay.strftime(sDict["verificationHead"] + "/%y%m%d_rpts_wind.csv")
                    fSPC_Hail = prevDay.strftime(sDict["verificationHead"] + "/%y%m%d_rpts_hail.csv")
                    fSPC_Tor = prevDay.strftime(sDict["verificationHead"] + "/%y%m%d_rpts_torn.csv")
                   
                    mesh = handle_lsr_mesh_verification(fVerf, 
                                         mesh, 
                                         ENSEMBLE_SAVE_DIR, 
                                         centroid_x,
                                         centroid_y,
                                         lDateTM,
                                         fSPC_Wnd, 
                                         fSPC_Hail, 
                                         fSPC_Tor,
                                         THRESH_WND_NAME = "Daily_LSR_WND_" + test,
                                         THRESH_WUP_NAME = "Daily_LSR_WUP_" + test,
                                         THRESH_WDN_NAME = "Daily_LSR_WDN_" + test,
                                         THRESH_UH_NAME = "Daily_LSR_UH_" + test,
                                         THRESH_REF_NAME = "Daily_LSR_REF_" + test,
                                         THRESH_TOT_NAME = "Daily_LSR_Total_" + test)
                    
                    tools.loggedPrint.instance().write("Complete.")
                # Save the mesh to a shapefile (Process sub-step).
                tools.loggedPrint.instance().write("Mesh file generation complete, saving.")
                mesh.to_file(ENSEMBLE_SAVE_DIR + "/verification_lsr_mesh_" + test + ".shp")
            # Load up the mesh file
            tools.loggedPrint.instance().write("Loading the LSR Mesh file.")
            mesh = gpd.read_file(ENSEMBLE_SAVE_DIR + "/verification_lsr_mesh_" + test + ".shp")
            for lDate in lsrDates:
                lDateTM = datetime.strptime(lDate, "%y%m%d")   

                stored_file = ENSEMBLE_SAVE_DIR + "/verification_output_" + lDateTM.strftime("%Y%m%d") + "_120000.nc"
                new_file = ENSEMBLE_SAVE_DIR + "/verification_output_" + lDateTM.strftime("%Y%m%d") + "_120000.nc.new"
                
                xrVerf = xr.open_dataset(stored_file, engine='netcdf4')
                # Check if we need to add in the verification mesh coordinates.
                HAS_VMESH_COORDS = False
                try:
                    HAS_VMESH_COORDS = bool(xrVerf.attrs["HAS_VMESH_COORDS"])
                except KeyError:
                    pass 
                if(HAS_VMESH_COORDS != True): 
                    tools.loggedPrint.instance().write("Adding LSR Mesh Coordinates to " + lDateTM.strftime("%y%m%d") + " netCDF file.")
                    centroid_xy = mesh.centroid
                    centroid_x = np.asarray([C.x for C in centroid_xy])
                    centroid_y = np.asarray([C.y for C in centroid_xy])

                    centroid_x = centroid_x.reshape((nX-1, nY-1))
                    centroid_y = centroid_y.reshape((nX-1, nY-1))        
                    xrVerf.coords["Longitude_LSRMESH"] = (['x_lsrmesh', 'y_lsrmesh'], centroid_x)
                    xrVerf.coords["Latitude_LSRMESH"] = (['x_lsrmesh', 'y_lsrmesh'], centroid_y)   
                    xrVerf.attrs["HAS_VMESH_COORDS"] = 1
                    xrVerf.to_netcdf(new_file)   
                    xrVerf.close()

                    tools.popen("rm " + stored_file)
                    tools.popen("mv " + new_file + " " + stored_file)              
                    # Reopen the file for additional analysis
                    xrVerf = xr.open_dataset(stored_file, engine='netcdf4')
                # Check for processing on LSRs
                FSS_LSRS_PROCESSED = False
                try:
                    FSS_LSRS_PROCESSED = bool(xrVerf.attrs["FSS_LSRS_PROCESSED_" + test])
                except KeyError:
                    pass 
                if(FSS_LSRS_PROCESSED != True):  
                    tools.loggedPrint.instance().write("Computing NProb / FSS Scores for Mesh on " + lDateTM.strftime("%y%m%d") + ".")
                    xrVerf.coords["FSS_Scales_LSRVerification"] = (["FSS_Scales_LSR"], FSS_Scales_LSR)
                    # Load in the surrogate LSRs and the SPC LSRs
                    surrogatesWND = mesh["WND" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    surrogatesWUP = mesh["WUP" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    surrogatesWDN = mesh["WDN" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    surrogatesUH = mesh["UH" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    surrogatesREF = mesh["REF" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    surrogatesTOT = mesh["TOT" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    spc = mesh["SPC" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                    # Compute neighborhood probability for surrogate LSRs and the SPC LSRs
                    NProbWND = calculation.NProb(surrogatesWND, 1, 5) # Probability of 1 report within 5 box radius
                    NProbWUP = calculation.NProb(surrogatesWUP, 1, 5) # Probability of 1 report within 5 box radius
                    NProbWDN = calculation.NProb(surrogatesWDN, 1, 5) # Probability of 1 report within 5 box radius
                    NProbUH = calculation.NProb(surrogatesUH, 1, 5) # Probability of 1 report within 5 box radius
                    NProbREF = calculation.NProb(surrogatesREF, 1, 5) # Probability of 1 report within 5 box radius
                    NProbTOT = calculation.NProb(surrogatesTOT, 1, 5) # Probability of 1 report within 5 box radius
                    NProbSPC = calculation.NProb(spc, 1, 5) # Probability of 1 report within 5 box radius
                    # Calculate the FSS
                    fss_wnd = []
                    fss_wup = []
                    fss_wdn = []
                    fss_uh = []
                    fss_ref = []
                    fss_tot = []
                    for f in FSS_Scales_LSR:        
                        fss_wnd.append(calculation.FSS(NProbWND, NProbSPC, 0.05, f))
                        fss_wup.append(calculation.FSS(NProbWUP, NProbSPC, 0.05, f))
                        fss_wdn.append(calculation.FSS(NProbWDN, NProbSPC, 0.05, f))
                        fss_uh.append(calculation.FSS(NProbUH, NProbSPC, 0.05, f))
                        fss_ref.append(calculation.FSS(NProbREF, NProbSPC, 0.05, f))
                        fss_tot.append(calculation.FSS(NProbTOT, NProbSPC, 0.05, f))
                    # Save to the existing file
                    xrVerf["V_SLSRWND_" + test] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesWND)
                    xrVerf["V_SLSRWUP_" + test] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesWUP)
                    xrVerf["V_SLSRWDN_" + test] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesWDN)
                    xrVerf["V_SLSRUH_" + test] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesUH)
                    xrVerf["V_SLSRREF_" + test] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesREF)
                    xrVerf["V_SLSRTOT_" + test] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesTOT)
                    xrVerf["V_LSRSPC_" + test] = (['x_lsrmesh', 'y_lsrmesh'], spc)            
                    xrVerf["V_NProbWND_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbWND)
                    xrVerf["V_NProbWUP_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbWUP)
                    xrVerf["V_NProbWDN_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbWDN)
                    xrVerf["V_NProbUH_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbUH)
                    xrVerf["V_NProbREF_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbREF)
                    xrVerf["V_NProbTOT_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbTOT)
                    xrVerf["V_NProbSPC_" + test] = (['x_lsrmesh', 'y_lsrmesh'], NProbSPC)
                    xrVerf["V_NProbWND_FSS_" + test] = (['FSS_Scales_LSRVerification'], fss_wnd) 
                    xrVerf["V_NProbWUP_FSS_" + test] = (['FSS_Scales_LSRVerification'], fss_wup) 
                    xrVerf["V_NProbWDN_FSS_" + test] = (['FSS_Scales_LSRVerification'], fss_wdn) 
                    xrVerf["V_NProbUH_FSS_" + test] = (['FSS_Scales_LSRVerification'], fss_uh) 
                    xrVerf["V_NProbREF_FSS_" + test] = (['FSS_Scales_LSRVerification'], fss_ref) 
                    xrVerf["V_NProbTOT_FSS_" + test] = (['FSS_Scales_LSRVerification'], fss_tot) 
                    xrVerf.attrs["FSS_LSRS_PROCESSED_" + test] = 1
                    xrVerf.to_netcdf(new_file)
                xrVerf.close()
                # Check if we made a new file.
                if os.path.exists(new_file):
                    tools.popen("rm " + stored_file)
                    tools.popen("mv " + new_file + " " + stored_file)                            
    else:   
        if not os.path.exists(sDict["vOutDir"] + "/verification_lsr_mesh.shp"):
            tools.loggedPrint.instance().write("LSR Mesh file not found, generating geopandas mesh object.")
            # Now that we have saved the surroage LSR totals, we can process the data.
            #  Start by generating a grid around the wrf domain to store counts
            xmin, ymin, xmax, ymax = (np.nanmin(wrf_bounds["lon"]), np.nanmin(wrf_bounds["lat"]), np.nanmax(wrf_bounds["lon"]), np.nanmax(wrf_bounds["lat"])) #(-135.9, 17, -56.8, 53)

            dX = (xmax - xmin) / nX
            dY = (ymax - ymin) / nY

            cols = list(np.linspace(xmin, xmax + dX, nX))
            rows = list(np.linspace(ymin, ymax + dY, nY))

            X, Y = np.meshgrid(cols, rows)

            gridBoxes = []

            x0 = 0
            for x1 in range(1, len(X)):
                y0 = 0
                for y1 in range(1, len(Y)):
                    newPoly = Polygon([[X[x0, y0], Y[x0, y0]], 
                                 [X[x1, y1], Y[x0, y0]], 
                                 [X[x1, y1], Y[x1, y1]], 
                                 [X[x0, y0], Y[x1, y1]]])
                    gridBoxes.append(newPoly)
                    
                    y0 = y1
                x0 = x1
            # Load the mesh into geopandas 
            tools.loggedPrint.instance().write("Loading the generated mesh into geopandas.")
            mesh = gpd.GeoDataFrame({"geometry":gridBoxes}, crs="epsg:4326")   
            mesh["meshIndex"] = mesh.index
            # Grab the centroid to be saved to our verification file
            centroid_xy = mesh.centroid
            centroid_x = np.asarray([C.x for C in centroid_xy])
            centroid_y = np.asarray([C.y for C in centroid_xy])

            centroid_x = centroid_x.reshape((nX-1, nY-1))
            centroid_y = centroid_y.reshape((nX-1, nY-1))        
            # loop through our LSR dates (Including the processing above and SPC files)    
            for lDate in lsrDates:
                lDateTM = datetime.strptime(lDate, "%y%m%d")
                prevDay = lDateTM - timedelta(days=1)
                tools.loggedPrint.instance().write("Processing LSR Data for " + lDateTM.strftime("%y%m%d") + ".")
                # Load the data
                fVerf = lDateTM.strftime(sDict["vOutDir"] + "/verification_output_%Y%m%d_120000.nc")
                fSPC_Wnd = prevDay.strftime(sDict["verificationHead"] + "/%y%m%d_rpts_wind.csv")
                fSPC_Hail = prevDay.strftime(sDict["verificationHead"] + "/%y%m%d_rpts_hail.csv")
                fSPC_Tor = prevDay.strftime(sDict["verificationHead"] + "/%y%m%d_rpts_torn.csv")
               
                mesh = handle_lsr_mesh_verification(fVerf, 
                                     mesh, 
                                     vDir, 
                                     centroid_x,
                                     centroid_y,
                                     lDateTM,
                                     fSPC_Wnd, 
                                     fSPC_Hail, 
                                     fSPC_Tor)
                
                tools.loggedPrint.instance().write("Complete.")
            # Save the mesh to a shapefile (Process sub-step).
            tools.loggedPrint.instance().write("Mesh file generation complete, saving.")
            mesh.to_file(sDict["vOutDir"] + "/verification_lsr_mesh.shp")
        # Load up the mesh file
        tools.loggedPrint.instance().write("Loading the LSR Mesh file.")
        mesh = gpd.read_file(sDict["vOutDir"] + "/verification_lsr_mesh.shp")     
    
        for lDate in lsrDates:
            lDateTM = datetime.strptime(lDate, "%y%m%d")      
            fVerf = lDateTM.strftime(sDict["vOutDir"] + "/verification_output_%Y%m%d_120000.nc")
            xrVerf = xr.open_dataset(fVerf, engine='netcdf4')
            # Check if we need to add in the verification mesh coordinates.
            HAS_VMESH_COORDS = False
            try:
                HAS_VMESH_COORDS = bool(xrVerf.attrs["HAS_VMESH_COORDS"])
            except KeyError:
                pass 
            if(HAS_VMESH_COORDS != True): 
                tools.loggedPrint.instance().write("Adding LSR Mesh Coordinates to " + lDateTM.strftime("%y%m%d") + " netCDF file.")
                centroid_xy = mesh.centroid
                centroid_x = np.asarray([C.x for C in centroid_xy])
                centroid_y = np.asarray([C.y for C in centroid_xy])

                centroid_x = centroid_x.reshape((nX-1, nY-1))
                centroid_y = centroid_y.reshape((nX-1, nY-1))        
                xrVerf.coords["Longitude_LSRMESH"] = (['x_lsrmesh', 'y_lsrmesh'], centroid_x)
                xrVerf.coords["Latitude_LSRMESH"] = (['x_lsrmesh', 'y_lsrmesh'], centroid_y)   
                xrVerf.attrs["HAS_VMESH_COORDS"] = 1
                xrVerf.to_netcdf(sDict["vOutDir"] + "/verification_output_" + lDateTM.strftime("%Y%m%d_") + "120000.nc.new")   
                xrVerf.close()

                tools.popen("rm " + sDict["vOutDir"] + "/verification_output_" + lDateTM.strftime("%Y%m%d_") + "120000.nc")
                tools.popen("mv " + sDict["vOutDir"] + "/verification_output_" + lDateTM.strftime("%Y%m%d_") + "120000.nc.new" + " " + sDict["vOutDir"] + "/verification_output_" + lDateTM.strftime("%Y%m%d_") + "120000.nc")              
                # Reopen the file for additional analysis
                xrVerf = xr.open_dataset(fVerf, engine='netcdf4')
            # Check for processing on LSRs
            FSS_LSRS_PROCESSED = False
            try:
                FSS_LSRS_PROCESSED = bool(xrVerf.attrs["FSS_LSRS_PROCESSED"])
            except KeyError:
                pass 
            if(FSS_LSRS_PROCESSED != True):  
                tools.loggedPrint.instance().write("Computing NProb / FSS Scores for Mesh on " + lDateTM.strftime("%y%m%d") + ".")
                xrVerf.coords["FSS_Scales_LSRVerification"] = (["FSS_Scales_LSR"], FSS_Scales_LSR)
                # Load in the surrogate LSRs and the SPC LSRs
                surrogatesWND = mesh["WND" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                surrogatesWUP = mesh["WUP" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                surrogatesWDN = mesh["WDN" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                surrogatesUH = mesh["UH" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                surrogatesREF = mesh["REF" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                surrogatesTOT = mesh["TOT" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                spc = mesh["SPC" + lDateTM.strftime("%y%m%d")].to_numpy().reshape((nX-1, nY-1))
                # Compute neighborhood probability for surrogate LSRs and the SPC LSRs
                NProbWND = calculation.NProb(surrogatesWND, 1, 5) # Probability of 1 report within 5 box radius
                NProbWUP = calculation.NProb(surrogatesWUP, 1, 5) # Probability of 1 report within 5 box radius
                NProbWDN = calculation.NProb(surrogatesWDN, 1, 5) # Probability of 1 report within 5 box radius
                NProbUH = calculation.NProb(surrogatesUH, 1, 5) # Probability of 1 report within 5 box radius
                NProbREF = calculation.NProb(surrogatesREF, 1, 5) # Probability of 1 report within 5 box radius
                NProbTOT = calculation.NProb(surrogatesTOT, 1, 5) # Probability of 1 report within 5 box radius
                NProbSPC = calculation.NProb(spc, 1, 5) # Probability of 1 report within 5 box radius
                # Calculate the FSS
                fss_wnd = []
                fss_wup = []
                fss_wdn = []
                fss_uh = []
                fss_ref = []
                fss_tot = []
                for f in FSS_Scales_LSR:        
                    fss_wnd.append(calculation.FSS(NProbWND, NProbSPC, 0.05, f))
                    fss_wup.append(calculation.FSS(NProbWUP, NProbSPC, 0.05, f))
                    fss_wdn.append(calculation.FSS(NProbWDN, NProbSPC, 0.05, f))
                    fss_uh.append(calculation.FSS(NProbUH, NProbSPC, 0.05, f))
                    fss_ref.append(calculation.FSS(NProbREF, NProbSPC, 0.05, f))
                    fss_tot.append(calculation.FSS(NProbTOT, NProbSPC, 0.05, f))
                # Save to the existing file
                xrVerf["V_SLSRWND"] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesWND)
                xrVerf["V_SLSRWUP"] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesWUP)
                xrVerf["V_SLSRWDN"] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesWDN)
                xrVerf["V_SLSRUH"] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesUH)
                xrVerf["V_SLSRREF"] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesREF)
                xrVerf["V_SLSRTOT"] = (['x_lsrmesh', 'y_lsrmesh'], surrogatesTOT)
                xrVerf["V_LSRSPC"] = (['x_lsrmesh', 'y_lsrmesh'], spc)            
                xrVerf["V_NProbWND"] = (['x_lsrmesh', 'y_lsrmesh'], NProbWND)
                xrVerf["V_NProbWUP"] = (['x_lsrmesh', 'y_lsrmesh'], NProbWUP)
                xrVerf["V_NProbWDN"] = (['x_lsrmesh', 'y_lsrmesh'], NProbWDN)
                xrVerf["V_NProbUH"] = (['x_lsrmesh', 'y_lsrmesh'], NProbUH)
                xrVerf["V_NProbREF"] = (['x_lsrmesh', 'y_lsrmesh'], NProbREF)
                xrVerf["V_NProbTOT"] = (['x_lsrmesh', 'y_lsrmesh'], NProbTOT)
                xrVerf["V_NProbSPC"] = (['x_lsrmesh', 'y_lsrmesh'], NProbSPC)
                xrVerf["V_NProbWND_FSS"] = (['FSS_Scales_LSRVerification'], fss_wnd) 
                xrVerf["V_NProbWUP_FSS"] = (['FSS_Scales_LSRVerification'], fss_wup) 
                xrVerf["V_NProbWDN_FSS"] = (['FSS_Scales_LSRVerification'], fss_wdn) 
                xrVerf["V_NProbUH_FSS"] = (['FSS_Scales_LSRVerification'], fss_uh) 
                xrVerf["V_NProbREF_FSS"] = (['FSS_Scales_LSRVerification'], fss_ref) 
                xrVerf["V_NProbTOT_FSS"] = (['FSS_Scales_LSRVerification'], fss_tot) 
                xrVerf.attrs["FSS_LSRS_PROCESSED"] = 1
                xrVerf.to_netcdf(sDict["vOutDir"] + "/verification_output_" + lDateTM.strftime("%Y%m%d_") + "120000.nc.new")
            xrVerf.close()
        
    ovFiles = None
    if ensembleMode:
        ovFiles = sorted(glob.glob(ENSEMBLE_SAVE_DIR + "/verification_output_*.nc.new"))
    else: 
        ovFiles = sorted(glob.glob(sDict["vOutDir"] + "/verification_output_*.nc.new"))
    if(len(ovFiles) > 0):
        tools.loggedPrint.instance().write("Overwritting existing netcdf files.")
        for of in ovFiles:
            oldFile = of[:-4]
            tools.popen("rm " + oldFile)
            tools.popen("mv " + of + " " + oldFile)        
        
    tools.loggedPrint.instance().write("LSR Verification Complete, returning to main script.")
    return True
    
def verify_watch_polygons(command_tuple):
    index = command_tuple[0]
    gdf = command_tuple[1]
    settings = command_tuple[2]
    
    tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": Starting analysis.")
    
    vFileDir = settings["vOutDir"]
    verificationHead = settings["verificationHead"]  

    wrf_bounds = settings["wrf_bounds_d01"]
    xlong = wrf_bounds["lon"]
    xlat = wrf_bounds["lat"]
    
    # Setup some constants
    timesteps_sh = []
    subhr_files = sorted(glob.glob(settings["outFileDir"] + "subhr_d01*"))
    for sh in subhr_files:
        t = datetime.strptime(sh[-19:], "%Y-%m-%d_%H_%M_%S")
        timesteps_sh.append(t) 
    timesteps_afwa = []
    afwa_files = sorted(glob.glob(settings["outFileDir"] + "AFWA_d01*"))
    for af in afwa_files:
        t = datetime.strptime(af[-19:], "%Y-%m-%d_%H_%M_%S")
        timesteps_afwa.append(t)   

    timesteps_calcs = []
    calc_files = sorted(glob.glob(settings["outFileDir"] + "calcs_d01*"))
    for cf in calc_files:
        t = datetime.strptime(cf[-19:], "%Y-%m-%d_%H_%M_%S")
        timesteps_calcs.append(t)
    can_use_calcs = True
    for t1 in subhr_files:
        if t1 not in calc_files:
            can_use_calcs = False
            break
    for t2 in afwa_files:
        if t2 not in calc_files:
            can_use_calcs = False
            break
    
    if can_use_calcs == False:
        tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": LOG: Insufficient Calculation Files to bypass NProb calculations, running the calculation job will generate these allowing some time to be saved on these computations.")
    
    geo_obj = gdf[gdf["POLY_INDEX"] == index]
    
    tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": DEBUG: " + str(geo_obj))
    
    # Create a "false" figure to store the contour information in.
    fig = plt.figure(figsize=(12,9))
    ax = plt.axes(projection=ccrs.LambertConformal())    
    
    # Collect a list of subhr files that have data within the time constraint of the polygons
    polyStart = datetime.strptime(str(geo_obj.ISSUED.values[0]), "%Y%m%d%H%M")
    polyEnd = datetime.strptime(str(geo_obj.EXPIRED.values[0]), "%Y%m%d%H%M")
    
    analysis_steps_subhr = []
    for i, t in enumerate(timesteps_sh):
        if (t >= polyStart) and (t <= polyEnd):
            analysis_steps_subhr.append(i)
    analysis_steps_afwa = []
    for i, t in enumerate(timesteps_afwa):
        if (t >= polyStart) and (t <= polyEnd):
            analysis_steps_afwa.append(i) 

    analysis_steps_calcs = []
    if can_use_calcs:      
        for i, t in enumerate(timesteps_calcs):
            if (t >= polyStart) and (t <= polyEnd):
                analysis_steps_calcs.append(i)    
            
    # Create an empty geodataframe for the UH objects
    uh_gdf = gpd.GeoDataFrame(columns = ["geometry", "level-index", "level-value", 
                                         "stroke", "stroke-width", "title", "UHLevel", "threshold"], geometry="geometry", crs="epsg:4326")    
    afwa_tor_gdf = gpd.GeoDataFrame(columns = ["geometry", "level-index", "level-value", 
                                               "stroke", "stroke-width", "title", "threshold"], geometry="geometry", crs="epsg:4326")
    # Turn off matplotlib plotting since we need to perform plot calls to generate the contour polygon objects
    plt.ioff()
    # Create empty arrays to store the highest NP
    NP_UH25 = np.zeros(xlong.shape)
    NP_UH03 = np.zeros(xlong.shape)
    NP_TOR = np.zeros(xlong.shape)    
    # Loop into the subhr files, generate contour objecs for the UH
    if can_use_calcs:
        tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": DEBUG: Analysis Steps for Calcs " + str(analysis_steps_calcs))
        for i in analysis_steps_calcs:
            xr_i = xr.open_dataset(calc_files[i])
            NP_UH25 = np.maximum(xr_i["NP_UH25"], NP_UH25)
            NP_UH03 = np.maximum(xr_i["NP_UH03"], NP_UH03)
            NP_TOR = np.maximum(xr_i["NP_TOR"], NP_TOR)
            xr_i.close()                  
    else:
        tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": DEBUG: Analysis Steps for SubHr " + str(analysis_steps_subhr))
        for i in analysis_steps_subhr:
            xr_i = xr.open_dataset(subhr_files[i])
            NP_UH25 = np.maximum(calculation.NProb(xr_i["UH"][0], 75, 10), NP_UH25)
            NP_UH03 = np.maximum(calculation.NProb(xr_i["UH03"][0], 100, 10), NP_UH03)
            xr_i.close()
        tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": DEBUG: Analysis Steps for AFWA " + str(analysis_steps_afwa))
        for i in analysis_steps_afwa:
            xr_i = xr.open_dataset(afwa_files[i])
            NP_TOR = np.maximum(calculation.NProb(xr_i["AFWA_TORNADO"][0], 29, 10), NP_TOR) ## 29 m/s = 65 mph (EF0+) 
            xr_i.close()

    tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": Generating Contour GeoJSONs")
    if np.nanmax(NP_UH25) > 0.01:
        geo_cntr = plt.contour(xlong, xlat, NP_UH25, [0.01], transform=ccrs.PlateCarree())
        geojson = geojsoncontour.contour_to_geojson(
            contour=geo_cntr,
            min_angle_deg=3.0,
            ndigits=3,
            unit="%"
        )
        # Convert the contour lines to a polygon object
        try:
            pd_json = pd.read_json(geojson)
            gpd_json = gpd.GeoDataFrame.from_features(pd_json["features"])
            gpd_json['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gpd_json.geometry]    
            gpd_json['UHLevel'] = "2_5"
            gpd_json['threshold'] = 75       
            gpd_json.crs = "epsg:4326"
            uh_gdf = gpd.GeoDataFrame(pd.concat([uh_gdf, gpd_json], ignore_index=True), crs=uh_gdf.crs)
        except ValueError:
            pass
        
    if np.nanmax(NP_UH03) > 0.01:        
        geo_cntr = plt.contour(xlong, xlat, NP_UH03, [0.01], transform=ccrs.PlateCarree())
        geojson = geojsoncontour.contour_to_geojson(
            contour=geo_cntr,
            min_angle_deg=3.0,
            ndigits=3,
            unit="%"
        )
        # Convert the contour lines to a polygon object
        try:
            pd_json = pd.read_json(geojson)
            gpd_json = gpd.GeoDataFrame.from_features(pd_json["features"])
            gpd_json['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gpd_json.geometry]    
            gpd_json['UHLevel'] = "0_3"
            gpd_json['threshold'] = 100        
            gpd_json.crs = "epsg:4326"
            uh_gdf = gpd.GeoDataFrame(pd.concat([uh_gdf, gpd_json], ignore_index=True), crs=uh_gdf.crs)
        except ValueError:
            pass            
       
    if np.nanmax(NP_TOR) > 0.01:  
        geo_cntr = plt.contour(xlong, xlat, NP_TOR, [0.01], transform=ccrs.PlateCarree())
        geojson = geojsoncontour.contour_to_geojson(
            contour=geo_cntr,
            min_angle_deg=3.0,
            ndigits=3,
            unit="%"
        )
        # Convert the contour lines to a polygon object
        try:
            pd_json = pd.read_json(geojson)
            gpd_json = gpd.GeoDataFrame.from_features(pd_json["features"])
            gpd_json['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gpd_json.geometry]    
            gpd_json['threshold'] = 29   
            gpd_json.crs = "epsg:4326"
            afwa_tor_gdf = gpd.GeoDataFrame(pd.concat([afwa_tor_gdf, gpd_json], ignore_index=True), crs=afwa_tor_gdf.crs)  
        except ValueError:
            pass            
    
    tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": Computing Areas")
    # Now that we have the contour objects, we can perform spatial analysis between the instances.
    # Start by computing the area for the polygon we are analyzing, convert to equal area projection along the way
    
    unwarned_area_uh = 0
    unwarned_area_tor = 0
    warned_area_uh = 0
    warned_area_tor = 0
    warned_other_area_uh = 0
    warned_other_area_tor = 0
    
    other_polys = gdf[gdf["POLY_INDEX"] != index]
    other_polys_s2 = other_polys[gdf.ISSUED >= int(polyStart.strftime("%Y%m%d%H%M"))]
    other_polys_s3 = other_polys_s2[gdf.ISSUED < int(polyEnd.strftime("%Y%m%d%H%M"))]
    other_windows = other_polys_s3["POLY_INDEX"].to_numpy()    
    
    if len(uh_gdf.index) > 0:
        total_area_uh = uh_gdf.to_crs('epsg:3395').area.sum() / 10**6
        overlap_uh = gpd.overlay(gdf, uh_gdf, how='intersection')
        overlap_uh["area"] = overlap_uh.to_crs('epsg:3395').area
        warned_area_uh = overlap_uh[overlap_uh["POLY_INDEX"] == index]["area"].sum() / 10**6
        warned_other_area_uh = overlap_uh[overlap_uh["POLY_INDEX"].isin(other_windows)]["area"].sum() / 10**6
        unwarned_area_uh = total_area_uh - (warned_area_uh + warned_other_area_uh)
    if len(afwa_tor_gdf.index) > 0:       
        total_area_tor = afwa_tor_gdf.to_crs('epsg:3395').area.sum() / 10**6   
        overlap_tor = gpd.overlay(gdf, afwa_tor_gdf, how='intersection')
        overlap_tor["area"] = overlap_tor.to_crs('epsg:3395').area    
        warned_area_tor = overlap_tor[overlap_tor["POLY_INDEX"] == index]["area"].sum() / 10**6    
        warned_other_area_tor = overlap_tor[overlap_tor["POLY_INDEX"].isin(other_windows)]["area"].sum() / 10**6    
        unwarned_area_tor = total_area_tor - (warned_area_tor + warned_other_area_tor)
    # Save this back on the original object   
    #  RF NOTE: This route is not thread-safe, use the callback method instead.
    #gdf.at[gdf["POLY_INDEX"] == index, "UHInside"] = warned_area_uh
    #gdf.at[gdf["POLY_INDEX"] == index, "UHOtherPoly"] = warned_other_area_uh
    #gdf.at[gdf["POLY_INDEX"] == index, "UHOutside"] = unwarned_area_uh
    #gdf.at[gdf["POLY_INDEX"] == index, "TORInside"] = warned_area_tor
    #gdf.at[gdf["POLY_INDEX"] == index, "TOROtherPoly"] = warned_other_area_tor
    #gdf.at[gdf["POLY_INDEX"] == index, "TOROutside"] = unwarned_area_tor
    tools.loggedPrint.instance().write("runV_geo: " + str(index) + ": Analysis complete")
    return (index, warned_area_uh, warned_other_area_uh, unwarned_area_uh, warned_area_tor, warned_other_area_tor, unwarned_area_tor)

def verify_watch_polygons_ensemble(command_tuple):
    index = command_tuple[0]
    gdf = command_tuple[1]
    settings = command_tuple[2]
    test = command_tuple[3]
    
    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": Starting analysis.")
    
    vFileDir = ENSEMBLE_SAVE_DIR
    verificationHead = settings["verificationHead"]  

    wrf_bounds = settings["wrf_bounds_d01"]
    xlong = wrf_bounds["lon"]
    xlat = wrf_bounds["lat"]    
    
    # Setup some constants
    # The timestep analysis will be identical across all ensemble members, so only the control member is "peeked" for the timesteps.
    timesteps_sh = []
    subhr_files = sorted(glob.glob(ENSEMBLE['Control'] + "/output/subhr_d01*"))
    for sh in subhr_files:
        t = datetime.strptime(sh[-19:], "%Y-%m-%d_%H_%M_%S")
        timesteps_sh.append(t) 
    timesteps_afwa = []
    afwa_files = sorted(glob.glob(ENSEMBLE['Control'] + "/output/AFWA_d01*"))
    for af in afwa_files:
        t = datetime.strptime(af[-19:], "%Y-%m-%d_%H_%M_%S")
        timesteps_afwa.append(t)   
    
    geo_obj = gdf[gdf["POLY_INDEX"] == index]
    
    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": DEBUG: " + str(geo_obj))
    
    # Create a "false" figure to store the contour information in.
    fig = plt.figure(figsize=(12,9))
    ax = plt.axes(projection=ccrs.LambertConformal())    
    
    # Collect a list of subhr files that have data within the time constraint of the polygons
    polyStart = datetime.strptime(str(geo_obj.ISSUED.values[0]), "%Y%m%d%H%M")
    polyEnd = datetime.strptime(str(geo_obj.EXPIRED.values[0]), "%Y%m%d%H%M")
    
    analysis_steps_subhr = []
    for i, t in enumerate(timesteps_sh):
        if (t >= polyStart) and (t <= polyEnd):
            analysis_steps_subhr.append(i)
    analysis_steps_afwa = []
    for i, t in enumerate(timesteps_afwa):
        if (t >= polyStart) and (t <= polyEnd):
            analysis_steps_afwa.append(i)    
            
    # Create an empty geodataframe for the UH objects
    uh_gdf = gpd.GeoDataFrame(columns = ["geometry", "level-index", "level-value", 
                                         "stroke", "stroke-width", "title", "UHLevel", "threshold"], geometry="geometry", crs="epsg:4326")    
    afwa_tor_gdf = gpd.GeoDataFrame(columns = ["geometry", "level-index", "level-value", 
                                               "stroke", "stroke-width", "title", "threshold"], geometry="geometry", crs="epsg:4326")
    # Turn off matplotlib plotting since we need to perform plot calls to generate the contour polygon objects
    plt.ioff()
    # Create empty arrays to store the highest NP
    NP_UH25 = np.zeros(xlong.shape)
    NP_UH03 = np.zeros(xlong.shape)
    NP_TOR = np.zeros(xlong.shape)    
    # Loop into the subhr files, generate contour objecs for the UH

    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": DEBUG: Analysis Steps for SubHr " + str(analysis_steps_subhr))
    for i in analysis_steps_subhr:
        array_UH25 = [] #np.zeros(xlong.shape)
        array_UH03 = [] #np.zeros(xlong.shape)
        ensTest = ENSEMBLE_TESTS[test]
        for iTest in ensTest:
            # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
            testItem = iTest.split("_")
            testType = testItem[0]
            testNumber = int(testItem[1])
            # Fetch the associated object
            eObj = ENSEMBLE[testType][testNumber] 
            for idx in eObj:
                dDir = eObj[idx]
                subhr_files = sorted(glob.glob(dDir + "/output/subhr_d01*"))
                xr_i = xr.open_dataset(subhr_files[i])                
                #tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": DEBUG [" + test + " => " + iTest + "]: MAX (2-5): " + str(np.nanmax(xr_i["UH"][0].data)) + ", (0-3): " + str(np.nanmax(xr_i["UH03"][0].data)))
                array_UH25.append(xr_i["UH"][0])
                array_UH03.append(xr_i["UH03"][0])

                xr_i.close()
        eMeanUH25 = np.array(array_UH25).mean(axis=0)
        eMeanUH03 = np.array(array_UH03).mean(axis=0)
        NP_UH25 = np.maximum(calculation.NProb(eMeanUH25, 75, 10), NP_UH25)
        NP_UH03 = np.maximum(calculation.NProb(eMeanUH03, 100, 10), NP_UH03)
        
    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": DEBUG: Analysis Steps for AFWA " + str(analysis_steps_afwa))
    for i in analysis_steps_afwa:
        array_TOR = []
        ensTest = ENSEMBLE_TESTS[test]
        for iTest in ensTest:
            # Individual tests read as: "SPPT_1", split the object based on the underscore to identify the object and the iterator (IE: SPPT, 1)
            testItem = iTest.split("_")
            testType = testItem[0]
            testNumber = int(testItem[1])
            # Fetch the associated object
            eObj = ENSEMBLE[testType][testNumber] 
            for idx in eObj:
                dDir = eObj[idx]
                subhr_files = sorted(glob.glob(dDir + "/output/AFWA_d01*"))
                xr_i = xr.open_dataset(subhr_files[i])
                array_TOR.append(xr_i["AFWA_TORNADO"][0])
                xr_i.close()
        eMeanTOR = np.array(array_TOR).mean(axis=0)        
        NP_TOR = np.maximum(calculation.NProb(eMeanTOR, 29, 10), NP_TOR) ## 29 m/s = 65 mph (EF0+) 

    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": Generating Contour GeoJSONs")
    if np.nanmax(NP_UH25) > 0.01:
        geo_cntr = plt.contour(xlong, xlat, NP_UH25, [0.01], transform=ccrs.PlateCarree())
        geojson = geojsoncontour.contour_to_geojson(
            contour=geo_cntr,
            min_angle_deg=3.0,
            ndigits=3,
            unit="%"
        )
        # Convert the contour lines to a polygon object
        try:
            pd_json = pd.read_json(geojson)
            gpd_json = gpd.GeoDataFrame.from_features(pd_json["features"])
            gpd_json['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gpd_json.geometry]    
            gpd_json['UHLevel'] = "2_5"
            gpd_json['threshold'] = 75       
            gpd_json.crs = "epsg:4326"
            #uh_gdf = uh_gdf.append(gpd_json, ignore_index=True) #RF: append removed in geopandas 2.0
            uh_gdf = gpd.GeoDataFrame(pd.concat([uh_gdf, gpd_json], ignore_index=True), crs=uh_gdf.crs)
        except ValueError:
            pass
        
    if np.nanmax(NP_UH03) > 0.01:        
        geo_cntr = plt.contour(xlong, xlat, NP_UH03, [0.01], transform=ccrs.PlateCarree())
        geojson = geojsoncontour.contour_to_geojson(
            contour=geo_cntr,
            min_angle_deg=3.0,
            ndigits=3,
            unit="%"
        )
        # Convert the contour lines to a polygon object
        try:
            pd_json = pd.read_json(geojson)
            gpd_json = gpd.GeoDataFrame.from_features(pd_json["features"])
            gpd_json['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gpd_json.geometry]    
            gpd_json['UHLevel'] = "0_3"
            gpd_json['threshold'] = 100        
            gpd_json.crs = "epsg:4326"
            uh_gdf = gpd.GeoDataFrame(pd.concat([uh_gdf, gpd_json], ignore_index=True), crs=uh_gdf.crs)
        except ValueError:
            pass            
       
    if np.nanmax(NP_TOR) > 0.01:  
        geo_cntr = plt.contour(xlong, xlat, NP_TOR, [0.01], transform=ccrs.PlateCarree())
        geojson = geojsoncontour.contour_to_geojson(
            contour=geo_cntr,
            min_angle_deg=3.0,
            ndigits=3,
            unit="%"
        )
        # Convert the contour lines to a polygon object
        try:
            pd_json = pd.read_json(geojson)
            gpd_json = gpd.GeoDataFrame.from_features(pd_json["features"])
            gpd_json['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gpd_json.geometry]    
            gpd_json['threshold'] = 29   
            gpd_json.crs = "epsg:4326"  
            afwa_tor_gdf = gpd.GeoDataFrame(pd.concat([afwa_tor_gdf, gpd_json], ignore_index=True), crs=afwa_tor_gdf.crs)           
        except ValueError:
            pass            
    
    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": Computing Areas")
    # Now that we have the contour objects, we can perform spatial analysis between the instances.
    # Start by computing the area for the polygon we are analyzing, convert to equal area projection along the way
    
    unwarned_area_uh = 0
    unwarned_area_tor = 0
    warned_area_uh = 0
    warned_area_tor = 0
    warned_other_area_uh = 0
    warned_other_area_tor = 0
    
    other_polys = gdf[gdf["POLY_INDEX"] != index]
    other_polys_s2 = other_polys[gdf.ISSUED >= int(polyStart.strftime("%Y%m%d%H%M"))]
    other_polys_s3 = other_polys_s2[gdf.ISSUED < int(polyEnd.strftime("%Y%m%d%H%M"))]
    other_windows = other_polys_s3["POLY_INDEX"].to_numpy()    
    
    if len(uh_gdf.index) > 0:
        total_area_uh = uh_gdf.to_crs('epsg:3395').area.sum() / 10**6
        overlap_uh = gpd.overlay(gdf, uh_gdf, how='intersection')
        overlap_uh["area"] = overlap_uh.to_crs('epsg:3395').area
        warned_area_uh = overlap_uh[overlap_uh["POLY_INDEX"] == index]["area"].sum() / 10**6
        warned_other_area_uh = overlap_uh[overlap_uh["POLY_INDEX"].isin(other_windows)]["area"].sum() / 10**6
        unwarned_area_uh = total_area_uh - (warned_area_uh + warned_other_area_uh)
    if len(afwa_tor_gdf.index) > 0:       
        total_area_tor = afwa_tor_gdf.to_crs('epsg:3395').area.sum() / 10**6   
        overlap_tor = gpd.overlay(gdf, afwa_tor_gdf, how='intersection')
        overlap_tor["area"] = overlap_tor.to_crs('epsg:3395').area    
        warned_area_tor = overlap_tor[overlap_tor["POLY_INDEX"] == index]["area"].sum() / 10**6    
        warned_other_area_tor = overlap_tor[overlap_tor["POLY_INDEX"].isin(other_windows)]["area"].sum() / 10**6    
        unwarned_area_tor = total_area_tor - (warned_area_tor + warned_other_area_tor)
    tools.loggedPrint.instance().write("verify_watch_polygons_ensemble: " + str(index) + ": Analysis complete")
    return (index, test, warned_area_uh, warned_other_area_uh, unwarned_area_uh, warned_area_tor, warned_other_area_tor, unwarned_area_tor)
    
"""
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
            Main Method (Script Run Point)
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
"""

def run_script(ensembleMode, tDir, vHeadDir, inGeoEM = ''): 
    # Spin up our MPI communicator.
    try:
        pool = mpipool.MPIPool()
    except ValueError:
        tools.loggedPrint.instance().write("This script must be run through mpirun/mpiexec")
        sys.exit()
    # Make sure only we run map() on the master process
    if not pool.is_master():
        pool.wait()
        sys.exit(0)  
    tools.loggedPrint.instance().write("Initialize Script.")
    aSet = settings.Settings()
    sDict = aSet.get_full_dict()
    
    if ensembleMode:
        tools.loggedPrint.instance().write("Ensemble Analysis Mode Active.")
        
        cDir = ENSEMBLE['Control']
        domainList = sDict['domains']
        sDict["verificationHead"] = vHeadDir
        
        if inGeoEM:
            tools.loggedPrint.instance().write("Geography Information Directory Set as: " + inGeoEM + ", checking for validity...")
            geoFile = sDict["geo_file"]
            for domain in sDict["domains"]:
                sDict["wrf_bounds_" + domain] = None
            
                tF = inGeoEM + "/" + geoFile
                tF = tF.replace("[domain]", domain)
                if not os.path.exists(tF):
                    tools.loggedPrint.instance().write("ERROR: Geography information file [" + tF + "] was not found, aborting script.")
                    pool.close()
                    sys.exit(1)
                    
                tools.loggedPrint.instance().write("Geography Information File [" + tF + "] found, pulling geography information.")
                geoNC = xr.open_dataset(tF, engine='netcdf4')
                try:
                    wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                    sDict["wrf_bounds_" + str(domain)] = wrf_bounds
                except KeyError:
                    pool.close()
                    raise ValueError("XLAT_M / XLONG_M not found for " + domain)      
                geoNC.close()                       
        else:  
            tools.loggedPrint.instance().write("Detecting geography information.")            
            
            if((int)(sDict["geo_seperate"]) == 1):
                tools.loggedPrint.instance().write("Pulling geography information from geo_em file.")
                geoFile = sDict["geo_file"]
                for domain in sDict["domains"]:
                    sDict["wrf_bounds_" + domain] = None
                
                    tF = cDir + "/output/" + geoFile
                    tF = tF.replace("[domain]", domain)
                    if not os.path.exists(tF):
                        tools.loggedPrint.instance().write("ERROR: Geography information file [" + tF + "] was not found, aborting script.")
                        pool.close()
                        sys.exit(1)
                    
                    geoNC = xr.open_dataset(tF, engine='netcdf4')
                    try:
                        wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                        sDict["wrf_bounds_" + str(domain)] = wrf_bounds
                    except KeyError:
                        pool.close()
                        raise ValueError("XLAT_M / XLONG_M not found for " + domain)      
                    geoNC.close()
            else:
                tools.loggedPrint.instance().write("Pulling geography information from wrfout file.")
                for domain in sDict["domains"]:
                    sDict["wrf_bounds_" + domain] = None
                
                    tFiles = sorted(glob.glob(cDir + "/output/wrfout_" + str(domain) + "*"))
                    geoNC = xr.open_dataset(tFiles[0], engine='netcdf4')
                    try:
                        wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                        sDict["wrf_bounds_" + str(domain)] = wrf_bounds
                    except KeyError:
                        pool.close()
                        raise ValueError("XLAT_M / XLONG_M not found for " + domain)    
                    geoNC.close() 

        # Create the numpy array for geopandas verification
        states = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lakes',
                                                    scale='110m',
                                                    facecolor='none')
        state_poly = gpd.GeoSeries(states.geometries())
        gdf_stpl = gpd.GeoDataFrame(geometry=state_poly)
        for domain in domainList:
            lsr_p = gpd.GeoSeries(map(Point, zip(np.ravel(sDict["wrf_bounds_" + domain]["lon"]), np.ravel(sDict["wrf_bounds_" + domain]["lat"]))))           
            gdf_lsrp = gpd.GeoDataFrame(geometry=lsr_p)        
            gdf_lsrp["inUS"] = False    
            points_within = gpd.sjoin(gdf_lsrp, gdf_stpl, op='within')
            gdf_lsrp.loc[points_within.index, "inUS"] = True
            npInUS = gdf_lsrp["inUS"].to_numpy()
            npInUS = np.reshape(npInUS, sDict["wrf_bounds_" + domain]["lon"].shape) 
            sDict["pointsInStates_" + domain] = npInUS
        # Scan for timesteps to process
        timesteps = []
        sFile = "subhr_[domain]_[time]"
        oFList = []
        tools.loggedPrint.instance().write("Scanning for output files.")
        for domain in domainList:
            sFile = sFile.replace("[domain]", domain)
            sFile = sFile.replace("[time]", "*")
            path = cDir + "/output/" + sFile
            fList_sorted = sorted(glob.glob(path))
            oFList.extend(fList_sorted)
        tools.loggedPrint.instance().write("Found " + str(len(fList_sorted)) + " files.")
        tools.loggedPrint.instance().write("Scanning for timesteps.")
        # Scan for unique timesteps
        for f in oFList:
            tStep = datetime.strptime(f[-19:], '%Y-%m-%d_%H_%M_%S')
            if not tStep in timesteps:
                timesteps.append(tStep)
        tools.loggedPrint.instance().write("There are " + str(len(timesteps)) + " unique timesteps to analyze.")
        # Ensure we have verification data for all the timesteps.
        instructions = []
        tools.loggedPrint.instance().write("Checking the timesteps to ensure we have verification data for each.")
        tsVerify = timesteps
        #era_bounds = None
        gridrad_bounds = None
        #prism_bounds = None
        stage4_bounds = None
        narr_bounds = None
        #
        gridrad_no_data = []
        if os.path.exists(vHeadDir + "gridrad_nonexisting_files.txt"):
            tools.loggedPrint.instance().write(" -> Found gridrad_nonexisting_files.txt, scanning...")
            with open(vHeadDir + "/gridrad_nonexisting_files.txt", "r") as grid_read:
                lineCount = 0
                for line in grid_read:
                    if(len(line) > 0):
                        gridrad_no_data.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
                        lineCount += 1
            tools.loggedPrint.instance().write(" -> " + str(lineCount) + " timesteps have been identified as non-data bearing")       

        if not os.path.exists(vHeadDir + "/gridrad_files.txt"):
            tools.loggedPrint.instance().write(" -> Missing " + vHeadDir + "/gridrad_files.txt. Please re-run the gridrad processing script to generate this file... exiting.")
            pool.close()
            sys.exit(1)
        gridrad_regular_timesteps = []
        gridrad_severe_timesteps = []
        with open(vHeadDir + "/gridrad_files.txt", "r") as grid_read:
            SEVERE_MODE = False
            for line in grid_read:
                if("SEVERE" in line):
                    SEVERE_MODE = True
                elif("REGULAR" in line):
                    SEVERE_MODE = False
                else:
                    if(len(line) > 1):
                        if(SEVERE_MODE):
                            gridrad_severe_timesteps.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
                        else:
                            gridrad_regular_timesteps.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
        
        for ts in tsVerify:           
            if not ts in gridrad_no_data:
                if ts in gridrad_regular_timesteps:
                    fileTest_gridrad = ts.strftime(vHeadDir + "gridrad_filtered_R_%Y%m%d_%H%M%S.nc")
                else:
                    fileTest_gridrad = ts.strftime(vHeadDir + "gridrad_filtered_S_%Y%m%d_%H%M%S.nc")
                if not os.path.exists(fileTest_gridrad):
                    tools.loggedPrint.instance().write("ERROR: Missing required verification data [" + fileTest_gridrad + "]")
                    pool.close()
                    sys.exit(1)
                if gridrad_bounds == None:
                    #tools.loggedPrint.instance().write("gridrad_bounds is still None. Checking timestep: " + str(ts))
                    if ts in gridrad_regular_timesteps:
                        #tools.loggedPrint.instance().write("Timstep is a regular file.")
                        grid_out = xr.open_dataset(fileTest_gridrad, engine='netcdf4')
                        gridrad_bounds = calculation.get_bounds(grid_out, 'Latitude', 'Longitude')
                        grid_out.close()
                        
            if ((int(ts.strftime("%H")) % 6 == 0) and (int(ts.strftime("%M")) == 0)):
                fileTest_stage4 = ts.strftime(vHeadDir + "Stage4_%Y%m%d%H.nc")
                if not os.path.exists(fileTest_stage4):
                    tools.loggedPrint.instance().write("ERROR: Missing required verification data [" + fileTest_stage4 + "]")
                    pool.close()
                    sys.exit(1)
                if stage4_bounds == None:
                    stage4_out = xr.open_dataset(fileTest_stage4, engine='netcdf4')
                    stage4_bounds = calculation.get_bounds(stage4_out, 'latitude', 'longitude')
                    stage4_out.close()        
               
            """
            fileTest_narr = ts.strftime(vHeadDir + "air.%Y%m.nc")
            if not os.path.exists(fileTest_narr):
                tools.loggedPrint.instance().write("ERROR: Missing required verification data [" + fileTest_narr + "]")
                pool.close()
                sys.exit(1)
            if narr_bounds == None:
                narr_out = xr.open_dataset(fileTest_narr, engine='netcdf4')
                narr_bounds = calculation.get_bounds(narr_out, 'lat', 'lon')
                narr_out.close()            
            """
             
            for domain in domainList:
                instructions.append(domain + ";" + ts.strftime("%Y-%m-%d_%H_%M_%S") + ";E")

        sDict["gridrad_bounds"] = gridrad_bounds
        sDict["stage4_bounds"] = stage4_bounds
        #sDict["narr_bounds"] = narr_bounds
        sDict["gridrad_regular_timesteps"] = gridrad_regular_timesteps
        sDict["gridrad_severe_timesteps"] = gridrad_severe_timesteps
        
        # Collect the bounds from GridRAD severe timesteps
        gridrad_severe_regridder_numbers = {}
        if os.path.exists(vHeadDir + "gridrad_severe_regridders.txt"):
            tools.loggedPrint.instance().write(" -> Found gridrad_severe_regridders.txt, scanning to ensure file validity")          
            with open(vHeadDir + "gridrad_severe_regridders.txt", "r") as grid_read:
                for line in grid_read:
                    line_splits = line.strip().split(";")
                    timestring = line_splits[0]
                    regridder_number = line_splits[1]
                    gridrad_severe_regridder_numbers[timestring] = regridder_number
                    # For now, we just need to verify the file exists.
                    if not os.path.exists(vHeadDir + "conservative_GRIDRAD_severe" + regridder_number + "_" + str(domain) + ".nc"):
                        tools.loggedPrint.instance().write("Error: Missing regridder conservative_GRIDRAD_severe" + regridder_number + "_" + str(domain) + ".nc, please re-run the post_processing script.")
                        pool.close()
                        sys.exit(1)
                    else:                   
                        if not "gridrad_severe" + regridder_number + "_bounds" in sDict:
                            tStep = datetime.strptime(timestring, "%Y%m%d%H%M%S")
                            fileTest_gridrad = tStep.strftime(vHeadDir + "gridrad_filtered_S_%Y%m%d_%H%M%S.nc")
                            grid_out = xr.open_dataset(fileTest_gridrad, engine='netcdf4')
                            gridrad_s_bounds = calculation.get_bounds(grid_out, 'Latitude', 'Longitude')
                            grid_out.close() 
                            sDict["gridrad_severe" + regridder_number + "_bounds"] = gridrad_s_bounds
        
        sDict["gridrad_severe_regridder_numbers"] = gridrad_severe_regridder_numbers
        # Construct our iterable.
        commands = []
        for instruction in instructions:
            commands.append([instruction, sDict])
        # tools.loggedPrint.instance().write("Command List:\n" + str(commands) + "\n\n")
        # Run the verification process.
        tools.loggedPrint.instance().write("Mapping WRF Output Verification to MPI.")
        pool.map(run_verification, commands)
        tools.loggedPrint.instance().write("WRF Output Verification Complete...")
        
        tools.loggedPrint.instance().write("Running Ensemble LSR Verification.")
        run_lsr_verification(timesteps, sDict, ensembleMode = True)
        tools.loggedPrint.instance().write("LSR Verification Complete...")
        
        tools.loggedPrint.instance().write("Constructing Ensemble GeoPandas verification for NWS Polygon Objects.")
        
        gdf_list = []
        df_files = []
        for t in timesteps:
            df_file = t.strftime("wwa_%Y01010000_%Y12312359.shp")
            if not df_file in df_files:
                df_files.append(df_file)
        # Open all the required shapefiles and slice out the polygons we need.
        for df_file in df_files:
            df = gpd.read_file(vHeadDir + "/" + df_file)
            df["ISSUED"] = pd.to_numeric(df["ISSUED"])
            df["EXPIRED"] = pd.to_numeric(df["EXPIRED"])
            df_slice1 = df[df.ISSUED > int(timesteps[0].strftime("%Y%m%d%H%M"))]
            df_slice2 = df_slice1[df_slice1.ISSUED < int(timesteps[-1].strftime("%Y%m%d%H%M"))]        
            df_slice3 = df_slice2[(df_slice2.SIG == "A")]
            df_slice4 = df_slice3[(df_slice3.PHENOM == "TO") | (df_slice3.PHENOM == "SV")]
            gdf_list.append(df_slice4)        
        

        
        # The Watch Box verification step is separate since we need to iterate over a geodataframe object.
        gdf = gpd.GeoDataFrame(columns=['WFO', 'ISSUED', 'EXPIRED', 'INIT_ISS', 'INIT_EXP', 'PHENOM', 'GTYPE',
                                              'SIG', 'ETN', 'STATUS', 'NWS_UGC', 'AREA_KM2', 'UPDATED', 'HVTEC_NWSLI',
                                              'HVTEC_SEVERITY', 'HVTEC_CAUSE', 'HVTEC_RECORD', 'IS_EMERGENCY', 'POLYBEGIN',
                                              'POLYEND', 'WINDTAG', 'HAILTAG', 'TORNADOTAG', 'DAMAGETAG', 'geometry'], geometry='geometry')
        gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        # Dissolve polygons into watch boxes (Reduces to ~60 total polygons, instead of ~900)
        gdf["MergeCode"] = gdf["INIT_ISS"] + gdf["INIT_EXP"] + gdf["PHENOM"] + gdf["SIG"]
               
        gdf_dissolve = gdf.dissolve(by='MergeCode')
        # Break off multi-polygons (IE: Watches that had the same values of MergeCode above, but were actually spatially separate)
        gdf_explode = gdf_dissolve.explode()
        gdf_explode["polyArea"] = gdf_explode.to_crs('epsg:3395').area
        # Remove watch boxes with tiny areas (IE: Single county watch boxes, these are artifacts from the recording of boxes catching trimmed watch areas)
        gdf_explode.drop(gdf_explode.loc[gdf_explode['polyArea'] < 2000000].index, inplace = True)    
        # Save a copy of the polygon indices, we use this in our overlap operations.
        gdf_explode.reset_index(inplace = True)
        gdf_explode["POLY_INDEX"] = gdf_explode.index       
        rng = gdf_explode.index.unique()
        
        # Iterate over the polygons.
        commands = []
        for test in ENSEMBLE_TESTS:           
            for r in rng:      
                commands.append([r, gdf_explode, sDict, test])
        tools.loggedPrint.instance().write("Mapping Ensemble GeoPandas Verification to MPI.")
        resultList = pool.map(verify_watch_polygons_ensemble, commands)
        tools.loggedPrint.instance().write("Task Complete... Saving Output.")
        # Loop into the MPI results, each worker will return a tuple containing the GDF index and the resulting values
        
        results_dict = {}
        for test in ENSEMBLE_TESTS:
            results_dict[test] = {}
            for r in rng:
                results_dict[test][r] = {}
        
        for r in resultList:
        
            index = r[0]
            test = r[1]
            
            results_dict[test][index] = {
                "UHInside": r[2],
                "UHOtherPoly": r[3],
                "UHOutside": r[4],
                "TORInside": r[5],
                "TOROtherPoly": r[6],
                "TOROutside": r[7],
            }     
        # Save the output dataframe to our verification directory.
        with open(ENSEMBLE_SAVE_DIR + "/verification_NWS_polygons.json", "w") as fileObj:
            json.dump(results_dict, fileObj)
        #for test in ENSEMBLE_TESTS:
        #    gdf_explode = gdf_explode_dict[test]
        #    gdf_explode.to_file(ENSEMBLE_SAVE_DIR + "/verification_NWS_polygons_" + test + ".shp")                    
    
    else:
        oFileDir = tDir + "/output/"
        vFileDir = tDir + "/Verification/"    
        
        domainList = sDict['domains']
        sDict["targetDirectory"] = tDir
        sDict["outFileDir"] = oFileDir
        sDict["vOutDir"] = vFileDir
        sDict["verificationHead"] = vHeadDir

        if inGeoEM:
            tools.loggedPrint.instance().write("Geography Information Directory Set as: " + inGeoEM + ", checking for validity...")
            geoFile = sDict["geo_file"]
            for domain in sDict["domains"]:
                sDict["wrf_bounds_" + domain] = None
            
                tF = inGeoEM + "/" + geoFile
                tF = tF.replace("[domain]", domain)
                if not os.path.exists(tF):
                    tools.loggedPrint.instance().write("ERROR: Geography information file [" + tF + "] was not found, aborting script.")
                    pool.close()
                    sys.exit(1)
                    
                tools.loggedPrint.instance().write("Geography Information File [" + tF + "] found, pulling geography information.")
                geoNC = xr.open_dataset(tF, engine='netcdf4')
                try:
                    wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                    sDict["wrf_bounds_" + str(domain)] = wrf_bounds
                except KeyError:
                    pool.close()
                    raise ValueError("XLAT_M / XLONG_M not found for " + domain)      
                geoNC.close()                       
        else:
            tools.loggedPrint.instance().write("Detecting geography information.")
            if((int)(sDict["geo_seperate"]) == 1):
                tools.loggedPrint.instance().write("Pulling geography information from geo_em file.")
                geoFile = sDict["geo_file"]
                for domain in sDict["domains"]:
                    sDict["wrf_bounds_" + domain] = None
                
                    tF = sDict["outFileDir"] + geoFile
                    tF = tF.replace("[domain]", domain)
                    if not os.path.exists(tF):
                        tools.loggedPrint.instance().write("ERROR: Geography information file [" + tF + "] was not found, aborting script.")
                        pool.close()
                        sys.exit(1)
                    
                    geoNC = xr.open_dataset(tF, engine='netcdf4')
                    try:
                        wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                        sDict["wrf_bounds_" + str(domain)] = wrf_bounds
                    except KeyError:
                        pool.close()
                        raise ValueError("XLAT_M / XLONG_M not found for " + domain)      
                    geoNC.close()
            else:
                tools.loggedPrint.instance().write("Pulling geography information from wrfout file.")
                for domain in sDict["domains"]:
                    sDict["wrf_bounds_" + domain] = None
                
                    tFiles = sorted(glob.glob(sDict["outFileDir"] + "wrfout_" + str(domain) + "*"))
                    geoNC = xr.open_dataset(tFiles[0], engine='netcdf4')
                    try:
                        wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                        sDict["wrf_bounds_" + str(domain)] = wrf_bounds
                    except KeyError:
                        pool.close()
                        raise ValueError("XLAT_M / XLONG_M not found for " + domain)    
                    geoNC.close()  
        # Create the numpy array for geopandas verification
        states = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                    name='admin_1_states_provinces_lakes',
                                                    scale='110m',
                                                    facecolor='none')
        state_poly = gpd.GeoSeries(states.geometries())
        gdf_stpl = gpd.GeoDataFrame(geometry=state_poly)
        for domain in domainList:
            lsr_p = gpd.GeoSeries(map(Point, zip(np.ravel(sDict["wrf_bounds_" + domain]["lon"]), np.ravel(sDict["wrf_bounds_" + domain]["lat"]))))           
            gdf_lsrp = gpd.GeoDataFrame(geometry=lsr_p)        
            gdf_lsrp["inUS"] = False    
            points_within = gpd.sjoin(gdf_lsrp, gdf_stpl, op='within')
            gdf_lsrp.loc[points_within.index, "inUS"] = True
            npInUS = gdf_lsrp["inUS"].to_numpy()
            npInUS = np.reshape(npInUS, sDict["wrf_bounds_" + domain]["lon"].shape) 
            sDict["pointsInStates_" + domain] = npInUS
        # Scan for timesteps to process
        timesteps = []
        sFile = "subhr_[domain]_[time]"
        oFList = []
        tools.loggedPrint.instance().write("Scanning for output files.")
        for domain in domainList:
            sFile = sFile.replace("[domain]", domain)
            sFile = sFile.replace("[time]", "*")
            path = oFileDir + sFile
            fList_sorted = sorted(glob.glob(path))
            oFList.extend(fList_sorted)
        tools.loggedPrint.instance().write("Found " + str(len(fList_sorted)) + " files.")
        tools.loggedPrint.instance().write("Scanning for timesteps.")
        # Scan for unique timesteps
        for f in oFList:
            tStep = datetime.strptime(f[-19:], '%Y-%m-%d_%H_%M_%S')
            if not tStep in timesteps:
                timesteps.append(tStep)
        tools.loggedPrint.instance().write("There are " + str(len(timesteps)) + " unique timesteps to analyze.")
        # Ensure we have verification data for all the timesteps.
        instructions = []
        tools.loggedPrint.instance().write("Checking the timesteps to ensure we have verification data for each.")
        tsVerify = timesteps
        #era_bounds = None
        gridrad_bounds = None
        #prism_bounds = None
        stage4_bounds = None
        #narr_bounds = None
        #
        gridrad_no_data = []
        if os.path.exists(vHeadDir + "gridrad_nonexisting_files.txt"):
            tools.loggedPrint.instance().write(" -> Found gridrad_nonexisting_files.txt, scanning...")
            with open(vHeadDir + "/gridrad_nonexisting_files.txt", "r") as grid_read:
                lineCount = 0
                for line in grid_read:
                    if(len(line) > 0):
                        gridrad_no_data.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
                        lineCount += 1
            tools.loggedPrint.instance().write(" -> " + str(lineCount) + " timesteps have been identified as non-data bearing")       

        if not os.path.exists(vHeadDir + "/gridrad_files.txt"):
            tools.loggedPrint.instance().write(" -> Missing " + vHeadDir + "/gridrad_files.txt. Please re-run the gridrad processing script to generate this file... exiting.")
            pool.close()
            sys.exit(1)
        gridrad_regular_timesteps = []
        gridrad_severe_timesteps = []
        with open(vHeadDir + "/gridrad_files.txt", "r") as grid_read:
            SEVERE_MODE = False
            for line in grid_read:
                if("SEVERE" in line):
                    SEVERE_MODE = True
                elif("REGULAR" in line):
                    SEVERE_MODE = False
                else:
                    if(len(line) > 1):
                        if(SEVERE_MODE):
                            gridrad_severe_timesteps.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
                        else:
                            gridrad_regular_timesteps.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
        
        for ts in tsVerify:           
            if not ts in gridrad_no_data:
                if ts in gridrad_regular_timesteps:
                    fileTest_gridrad = ts.strftime(vHeadDir + "gridrad_filtered_R_%Y%m%d_%H%M%S.nc")
                else:
                    fileTest_gridrad = ts.strftime(vHeadDir + "gridrad_filtered_S_%Y%m%d_%H%M%S.nc")
                if not os.path.exists(fileTest_gridrad):
                    tools.loggedPrint.instance().write("ERROR: Missing required verification data [" + fileTest_gridrad + "]")
                    pool.close()
                    sys.exit(1)
                if gridrad_bounds == None:
                    #tools.loggedPrint.instance().write("gridrad_bounds is still None. Checking timestep: " + str(ts))
                    if ts in gridrad_regular_timesteps:
                        #tools.loggedPrint.instance().write("Timstep is a regular file.")
                        grid_out = xr.open_dataset(fileTest_gridrad, engine='netcdf4')
                        gridrad_bounds = calculation.get_bounds(grid_out, 'Latitude', 'Longitude')
                        grid_out.close()
            
            if ((int(ts.strftime("%H")) % 6 == 0) and (int(ts.strftime("%M")) == 0)):
                fileTest_stage4 = ts.strftime(vHeadDir + "Stage4_%Y%m%d%H.nc")
                if not os.path.exists(fileTest_stage4):
                    tools.loggedPrint.instance().write("ERROR: Missing required verification data [" + fileTest_stage4 + "]")
                    pool.close()
                    sys.exit(1)
                if stage4_bounds == None:
                    stage4_out = xr.open_dataset(fileTest_stage4, engine='netcdf4')
                    stage4_bounds = calculation.get_bounds(stage4_out, 'latitude', 'longitude')
                    stage4_out.close()        
               
            """
            fileTest_narr = ts.strftime(vHeadDir + "air.%Y%m.nc")
            if not os.path.exists(fileTest_narr):
                tools.loggedPrint.instance().write("ERROR: Missing required verification data [" + fileTest_narr + "]")
                pool.close()
                sys.exit(1)
            if narr_bounds == None:
                narr_out = xr.open_dataset(fileTest_narr, engine='netcdf4')
                narr_bounds = calculation.get_bounds(narr_out, 'lat', 'lon')
                narr_out.close()
            """
                
            for domain in domainList:
                instructions.append(domain + ";" + ts.strftime("%Y-%m-%d_%H_%M_%S"))

        #sDict["era_bounds"] = era_bounds
        sDict["gridrad_bounds"] = gridrad_bounds
        #sDict["prism_bounds"] = prism_bounds
        sDict["stage4_bounds"] = stage4_bounds
        #sDict["narr_bounds"] = narr_bounds
        sDict["gridrad_regular_timesteps"] = gridrad_regular_timesteps
        sDict["gridrad_severe_timesteps"] = gridrad_severe_timesteps
        
        #tools.loggedPrint.instance().write("DEBUG: " + str(sDict["gridrad_bounds"]))
        
        # Collect the bounds from GridRAD severe timesteps
        gridrad_severe_regridder_numbers = {}
        if os.path.exists(vHeadDir + "gridrad_severe_regridders.txt"):
            tools.loggedPrint.instance().write(" -> Found gridrad_severe_regridders.txt, scanning to ensure file validity")          
            with open(vHeadDir + "gridrad_severe_regridders.txt", "r") as grid_read:
                for line in grid_read:
                    line_splits = line.strip().split(";")
                    timestring = line_splits[0]
                    regridder_number = line_splits[1]
                    gridrad_severe_regridder_numbers[timestring] = regridder_number
                    # For now, we just need to verify the file exists.
                    if not os.path.exists(vHeadDir + "conservative_GRIDRAD_severe" + regridder_number + "_" + str(domain) + ".nc"):
                        tools.loggedPrint.instance().write("Error: Missing regridder conservative_GRIDRAD_severe" + regridder_number + "_" + str(domain) + ".nc, please re-run the post_processing script.")
                        pool.close()
                        sys.exit(1)
                    else:                   
                        if not "gridrad_severe" + regridder_number + "_bounds" in sDict:
                            tStep = datetime.strptime(timestring, "%Y%m%d%H%M%S")
                            fileTest_gridrad = tStep.strftime(vHeadDir + "gridrad_filtered_S_%Y%m%d_%H%M%S.nc")
                            grid_out = xr.open_dataset(fileTest_gridrad, engine='netcdf4')
                            gridrad_s_bounds = calculation.get_bounds(grid_out, 'Latitude', 'Longitude')
                            grid_out.close() 
                            sDict["gridrad_severe" + regridder_number + "_bounds"] = gridrad_s_bounds
        
        sDict["gridrad_severe_regridder_numbers"] = gridrad_severe_regridder_numbers
        # Construct our iterable.
        commands = []
        for instruction in instructions:
            commands.append([instruction, sDict])
        # tools.loggedPrint.instance().write("Command List:\n" + str(commands) + "\n\n")
        # Run the verification process.
        tools.loggedPrint.instance().write("Mapping WRF Output Verification to MPI.")
        pool.map(run_verification, commands)
        tools.loggedPrint.instance().write("WRF Output Verification Complete...")
        tools.loggedPrint.instance().write("Running LSR Verification.")
        run_lsr_verification(timesteps, sDict)
        tools.loggedPrint.instance().write("LSR Verification Complete...")
        tools.loggedPrint.instance().write("Constructing GeoPandas verification for NWS Polygon Objects.")
        # The Watch Box verification step is separate since we need to iterate over a geodataframe object.
        gdf = gpd.GeoDataFrame(columns=['WFO', 'ISSUED', 'EXPIRED', 'INIT_ISS', 'INIT_EXP', 'PHENOM', 'GTYPE',
                                              'SIG', 'ETN', 'STATUS', 'NWS_UGC', 'AREA_KM2', 'UPDATED', 'HVTEC_NWSLI',
                                              'HVTEC_SEVERITY', 'HVTEC_CAUSE', 'HVTEC_RECORD', 'IS_EMERGENCY', 'POLYBEGIN',
                                              'POLYEND', 'WINDTAG', 'HAILTAG', 'TORNADOTAG', 'DAMAGETAG', 'geometry'], geometry='geometry')
        gdf_list = []
        df_files = []
        for t in timesteps:
            df_file = t.strftime("wwa_%Y01010000_%Y12312359.shp")
            if not df_file in df_files:
                df_files.append(df_file)
        # Open all the required shapefiles and slice out the polygons we need.
        for df_file in df_files:
            df = gpd.read_file(vHeadDir + "/" + df_file)
            df["ISSUED"] = pd.to_numeric(df["ISSUED"])
            df["EXPIRED"] = pd.to_numeric(df["EXPIRED"])
            df_slice1 = df[df.ISSUED > int(timesteps[0].strftime("%Y%m%d%H%M"))]
            df_slice2 = df_slice1[df_slice1.ISSUED < int(timesteps[-1].strftime("%Y%m%d%H%M"))]        
            df_slice3 = df_slice2[(df_slice2.SIG == "A")]
            df_slice4 = df_slice3[(df_slice3.PHENOM == "TO") | (df_slice3.PHENOM == "SV")]
            gdf_list.append(df_slice4)
        gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        # Dissolve polygons into watch boxes (Reduces to ~60 total polygons, instead of ~900)
        gdf["MergeCode"] = gdf["INIT_ISS"] + gdf["INIT_EXP"] + gdf["PHENOM"] + gdf["SIG"]
        gdf_dissolve = gdf.dissolve(by='MergeCode')
        # Break off multi-polygons (IE: Watches that had the same values of MergeCode above, but were actually spatially separate)
        gdf_explode = gdf_dissolve.explode()
        gdf_explode["polyArea"] = gdf_explode.to_crs('epsg:3395').area
        # Remove watch boxes with tiny areas (IE: Single county watch boxes, these are artifacts from the recording of boxes catching trimmed watch areas)
        gdf_explode.drop(gdf_explode.loc[gdf_explode['polyArea'] < 2000000].index, inplace = True)    
        # Save a copy of the polygon indices, we use this in our overlap operations.
        gdf_explode.reset_index(inplace = True)
        gdf_explode["POLY_INDEX"] = gdf_explode.index
        # Iterate over the polygons.
        commands = []
        rng = gdf_explode.index.unique()
        for r in rng:
            commands.append([r, gdf_explode, sDict])
        tools.loggedPrint.instance().write("Mapping GeoPandas Verification to MPI.")
        resultList = pool.map(verify_watch_polygons, commands)
        tools.loggedPrint.instance().write("Saving Output.")
        # Loop into the MPI results, each worker will return a tuple containing the GDF index and the resulting values
        """
        for r in resultList:
            index = r[0]
            gdf_explode.at[gdf_explode["POLY_INDEX"] == index, "UHInside"] = r[1]
            gdf_explode.at[gdf_explode["POLY_INDEX"] == index, "UHOtherPoly"] = r[2]
            gdf_explode.at[gdf_explode["POLY_INDEX"] == index, "UHOutside"] = r[3]
            gdf_explode.at[gdf_explode["POLY_INDEX"] == index, "TORInside"] = r[4]
            gdf_explode.at[gdf_explode["POLY_INDEX"] == index, "TOROtherPoly"] = r[5]
            gdf_explode.at[gdf_explode["POLY_INDEX"] == index, "TOROutside"] = r[6]        
        # Save the output dataframe to our verification directory.
        gdf_explode.to_file(vFileDir + "verification_NWS_polygons.shp")
        """
        results_dict = {}
        for r in rng:
            results_dict[r] = {}
        
        for r in resultList:       
            index = r[0]
            
            results_dict[index] = {
                "UHInside": r[1],
                "UHOtherPoly": r[2],
                "UHOutside": r[3],
                "TORInside": r[4],
                "TOROtherPoly": r[5],
                "TOROutside": r[6],
            }     
        # Save the output dataframe to our verification directory.
        with open(vFileDir + "/verification_NWS_polygons.json", "w") as fileObj:
            json.dump(results_dict, fileObj)        
        
        
    # Clean up.
    pool.close()
    tools.loggedPrint.instance().write("Program complete.")   

def main(argv):
    inDir = ''
    inGeoEM = ''
    headVDir = ''
    ensembleMode = False
    try:
        opts, args = getopt.getopt(argv, "hei:v:g:", ["idir="])
    except getopt.GetoptError:
        print("Error: Usage: process_verification.py -i <inputdirectory> -v <headVerificationDir> -g <geo_em> -e")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Usage: process_verification.py: Pass the following arguments; (R) indicates required")
            print(" -e: Activates ensemble verification mode, this disables the input directory parameter being required (-v is still required)")
            print(" -i <inputdirectory> (R): The path to the output files (wrfout*, subhr*, AFWA*, etc)")
            print(" -v <headVerificationDir> (R): The path to verification files created through post_processing.py")
            print(" -g <geo_em>: If the geography information (geo_em.d01.nc, etc) is not in the /output/ folder, or is in a shared location, pass that directory to this argument")
            sys.exit()
        elif opt == '-e':
            ensembleMode = True
        elif opt in ("-i", "--idir"):
            inDir = arg
        elif opt in ("-v"):
            headVDir = arg
        
    if headVDir == '':
        print("Error: Verification head directory not defined, exiting.")
        sys.exit()         
        
    if not ensembleMode:
        if inDir == '':
            print("Error: input directory not defined, exiting.")
            sys.exit()
            
    run_script(ensembleMode, inDir, headVDir, inGeoEM)
    
if __name__ == "__main__":
    main(sys.argv[1:])   
    
    
    
"""
OLD CODE

Saved here just in case I need it for anything.
"""    
    
"""        
# PRISM
if(int(TIMESTR.strftime("%H")) == 12 and int(TIMESTR.strftime("%M")) == 0 and V_PRISM is not None):
    one_day_precip_prism = V_PRISM["precip"]
    last_day_file = settings["outFileDir"] + "wrfout_" + domain + "_" + PREVDAY.strftime("%Y-%m-%d_%H_%M_%S")
    if os.path.exists(last_day_file):
        WRF_P = xr.open_dataset(last_day_file, engine='netcdf4')
        precip_ending_now = WRF["RAINC"][0] + WRF["RAINNC"][0]
        precip_ending_before = WRF_P["RAINC"][0] + WRF_P["RAINNC"][0]
        WRF_P.close()
        one_day_precip = precip_ending_now - precip_ending_before
        one_day_precip_regridded = prism_regridder(one_day_precip)
        
        V_AccP_Diff = one_day_precip_regridded - one_day_precip_prism
        V_AccP_RMSE = np.sqrt(np.mean((one_day_precip_regridded - one_day_precip_prism)**2))
        
        for iPC, PC_T in enumerate(FSS_THRESH_PRECIP):
            fss = []
            for f in FSS_Scales:
                fss.append(calculation.FSS(one_day_precip_regridded, one_day_precip_prism, PC_T, f)) 
            
            VERIF_O["V_AccP_" + str(FSS_THRESHS_PRECIP_PERCENTILES[iPC]) + "_FSS"] = (['FSS_Scales'], fss)
        
        VERIF_O["AccP_OneDay"] = (['y_wrf', 'x_wrf'], one_day_precip.data)
        VERIF_O["V_AccP_Diff"] = (['y_prism', 'x_prism'], V_AccP_Diff.data)
        VERIF_O["V_AccP_RMSE"] = (['TimeFile'], [V_AccP_RMSE.data])
        
    else:
        tools.loggedPrint.instance().write("runV: " + timestep + ": WARN: File (" + last_day_file + ") not found, ignoring PRISM verification for this day.")
        
# Total Precip & PRISM
if(TIMESTR == END_ANALYSIS_TIME):
    first_date_file = settings["outFileDir"] + "wrfout_" + domain + "_" + START_ANALYSIS_TIME.strftime("%Y-%m-%d_%H_%M_%S")
    if not os.path.exists(first_date_file):
        tools.loggedPrint.instance().write("runV: " + timestep + ": WARN: Could not locate a wrfout file at " + START_ANALYSIS_TIME.strftime("%Y-%m-%d_%H_%M_%S") + ", cannot calculate the total precipitation in the analysis window.")
    else:
        WRF_P = xr.open_dataset(first_date_file, engine='netcdf4')
        precip_ending_now = WRF["RAINC"][0] + WRF["RAINNC"][0]
        precip_ending_before = WRF_P["RAINC"][0] + WRF_P["RAINNC"][0]
        WRF_P.close()
        
        Total_Prec = precip_ending_now - precip_ending_before
        VERIF_O["V_AccP_VerfWindow"] = (['y_wrf', 'x_wrf'], Total_Prec.data)
        
        # Calculate the prism timesteps
        prism_start = datetime.strptime(START_ANALYSIS_TIME.strftime("%Y%m%d_120000"), "%Y%m%d_%H%M%S")
        prism_steps = []
        prism_now = prism_start
        while prism_now <= END_ANALYSIS_TIME:
            # Make sure to skip the first.
            if prism_now != START_ANALYSIS_TIME:
                prism_steps.append(prism_now)
            prism_now = prism_now + timedelta(days = 1)
        
        # Calculate the total accumulated precipitation.
        prism_total_precip = None
        for t in prism_steps:
            fileTest_prism = t.strftime(verificationHead + "PRISM_ppt_%Y%m%d.nc")
            vPRISMSTEP = xr.open_dataset(fileTest_prism, engine='netcdf4')                    
            if prism_total_precip is None:
                prism_total_precip = vPRISMSTEP["precip"]
            else:
                prism_total_precip += vPRISMSTEP["precip"]
        
        VERIF_O["PrismPrecip_VerfWindow"] = (['y_prism', 'x_prism'], prism_total_precip.data)
        # Calculate the difference
        window_precip_regridded = prism_regridder(Total_Prec.data)
        
        V_AccP_Window_Diff = window_precip_regridded - prism_total_precip
        V_AccP_Window_RMSE = np.sqrt(np.mean((window_precip_regridded - prism_total_precip)**2))
        
        for iPC, PC_T in enumerate(FSS_THRESH_PRECIP):
            fss = []
            for f in FSS_Scales:
                fss.append(calculation.FSS(window_precip_regridded, prism_total_precip, PC_T, f))
            VERIF_O["V_AccP_" + str(FSS_THRESHS_PRECIP_PERCENTILES[iPC]) + "_VerfWindow_FSS"] = (['FSS_Scales'], fss)  
        
        VERIF_O["V_AccP_VerfWindow_Diff"] = (['y_prism', 'x_prism'], V_AccP_Window_Diff.data)
        VERIF_O["V_AccP_VerfWindow_RMSE"] = (['TimeFile'], [V_AccP_Window_RMSE.data])                            
"""    
#!/usr/bin/python
# calculation.py
# Robert C Fritzen - Dpt. Geographic & Atmospheric Sciences
#
# Python script file containing computation tasks
#
# This script file should not be called.

import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import xesmf as xe
import scipy.stats as stats
from scipy.ndimage.filters import uniform_filter, generic_filter
import glob
import re
import sys, getopt
import mpipool
import tools
import settings

from netCDF4 import Dataset
from wrf import getvar, interplevel

"""
Constants
"""
FILE_VERSION = 1.1

"""
Script
"""
#get_bounds(): Construct a dictionary object of boundaries, used for xESMF regridder object, Pass the netCDF object and the keys for latitude / longitude desired
#def get_bounds(ncArray, lat_keyword, lon_keyword):    
#    bounds = {}
#    
#    bounds["lon"] = ncArray[lon_keyword]
#    bounds["lat"] = ncArray[lat_keyword]
#    
#    return bounds 
def get_bounds(arr, latName, lonName, wTime = False):
    if(wTime):
        X = arr[lonName][0]
        Y = arr[latName][0]       
    else:
        X = arr[lonName]
        Y = arr[latName]
    
    bounds = {}
    
    if(len(X.shape) == 1 and len(Y.shape) == 1):
        lonMin = np.nanmin(X)
        latMin = np.nanmin(Y)
        lonMax = np.nanmax(X)
        latMax = np.nanmax(Y)

        sizeLon = len(X)
        sizeLat = len(Y)
        
        gridSize = np.abs(X[1] - X[0])
        
        bounds["lon"] = X
        bounds["lat"] = Y
        bounds["lon_b"] = np.linspace(lonMin-(gridSize/2), lonMax+(gridSize/2), sizeLon+1)
        bounds["lat_b"] = np.linspace(latMin-(gridSize/2), latMax+(gridSize/2), sizeLat+1).clip(-90, 90)    
    elif(len(X.shape) == 2 and len(Y.shape) == 2):                
        Xwise_diff = X[-2].values - X[-1].values
        Xconcat = (X[-1].values + Xwise_diff)[None, ...]
        Xb_x = np.append(X, Xconcat, axis=0)
        Ywise_diff = Xb_x[:,-2] - Xb_x[:,-1]
        Yconcat = (Xb_x[:,-1] + Ywise_diff)[..., None]
        Xb = np.append(Xb_x, Yconcat, axis=1)

        XD = np.diff(Xb, axis=1)
        lastXD = (XD[:, -1])[..., None]
        XD_App = np.append(XD, lastXD, axis=1)

        X_Bounds = Xb + XD_App

        Xwise_diff = Y[-2].values - Y[-1].values
        Xconcat = (Y[-1].values + Xwise_diff)[None, ...]
        Xb_x = np.append(Y, Xconcat, axis=0)
        Ywise_diff = Xb_x[:,-2] - Xb_x[:,-1]
        Yconcat = (Xb_x[:,-1] + Ywise_diff)[..., None]
        Yb = np.append(Xb_x, Yconcat, axis=1)

        YD = np.diff(Yb, axis=0)
        lastYD = (YD[-1])[None, ...]
        YD_App = np.append(YD, lastYD, axis=0)

        Y_Bounds = Yb + YD_App 
        
        bounds["lon"] = X
        bounds["lat"] = Y    
        bounds["lon_b"] = X_Bounds
        bounds["lat_b"] = Y_Bounds.clip(-90, 90)
    else:
        raise ValueError("Shape of the X/Y arrays either mismatch, or ar >2")
        
    return bounds 
 
# FSS(): Compute the fraction skill score
def FSS(xInput, xObservations, thresh, scale):
    if(len(xInput.shape) != 2 and len(xObservations.shape != 2)):
        raise ValueError("FSS(): xInput and xObservations must be arrays of shape 2 (One or both do not match this criterion)")
    # Create deep-copies of the arrays to perform calculations on
    if(type(xInput) == xr.DataArray):
        xInput = xInput.copy().values
    if(type(xObservations) == xr.DataArray):
        xObservations = xObservations.copy().values
    # Reduce finite values by the defined threshold
    xInput[~np.isfinite(xInput)] = thresh - 1
    xObservations[~np.isfinite(xObservations)] = thresh - 1
    # Convert the numeric fields to binary (T/F) fields for usage in the moving window calculations
    bxInput = (xInput >= thresh).astype(float)
    bxObservations = (xObservations >= thresh).astype(float)
    # Compute the number of pixels above the threshold within the square moving window area defined by scale
    if scale > 1:
        sInput = uniform_filter(bxInput, size=scale, mode='constant', cval=0.0)
        sObservations = uniform_filter(bxObservations, size=scale, mode='constant', cval=0.0)
    else:
        sInput = bxInput
        sObservations = bxObservations
    # Store some quick squares for the FSS calculations
    sumSqInput = np.nansum(sInput**2)
    sumSqObservations = np.nansum(sObservations**2)
    sumTotals = np.nansum(sInput * sObservations)
    # Compute the FSS
    fssNum = sumSqInput - 2.0 * sumTotals + sumSqObservations
    fssDen = sumSqInput + sumSqObservations
    if fssDen > 0:
        return (1.0 - (fssNum / fssDen))
    else:
        return np.nan    
    
# NProb(): Compute the neighborhood probability of a threshold within a window (scale)    
def NProb(xInput, thresh, scale):
    # Create a deep copy for alterations
    if(type(xInput) == xr.DataArray):
        xInput = xInput.copy().values
    # Reduce finite values by the defined threshold
    xInput[~np.isfinite(xInput)] = thresh - 1
    # Create a binary (T/F) field for values over the threshold
    bxInput = (xInput >= thresh).astype(float)
    # Define an internal sum method for scipy.generic_filter
    def NPROB_FILTER(X):
        return X.sum() / len(X)
    # Compute the neighborhood probability (Number of values within a window, divided by the number of obs)
    if scale > 1:
        sInput = generic_filter(bxInput, NPROB_FILTER, size=scale, mode='constant', cval=0.0)
    else:
        sInput = bxInput
    return sInput 

#calc_td(): Compute dewpoint temperature from pressure / Wv
def calc_td(pressure, Wv):        
    Wv_max = xr.where(Wv < 0, 0, Wv) # np.max(Wv, 0)
    first_guess = (Wv_max * pressure) / (0.622 + Wv_max)
    # Adjust the "first guess" such that 0 is bumped to an infintesimally small value.
    first_guess_fix = xr.where(first_guess < 0.001, 0.001, first_guess) # np.maximum(first_guess, 0.001)
    
    td = ((243.5 * np.log(first_guess_fix)) - 440.8) / (19.48 - np.log(first_guess_fix))
    return td
    
#calc_es(): Calculate the saturation vapor pressure from a given temperature (Kelvin)
def calc_es(temperature):
    return 6.112 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
   
#calc_ws(): Calculate the saturation vapor pressure from atmospheric pressure and temperature
def calc_ws(pressure, temperature):
    es = calc_es(temperature)
    ws = 0.622 * es / (pressure - es)
    return ws
    
#calc_theta(): Calculate the potential temperature
def calc_theta(pressure, temperature):
    return temperature / ((pressure / 100000) ** 0.286)

#theta_to_temp(): Convert potential temperature to air temperature
def theta_to_temp(theta, pressure):
    return ((pressure / 100000.0) ** 0.286) * theta

#calc_thetae(): Calculate the equivalent potential temperature
# q - mixing ratio (QVAPOR in WRF)
# t - Temperature (K)
# p - Pressure (Pa)
def calc_thetae(q, t, p):
    epsilon = 0.622
    gamma = 0.2857142857142857 # Rd/Cp
    gamma_moist = -0.279
    tlclcon = [2840.0, 3.5, 4.8, 55.0]
    thecon = [3376.0, 2.54, 0.81]
    
    phpa = p / 100
    
    # Ensure only >0 q values are present
    q_max = xr.where(q < 0, 0, q)
    
    # Compute vapor pressure
    e = q_max * (phpa) / (epsilon + q_max)
    
    # compute Tlcl
    tlcl = tlclcon[0] / (np.log(t**tlclcon[1]/e) - tlclcon[2]) + tlclcon[3]
    
    # Compute theta_e
    th_e = t * (1000 / phpa) ** (gamma * (1.0 + gamma_moist * q_max)) * np.exp((thecon[0] / tlcl - thecon[1]) * q_max * (1.0 + thecon[2] * q_max))
    
    return th_e

#q_to_w(): Convert specific humidity (q) to mixing ratio (w)    
def q_to_w(qv):
    w = qv / (1 - qv)
    return w

#td2_from_wrfout(): Run calc_td2 on a input WRF file
def td2_from_wrfout(wrfout):
    p = wrfout["PSFC"][0]
    qv = wrfout["Q2"][0].copy()
    
    P_in_hpa = 0.01 * (p)
    #qv[qv < 0] = 0 #xarray doesn't support multi-dimensional indexing, use where
    qv = xr.where(qv < 0, 0, qv)
     
    td = calc_td(P_in_hpa, qv)
    return td
    
#td_from_wrfout(): Calculate the 3D dewpoint temperature from an input WRFOUT file
def td_from_wrfout(wrfout):
    p = wrfout["P"][0] + wrfout["PB"][0]
    qv = wrfout["QVAPOR"][0].copy()
    
    pHPA = p * 0.01
    qv = xr.where(qv < 0, 0, qv)
    td = calc_td(pHPA, qv)
    
# Replacement for wrf-python.interplevel in the event Dataset() is not MPI safe
def linear_interp(data, totalp, plev):       
    #find index of level above plev        
    above = np.argmax(totalp < plev*100., axis=0)
    below = above - 1 # index of pressure below
    # pressure at level above plev 
    nz,ny,nx = totalp.shape
    upperP = totalp.reshape(nz,ny*nx)[above.flatten(),range(ny*nx)].reshape(ny,nx)
    #pressure at level below plev
    lowerP = totalp.reshape(nz,ny*nx)[below.flatten(),range(ny*nx)].reshape(ny,nx)
    # value above plev
    nz,ny,nx = data.shape
    aboveVal = data.reshape(nz, ny*nx)[above.flatten(),range(ny*nx)].reshape(ny,nx)
    #value below plev
    belowVal = data.reshape(nz, ny*nx)[below.flatten(),range(ny*nx)].reshape(ny,nx)
    # calc total dist betweek upper and lower
    totaldist = upperP - lowerP
    #calc weighting
    weight = np.abs( ( (plev*100.) - lowerP) / (totaldist) )
    #calc interpolated value    
    outVal = ( belowVal * (1 - weight) ) + ( aboveVal * weight )
    
    return outVal    
    
"""
    calculation.py - MPI Script Begin
"""    
    
def compute(command_tuple):
    command_list = command_tuple[0].split(";")
    domain = command_list[0]
    timestep = command_list[1]
    #day_list = command_list[2][1:-1].replace(" ", "").split(",") #RF: No longer used.
    
    settings = command_tuple[1]
    
    wrf_bounds = settings["wrf_bounds_" + str(domain)]
    
    tools.loggedPrint.instance().write("calcs: " + timestep + ": LOG: Begin.")
    
    TIMESTR = datetime.strptime(timestep, '%Y-%m-%d_%H_%M_%S')
    # Check if we need calculations at this time step.
    if(os.path.exists(settings["outFileDir"] + "/calcs_" + domain + "_" + TIMESTR.strftime("%Y-%m-%d_%H_%M_%S") + ".nc")):
        VERF = xr.open_dataset(settings["outFileDir"] + "/calcs_" + domain + "_" + TIMESTR.strftime("%Y-%m-%d_%H_%M_%S") + ".nc")
        try:
            fVer = float(VERF.attrs["FILE_VERSION"])
            if(fVer >= FILE_VERSION):
                tools.loggedPrint.instance().write("calcs: " + timestep + ": LOG: Calculation output at same or newer file version already present for this timestep, skipping.")
                VERF.close()
                return True
        except KeyError:
            pass
        VERF.close()
        
    subhrout = settings["outFileDir"] + "subhr_" + domain + "_" + timestep  
    afwaout = settings["outFileDir"] + "AFWA_" + domain + "_" + timestep
    pgrbout = settings["outFileDir"] + "pgrb3D_" + domain + "_" + timestep
    wrfout = settings["outFileDir"] + "wrfout_" + domain + "_" + timestep        
    tools.loggedPrint.instance().write("calcs: " + timestep + ": Opening Files")
    
    SUBHR = None
    AFWA = None #xr.open_dataset(afwaout, engine='netcdf4')
    PGRB = None #xr.open_dataset(pgrbout, engine='netcdf4')
    WRF = None #xr.open_dataset(wrfout, engine='netcdf4')        
    
    if os.path.exists(subhrout):
        SUBHR = xr.open_dataset(subhrout, engine='netcdf4') 
    if os.path.exists(afwaout):
        AFWA = xr.open_dataset(afwaout, engine='netcdf4') 
    if os.path.exists(pgrbout):
        PGRB = xr.open_dataset(pgrbout, engine='netcdf4') 
    if os.path.exists(wrfout):
        WRF = xr.open_dataset(wrfout, engine='netcdf4')   

    # Create our output file, reorder the dimensions appropriately
    CALC = xr.Dataset(coords = {
        "Time": [0],
        "XLAT": (["y_wrf", "x_wrf"], wrf_bounds["lat"].data),
        "XLONG": (["y_wrf", "x_wrf"], wrf_bounds["lon"].data),
    })     
    CALC.attrs["FILE_VERSION"] = FILE_VERSION
    
    # Perform calculations...
    if WRF is not None:
        td2 = td2_from_wrfout(WRF)
        CALC["TD2"] = (['y_wrf', 'x_wrf'], td2.data)
         
        p_wrfout = WRF["P"][0] + WRF["PB"][0]
        theta_wrf = WRF["T"][0] + 300.0
        q_wrf = WRF["QVAPOR"][0]
        temp_wrf = theta_to_temp(theta_wrf, p_wrfout)
        
        th_e_wrfout = calc_thetae(q_wrf, temp_wrf, p_wrfout)
        th_e_wrfout_1000 = linear_interp(th_e_wrfout.data, p_wrfout.data, 1000)
        th_e_wrfout_925 = linear_interp(th_e_wrfout.data, p_wrfout.data, 925)
        th_e_wrfout_850 = linear_interp(th_e_wrfout.data, p_wrfout.data, 850)
        th_e_wrfout_700 = linear_interp(th_e_wrfout.data, p_wrfout.data, 700)
        th_e_wrfout_500 = linear_interp(th_e_wrfout.data, p_wrfout.data, 500)
        
        CALC["THE_E_1000"] = (['y_wrf', 'x_wrf'], th_e_wrfout_1000.data)
        CALC["THE_E_925"] = (['y_wrf', 'x_wrf'], th_e_wrfout_925.data)
        CALC["THE_E_850"] = (['y_wrf', 'x_wrf'], th_e_wrfout_850.data)
        CALC["THE_E_700"] = (['y_wrf', 'x_wrf'], th_e_wrfout_700.data)
        CALC["THE_E_500"] = (['y_wrf', 'x_wrf'], th_e_wrfout_500.data)       
         
    if SUBHR is not None:
        # Calculate the UH NProbs
        np_uh25 = NProb(SUBHR["UH"][0], 75, 10)
        np_uh03 = NProb(SUBHR["UH"][0], 100, 10)
        CALC["NP_UH25"] = (['y_wrf', 'x_wrf'], np_uh25)
        CALC["NP_UH03"] = (['y_wrf', 'x_wrf'], np_uh03)
    if AFWA is not None:
        np_tor = NProb(AFWA["AFWA_TORNADO"][0], 29, 10)
        CALC["NP_TOR"] = (['y_wrf', 'x_wrf'], np_tor)
    
    if SUBHR is not None:
        SUBHR.close()
    if AFWA is not None:
        AFWA.close()
    if PGRB is not None:
        PGRB.close()
    if WRF is not None:
        WRF.close()
    
    tools.loggedPrint.instance().write("calcs: " + timestep + ": Save Output")
    # Write the output
    CALC.to_netcdf(settings["outFileDir"] + "/calcs_" + domain + "_" + TIMESTR.strftime("%Y-%m-%d_%H_%M_%S") + ".nc")    
    
    CALC.close()
    
def run_script(tDir): 
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

    oFileDir = tDir + "/output/"
    vFileDir = tDir + "/Verification/"    
    
    domainList = sDict['domains']
    sDict["targetDirectory"] = tDir
    sDict["outFileDir"] = oFileDir
    sDict["vOutDir"] = vFileDir
    
    tools.loggedPrint.instance().write("Detecting geography information.")
    if((int)(sDict["geo_seperate"]) == 1):
        tools.loggedPrint.instance().write("Pulling geography information from geo_em file.")
        geoFile = sDict["geo_file"]
        for domain in sDict["domains"]:
            sDict["wrf_bounds_" + domain] = None
        
            tF = sDict["outFileDir"] + geoFile
            tF = tF.replace("[domain]", domain)
            geoNC = xr.open_dataset(tF, engine='netcdf4')
            try:
                wrf_bounds = get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                sDict["wrf_bounds_" + str(domain)] = wrf_bounds
            except KeyError:
                raise ValueError("XLAT_M / XLONG_M not found for " + domain)      
            geoNC.close()
    else:
        tools.loggedPrint.instance().write("Pulling geography information from wrfout file.")
        for domain in sDict["domains"]:
            sDict["wrf_bounds_" + domain] = None
        
            tFiles = sorted(glob.glob(sDict["outFileDir"] + "wrfout_" + str(domain) + "*"))
            geoNC = xr.open_dataset(tFiles[0], engine='netcdf4')
            try:
                wrf_bounds = get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                sDict["wrf_bounds_" + str(domain)] = wrf_bounds
            except KeyError:
                raise ValueError("XLAT_M / XLONG_M not found for " + domain)    
            geoNC.close()    
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
    # Construct our iterable.
    instructions = []
    for ts in timesteps:
        for domain in domainList:
            instructions.append(domain + ";" + ts.strftime("%Y-%m-%d_%H_%M_%S"))        
    commands = []
    for instruction in instructions:
        commands.append([instruction, sDict])
    # tools.loggedPrint.instance().write("Command List:\n" + str(commands) + "\n\n")
    # Run the verification process.
    tools.loggedPrint.instance().write("Mapping Calculaton commands to MPI.")
    pool.map(compute, commands)
    pool.close()
    tools.loggedPrint.instance().write("Script Complete.")
    return True

def main(argv):
    inDir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ["idir="])
    except getopt.GetoptError:
        print("Error: Usage: calculation.py -i <inputdirectory>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Usage: calculation.py -i <inputDirectory>")
            sys.exit()
        elif opt in ("-i", "--idir"):
            inDir = arg
         
    if inDir == '':
        print("Error: input directory not defined, exiting.")
        sys.exit()
    else:  
        run_script(inDir)
    
if __name__ == "__main__":
    main(sys.argv[1:])       
    
"""
OLD CODE

    elif instruction == "constructV":
        domain = command_list[1]
        timestep = command_list[2] 
        TIMESTR = datetime.strptime(timestep, '%Y-%m-%d_%H_%M_%S')
        # The construct verification task is used to merge our verification sets (PRISM, ERA, GridRAD) into a single netCDF file
        #  start by loading up the files we need.
        # PRISM ppt goes from 12z - 12z, must consider this when doing the "runV" steps.
        gridrad_f = constants["targetDirectory"] + "/Verification/gridrad_filtered_%Y%m%d_%H%M%S.nc"
        gridradF = TIMESTR.strftime(gridrad_f)
        GRIDRAD = xr.open_dataset(gridradF, engine='netcdf4', chunks={})
        PRISM = None
        if(TIMESTR.strftime("%H") == "12"):
            prism_f = constants["targetDirectory"] + "/Verification/PRISM_ppt_%Y%m%d.nc"
            prismF = TIMESTR.strftime(prism_f)
            PRISM = xr.open_dataset(prismF, engine='netcdf4', chunks={})
        era_f1 = constants["targetDirectory"] + "/Verification/era5-pres-%Y%m%d.nc"
        eraF1 = TIMESTR.strftime(era_f1)
        ERA1 = xr.open_dataset(eraF1, engine='netcdf4', chunks={})
        era_f2 = constants["targetDirectory"] + "/Verification/era5-slev-%Y%m%d.nc"
        eraF2 = TIMESTR.strftime(era_f2)
        ERA2 = xr.open_dataset(eraF2, engine='netcdf4', chunks={})        
        # Now build our new netCDF4 file...
        
        xs_coords = {
            'era_lon': ERA1["longitude"],
            'era_lat': ERA1["latitude"],
            'era_vlev': ERA1["level"],
            'gridrad_lon': GRIDRAD["Longitude"],
            'gridrad_lat': GRIDRAD["Latitude"],
            # 'gridrad_vlev': GRIDRAD["Altitude"],            
        }
        if(PRISM is not None):
            xs_coords['prism_lon'] = PRISM['prism_lon']
            xs_coords['prism_lat'] = PRISM['prism_lat']
        
        tools.loggedPrint.instance().write("xs_coords [" + timestep + "]: " + str(xs_coords))
        
        xs_full = xr.Dataset(coords = xs_coords)
        # ERA variables have a time index up front, we have a file for each day, so we need the correct index from the file.
        hIdx = int(TIMESTR.strftime("%H"))
        xs_full["u"] = (['era_vlev', 'era_lat', 'era_lon'], ERA1["u"][hIdx].data)
        xs_full["v"] = (['era_vlev', 'era_lat', 'era_lon'], ERA1["v"][hIdx].data)
        xs_full["cape"] = (['era_lat', 'era_lon'], ERA2["cape"][hIdx].data)
        xs_full["cin"] = (['era_lat', 'era_lon'], ERA2["cin"][hIdx].data)
        xs_full["d2m"] = (['era_lat', 'era_lon'], ERA2["d2m"][hIdx].data)
        xs_full["ref_0"] = (['gridrad_lat', 'gridrad_lon'], GRIDRAD["Reflectivity"][0].data)
        if(PRISM is not None):
            xs_full["precip"] = (['prism_lat', 'prism_lon'], PRISM["precip"].data)
        
        xs_full.to_netcdf(constants["targetDirectory"] + "/Verification/verification_data_" + TIMESTR.strftime("%Y%m%d_%H%M%S") + ".nc")
        
        ERA1.close()
        ERA2.close()
        if(PRISM):
            PRISM.close()
            
        xs_full.close()
"""
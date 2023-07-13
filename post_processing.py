#!/usr/bin/python
# post_processing.py
# Robert C Fritzen - Dpt. Geographic & Atmospheric Sciences
#
# Primary script file, runs the job.
#
# USAGE: post_processing.py -i [targetDirectory]
#  -> Where [targetDirectory] is the main run directory (Not the output directory)


"""
    Desired Product(s) / Comparisons
    --------------------------------
    
    Mesoscale Predictors:
    
    Sim.Ref - Grid Rad / FNexRad
        -> Convective Precipitation (> 40 dBz) - Grid Rad
    Accumulated Precip - PRISM / Stage IV
    Updraft Helicity (2 - 5 KM, 0 - 3 KM) - NARR
    
    Other Comparisons:
    
    AFWA Hail - LSRs (Did hail occur where a AFWA spike occurred?) / Severe Watches & Warnings
    AFWA Tornado - LSRs (Same as above) / Severe Watches & Warnings
"""


from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import time
import os
import glob
import sys, getopt
import cdsapi
from osgeo import gdal
from zipfile import ZipFile
import xesmf as xe

import settings
import tools
import calculation

DEBUG_MODE = False

def post_processing(directory):
    print("WRF Python Post-Processing Script")
    print(" - Target Directory: " + directory)
    
    constantsDict = {}
    constantsDict["targetDirectory"] = directory
    
    print(" - 1. Scanning for control file")
    aSet = settings.Settings()
    sDict = aSet.get_full_dict()
    domainList = sDict['domains']    
    print(" - 1. Done")
    
    """
    
    Old Code, Checked for a multi-day structure, no need for this now.
    
    print(" - 2. Scanning directory structure, checking for list of run days")
    subDirs = glob.glob(directory + "/Day*/")
    if(len(subDirs) == 0):
        print(" - 2. Error: Did not find Day subdirectories, did you target an individual day's directory?")
        sys.exit(0)
    run_directories = {}
    for iDay in subDirs:
        dayStr = iDay[iDay.rfind("Day"):-1]
        dayNumber = int(dayStr[3:])
        run_directories[dayNumber] = {'directory': iDay, 'day_string': dayStr}
    print(" - 2. Found " + str(len(run_directories)) + " directories following the naming convention, scanning each for required files.")
    """
    
    print(" - 2. Scanning for requied directory structure.")
    headVDir = os.path.dirname(os.path.abspath(__file__)) + "/VerificationFiles/"
    vFileDir = directory + "/Verification/"
    if not os.path.exists(vFileDir):
        print(" - 2. Creating Verification directory.")
        os.mkdir(vFileDir)
    if not os.path.exists(headVDir):
        print(" - 2. Creating Local Verification directory.")
        os.mkdir(headVDir)
    run_data = {}
    
    print(" - 2. Head Verification Directory: [" + headVDir+ "]")
    print(" - 2. Local Verification Directory: [" + vFileDir + "]")
    
    """
    My plotting code originally was going to have a shared plots folder, now I just have a separate make_verification_plots script to put them in the same folder.
    
    spFileDir = directory + "/SharedPlots/"
    if not os.path.exists(spFileDir):
        print(" - 2. Creating Shared Plots directory.")
        os.mkdir(spFileDir)        
    sDict["spFileDir"] = spFileDir
    """

    #for key, value in run_directories.items():
    #headDir = value["directory"]
    inputFiles = []
    tFiles = ["wrfout_[domain]_[time]", "subhr_[domain]_[time]", "pgrb3D_[domain]_[time]", "AFWA_[domain]_[time]"]

    oFileDir = directory + "/output/"   
    postDir = directory + "/postprd/"
    verfDir = directory + "/Verification/"
    
    if not os.path.exists(oFileDir):
        print(" - 2. Error: Could not locate output directory (" + oFileDir + "), did you target the correct directory?")
        sys.exit(0)

    sDict["outFileDir"] = oFileDir  
    sDict["postDir"] = postDir

    for sFile in tFiles:
        for domain in domainList:
            sFile = sFile.replace("[domain]", domain)
            sFile = sFile.replace("[time]", "*")
            path = oFileDir + sFile
            fList = sorted(glob.glob(path))
            inputFiles.extend(fList)
    
    for sFile in inputFiles:
        # Sanity check....
        if not os.path.exists(sFile):
            print("  - 2: Failed to locate file: " + str(sFile) + ", Abort.")
            sys.exit(0)
    if not os.path.exists(postDir):
        print("  - 2. Creating postprd directory.")
        os.mkdir(postDir) 
    if not os.path.exists(verfDir):
        print("  - 2. Creating verification directory.")
        os.mkdir(verfDir)         

    run_data["input_files"] = inputFiles
    print(" - 2. Done")
    
    print(" - 3. Constructing MPI Scripts (Initial Post-Processing)")
    print("   - 3a. Assembling Required Information")
     
    for domain in sDict["domains"]:
        constantsDict["wrf_bounds_" + str(domain)] = None
        constantsDict["timesteps_" + domain] = []
        wrfout = "subhr_[domain]_*"
        wrfout = wrfout.replace("[domain]", domain)
        woList = sorted(glob.glob(sDict["outFileDir"] + wrfout))
        for wrfFile in woList:
            timestep = wrfFile[-19:]
            TIMESTR = datetime.strptime(timestep, '%Y-%m-%d_%H_%M_%S')
            constantsDict["timesteps_" + domain].append(TIMESTR)        
    
    print("   - 3b. Finding geography information")

    if((int)(sDict["geo_seperate"]) == 1):
        print("  -- 3b. Pulling geography information from separate file.")
        geoFile = sDict["geo_file"]
        for domain in sDict["domains"]:
            tF = sDict["outFileDir"] + geoFile
            tF = tF.replace("[domain]", domain)
            geoNC = xr.open_dataset(tF, engine='netcdf4')              
            try:
                wrf_bounds = calculation.get_bounds(geoNC, "XLAT_M", "XLONG_M", wTime=True) 
                constantsDict["wrf_bounds_" + str(domain)] = wrf_bounds
            except KeyError:
                raise ValueError("XLAT_M / XLONG_M not found for " + domain)                
            geoNC.close()
    else:
        print("  -- 3b. Pulling geography information from wrf output files.")
        for domain in sDict["domains"]:
            tFiles = sorted(glob.glob(sDict["outFileDir"] + "wrfout_" + str(domain) + "*"))
            geoNC = xr.open_dataset(tFiles[0], engine='netcdf4')
            try:
                wrf_bounds = calculation.get_bounds(geoNC, "XLAT", "XLONG", wTime=True) 
                constantsDict["wrf_bounds_" + str(domain)] = wrf_bounds
            except KeyError:
                raise ValueError("XLAT / XLONG not found for " + domain)  
            geoNC.close()
    
    #era_client = cdsapi.Client()
    print("   - 3c. Preparing job scripts.")  
    print("   -> 3c. Checking if a GridRAD filtering job has been previously run")
    gridrad_no_data = []
    if os.path.exists(headVDir + "gridrad_nonexisting_files.txt"):
        print("   -> 3c. Found gridrad_nonexisting_files.txt, scanning...")
        with open(headVDir + "/gridrad_nonexisting_files.txt", "r") as grid_read:
            lineCount = 0
            for line in grid_read:
                if(len(line) > 0):
                    gridrad_no_data.append(datetime.strptime(line.strip(), "%Y%m%dT%H%M%S"))
                    lineCount += 1
        print("   -> 3c. " + str(lineCount) + " timesteps have been identified as non-data bearing")        
    for domain in domainList:
        for timestep in constantsDict["timesteps_" + domain]:           
            print("    -> 3c. Checking verification data for " + timestep.strftime("%Y-%m-%d_%H_%M_%S") + "...")
            with tools.cd(headVDir):
                if not timestep in gridrad_no_data:
                    fName1 = "gridrad_filtered_R_%Y%m%d_%H%M%S.nc"
                    fileTest1 = timestep.strftime(fName1)
                    fName2 = "gridrad_filtered_S_%Y%m%d_%H%M%S.nc"
                    fileTest2 = timestep.strftime(fName2)                    
                    if not os.path.exists(headVDir + fileTest1) and not os.path.exists(headVDir + fileTest2):
                        # If one of the filtered files is missing, we need to exit and alert the user to run a filtering script
                        print("    -> 3c. Missing filtered file for " + timestep.strftime("%Y-%m-%d_%H_%M_%S") + ", Constructing a job-script.")
                        jobFileName = os.path.dirname(os.path.realpath(__file__)) + "/" + timestep.strftime("filter_gridrad_%Y%m%d.job")
                        with open(jobFileName, 'w') as target_file:
                            target_file.write("#!/bin/bash\n" +
                                              "#PBS -N GridRad_Processing_" + timestep.strftime("%Y%m%d.job") + "\n" +
                                              "#PBS -A " + sDict["request_proj"] + "\n" +
                                              "#PBS -l nodes=" + sDict["gridrad_cores"] + ":ppn=" + sDict["gridrad_procs"] + "\n" + 
                                              "#PBS -l walltime=" + sDict["gridrad_walltime"] + "\n\n" + 
                                              "cd " + os.path.dirname(os.path.realpath(__file__)) + "\n\n"
                                              "export PYTHON=~/miniconda3/envs/wrf-work/bin/python\n\n"
                                              "mpiexec -machinefile $PBS_NODEFILE -n $PBS_NP $PYTHON process_gridrad.py -i \"" + directory + "\" -v \"" + headVDir + "\"")
                        
                        print("    -> 3c. Job script is constructed, please complete job script (" + jobFileName + "), then you may rerun this script.... exiting.")
                        tools.popen("chmod +x " + jobFileName)
                        sys.exit(1)
                """
                # PRISM - Replaced w/ Stage IV below
                if not os.path.exists(headVDir + "PRISM_ppt_" + timestep.strftime("%Y%m%d") + ".nc"):
                    fName = "PRISM_ppt_stable_4kmD2_%Y%m%d_bil.zip"
                    fileTest = timestep.strftime(fName)    
                    if not os.path.exists(headVDir + fileTest):
                        print("    -> 3c. Downloading PRISM (" + fileTest + ")")
                        tools.popen("wget --content-disposition http://services.nacse.org/prism/data/public/4km/ppt/" + timestep.strftime("%Y%m%d"))
                    # Unzip the file, convert to netCDF
                    with ZipFile(fileTest, 'r') as zipObj:
                        zipObj.extractall()
                        
                    gdalfName = "PRISM_ppt_stable_4kmD2_%Y%m%d_bil.bil"
                    gdalFile = timestep.strftime(gdalfName)
                        
                    gdal.GetDriverByName('EHdr').Register()
                    img = gdal.Open(gdalFile)

                    band = img.GetRasterBand(1)
                    geot = img.GetGeoTransform()
                    ncol = img.RasterXSize
                    nrow = img.RasterYSize

                    originX = geot[0]
                    originY = geot[3]
                    dX = geot[1]
                    dY = geot[5]

                    nodata = band.GetNoDataValue()
                    data = band.ReadAsArray()
                    data = np.flipud(data)
                    data = np.ma.masked_where(data == nodata, data)
                    
                    # Clean up files
                    img = None
                    gdal.Unlink(gdalFile)
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.bil")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.bil.aux.xml")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.hdr")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.info.txt")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.prj")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.stn.csv")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.stx")
                    tools.popen("rm PRISM_ppt_stable_4kmD2_" + timestep.strftime("%Y%m%d") + "_bil.xml")
                    
                    # Construct PRISM xarray > netCDF
                    xN = np.zeros([ncol]) 
                    yN = np.zeros([nrow])

                    for i in range(len(xN)):
                        xN[i] = originX + i * dX
                    for j in range(len(yN)):
                        yN[j] = originY + j * dY
                        
                    yN = np.flip(yN)
                    #X, Y = np.meshgrid(xN, yN)
                    xs = xr.Dataset(
                        coords = {
                            'prism_lon': xN,
                            'prism_lat': yN,
                        }
                    )
                    xs["precip"] = (['prism_lat', 'prism_lon'], data)     
                    xs.to_netcdf("PRISM_ppt_" + timestep.strftime("%Y%m%d") + ".nc")
                """
                
                # Stage (IV) 4
                #  As of right now, I could not find a way to automate downloads easily, so just do a quick file check for all the Stage IV files.
                if (int(timestep.strftime("%H")) % 6) == 0:
                    print("    -> 3c. Checking for Stage IV file")
                    if not os.path.exists("Stage4_" + timestep.strftime("%Y%m%d%H") + ".nc"):
                        print("    -> 3c. Error: Missing Required Stage IV Verification File (Stage4_" + timestep.strftime("%Y%m%d%H") + ".nc)")
                        sys.exit(1)

                # GridRAD Severe Mesh
                if not os.path.exists("GridRAD_Severe_Mesh_" + timestep.strftime("%Y%m%d") + ".nc"):
                    trgURL = timestep.strftime("http://weather.ou.edu/~chomeyer/GridRad_Severe_MESH/%Y/%Y%m%d.nc")
                    lFile = timestep.strftime("GridRAD_Severe_Mesh_%Y%m%d.nc")
                    if tools.url_exists(trgURL, None):
                        print("    -> 3c. Downloading GridRAD Severe Mesh (" + lFile + ")")
                        tools.popen("wget " + trgURL + " -O " + lFile)
                        
                # NARR (Theta-E) (Needs two files)
                narr1 = "shum." + timestep.strftime("%Y%m") + ".nc"
                narr2 = "air." + timestep.strftime("%Y%m") + ".nc"
                if not os.path.exists(narr1):
                    trgURL1 = timestep.strftime("https://downloads.psl.noaa.gov/Datasets/NARR/pressure/" + narr1)
                    if tools.url_exists(trgURL1, None):
                        print("    -> 3c. Downloading NARR Specific Humidity (" + narr1 + ")")
                        tools.popen("wget " + trgURL1)                    
                if not os.path.exists(narr2):
                    trgURL2 = timestep.strftime("https://downloads.psl.noaa.gov/Datasets/NARR/pressure/" + narr2)
                    if tools.url_exists(trgURL2, None):
                        print("    -> 3c. Downloading NARR Air Temperature (" + narr2 + ")")
                        tools.popen("wget " + trgURL2)                     
                    
                # Warning Polygons
                reqFiles = ["cpg", "csv", "dbf", "prj", "shp", "shx"]
                missingSvrPolygonFile = False
                for r in reqFiles:
                    fTest = timestep.strftime("wwa_%Y01010000_%Y12312359") + "." + r
                    if not os.path.exists(fTest):
                        missingSvrPolygonFile = True
                        break
                if missingSvrPolygonFile:
                    trgURL = timestep.strftime("https://mesonet.agron.iastate.edu/pickup/wwa/%Y_all.zip")
                    lFile = timestep.strftime("%Y_all.zip")
                    if tools.url_exists(trgURL, None):
                        print("    -> 3c. Downloading SPC Watch / Warning Polygons (" + trgURL + ")")
                        tools.popen("wget " + trgURL)     
                    with ZipFile(lFile, 'r') as zipObj:
                        zipObj.extractall() 

                # Local Storm Reports (LSRs)
                reqFiles = ["_rpts_hail", "_rpts_wind", "_rpts_torn"]
                dFile = timestep.strftime("%y%m%d")
                if(int(timestep.strftime("%H")) < 12):
                    # SPC LSRs are on a "convective day" IE: 12z - 12z, if our timestep is before 12Z, we need to get the data from the previous day.
                    prevDay = timestep - timedelta(days = 1)
                    dFile = prevDay.strftime("%y%m%d")
                for r in reqFiles:
                    fTest = dFile + r + ".csv"
                    trgURL = "http://www.spc.noaa.gov/climo/reports/" + fTest
                    if not os.path.exists(fTest):
                        tools.popen("wget " + trgURL)

        timestep_one = constantsDict["timesteps_" + domain][0]
    
        jobFileName = os.path.dirname(os.path.realpath(__file__)) + "/" + timestep_one.strftime("verification_process_%Y%m%d.job")
        print("    -> 3c. Constructing a job-script for verification.")
        
        with open(jobFileName, 'w') as target_file:
            target_file.write("#!/bin/bash\n" +
                              "#PBS -N WRF_Verification_" + timestep_one.strftime("%Y%m%d.job") + "\n" +
                              "#PBS -A " + sDict["request_proj"] + "\n" +
                              "#PBS -l nodes=" + sDict["verification_cores"] + ":ppn=" + sDict["verification_procs"] + "\n" + 
                              "#PBS -l walltime=" + sDict["verification_walltime"] + "\n\n" + 
                              "cd " + os.path.dirname(os.path.realpath(__file__)) + "\n\n"
                              "export PYTHON=~/miniconda3/envs/wrf-work/bin/python\n\n"
                              "mpiexec -machinefile $PBS_NODEFILE -n $PBS_NP $PYTHON process_verification.py -i \"" + directory + "\" -v \"" + headVDir + "\"")
        
        print("    -> 3c. Job script is constructed, please complete job script (" + jobFileName + ") to run verification.")
        tools.popen("chmod +x " + jobFileName)                
                
    print(" - 3. Done")

    print(" -  4. Checking for xESMF regridder instances")
    
    if not os.path.exists(headVDir + "/gridrad_files.txt"):
        print("    -> 4. Missing " + headVDir + "/gridrad_files.txt. Please re-run the gridrad processing script to generate this file... exiting.")
        sys.exit(1)
    gridrad_regular_timesteps = []
    gridrad_severe_timesteps = []
    with open(headVDir + "/gridrad_files.txt", "r") as grid_read:
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
    
    gridrad_severe_latlonbounds = {}
    gridrad_severe_bounds = {}
    for s in gridrad_severe_timesteps:
        gridrad_severe_bounds[s.strftime("%Y%m%d%H%M%S")] = None
    
    for domain in domainList:       

        timestep = gridrad_regular_timesteps[0]
        gridradF = xr.open_dataset(headVDir + "gridrad_filtered_R_" + timestep.strftime("%Y%m%d_%H%M%S") + ".nc", engine='netcdf4')
        constantsDict['gridrad_bounds'] = calculation.get_bounds(gridradF, "Latitude", "Longitude")  
        if not os.path.exists(headVDir + "conservative_GRIDRAD_" + str(domain) + ".nc"):
            print("   -- 4. GridRAD regridder must be built... proceeding")
            gridrad_regridder = xe.Regridder(constantsDict["wrf_bounds_" + str(domain)], constantsDict['gridrad_bounds'], 'conservative', filename=headVDir+"conservative_GRIDRAD_" + str(domain) + ".nc")               
        gridradF.close()

        # Check for existing regridders
        print("   -- 4. Scanning GridRAD Severe files to check domains")
        if os.path.exists(headVDir + "gridrad_severe_regridders.txt"):
            missing_regridders = False
            print("   -- 4. Found gridrad_severe_regridders.txt, scanning to ensure file validity")          
            with open(headVDir + "gridrad_severe_regridders.txt", "r") as grid_read:
                for line in grid_read:
                    line_splits = line.strip().split(";")
                    timestring = line_splits[0]
                    regridder_number = line_splits[1]
                    # For now, we just need to verify the file exists.
                    if not os.path.exists(headVDir + "conservative_GRIDRAD_severe" + regridder_number + "_" + str(domain) + ".nc"):
                        print("    -- 4. Error: Regridder conservative_GRIDRAD_severe" + regridder_number + "_" + str(domain) + ".nc not found. The gridrad_severe_regridders file has an error and is now deleted, please re-run this script.")
                        missing_regridders = True
                        break
            if missing_regridders:
                os.remove(headVDir + "gridrad_severe_regridders.txt")
                sys.exit(1)
        else:    
            boundsIdx = 1
            with open(headVDir + "gridrad_severe_regridders.txt", "w") as grid_write:
                for tSVR in gridrad_severe_timesteps:
                    gridradF = xr.open_dataset(headVDir + "gridrad_filtered_S_" + tSVR.strftime("%Y%m%d_%H%M%S") + ".nc", engine='netcdf4')
                    bounds = calculation.get_bounds(gridradF, "Latitude", "Longitude")
                    new_bounds = True
                    # First, see if it matches the gridrad (standard) box.
                    if (np.array_equal(bounds["lon"], constantsDict['gridrad_bounds']["lon"]) and np.array_equal(bounds["lat"], constantsDict['gridrad_bounds']["lat"])):
                        gridrad_severe_bounds[timestep.strftime("%Y%m%d%H%M%S")] = 0
                        grid_write.write(tSVR.strftime("%Y%m%d%H%M%S") + ";0\n")
                    else:
                        # Check the numpy arrays
                        for key, bound_dict in gridrad_severe_bounds.items():
                            if bound_dict is not None:
                                bounds_compare = gridrad_severe_latlonbounds[bound_dict]
                                if (np.array_equal(bounds["lon"], bounds_compare["lon"]) and np.array_equal(bounds["lat"], bounds_compare["lat"])):
                                    grid_write.write(tSVR.strftime("%Y%m%d%H%M%S") + ";" + str(bound_dict) + "\n")
                                    gridrad_severe_bounds[tSVR.strftime("%Y%m%d%H%M%S")] = bound_dict
                                    new_bounds = False
                                    break
                        if new_bounds:
                            # No matches, this is a new regridder
                            print("   -- 4. Timestep " + tSVR.strftime("%Y%m%d_%H%M%S") + "'s grid does not match existing grids, building a new regridder") 
                            new_regridder = xe.Regridder(constantsDict["wrf_bounds_" + str(domain)], bounds, 'conservative', filename=headVDir+"conservative_GRIDRAD_severe" + str(boundsIdx) + "_" + str(domain) + ".nc")
                            gridrad_severe_bounds[tSVR.strftime("%Y%m%d%H%M%S")] = boundsIdx
                            gridrad_severe_latlonbounds[boundsIdx] = bounds
                            grid_write.write(tSVR.strftime("%Y%m%d%H%M%S") + ";" + str(boundsIdx) + "\n")
                            boundsIdx += 1

        if not os.path.exists(headVDir + "conservative_NARR_" + str(domain) + ".nc"):
            print("   -- 4. NARR regridder must be built... proceeding")
            timestep = constantsDict["timesteps_" + domain][0]
            narrF = xr.open_dataset(headVDir + "air." + timestep.strftime("%Y%m") + ".nc", engine='netcdf4')
            constantsDict['narr_bounds'] = calculation.get_bounds(narrF, "lat", "lon")
            narr_regridder = xe.Regridder(constantsDict["wrf_bounds_" + str(domain)], constantsDict['narr_bounds'], 'conservative', filename=headVDir+"conservative_NARR_" + str(domain) + ".nc")

        """
        if not os.path.exists(vFileDir + "conservative_PRISM_" + str(domain) + ".nc"):
            print("   -- 4. PRISM regridder must be built... proceeding")
            timestep = constantsDict["timesteps_" + domain][0]
            prismF = xr.open_dataset(headVDir + "PRISM_ppt_" + timestep.strftime("%Y%m%d") + ".nc", engine='netcdf4')
            constantsDict['prism_bounds'] = calculation.get_bounds(prismF, "prism_lat", "prism_lon")
            prism_regridder = xe.Regridder(constantsDict["wrf_bounds_" + str(domain)], constantsDict['prism_bounds'], 'conservative', filename=vFileDir+"conservative_PRISM_" + str(domain) + ".nc")
        """
        
        if not os.path.exists(headVDir + "conservative_STAGE4_" + str(domain) + ".nc"):
            print("   -- 4. Stage IV regridder must be built... proceeding")
            timestep = constantsDict["timesteps_" + domain][0]
            st4F = xr.open_dataset(headVDir + "Stage4_" + timestep.strftime("%Y%m%d18") + ".nc", engine='netcdf4')
            constantsDict['stage4_bounds'] = calculation.get_bounds(st4F, "latitude", "longitude")
            st4_regridder = xe.Regridder(constantsDict["wrf_bounds_" + str(domain)], constantsDict['stage4_bounds'], 'conservative', filename=headVDir+"conservative_STAGE4_" + str(domain) + ".nc")

        for timestep in constantsDict["timesteps_" + domain]:
            tFileSvr = headVDir + timestep.strftime("conservative_GridRad_Severe_Mesh_%Y%m%d") + "_" + str(domain) + ".nc"
            meshFile = headVDir + "GridRAD_Severe_Mesh_" + timestep.strftime("%Y%m%d") + ".nc"
            if os.path.exists(meshFile):
                if not os.path.exists(tFileSvr):
                    print("   -- 4. GridRAD Severe Mesh for " + timestep.strftime("%Y-%m-%d") + " needs to be built.")
                    mesh = xr.open_dataset(meshFile, engine='netcdf4')
                    bounds = calculation.get_bounds(mesh, "Latitude", "Longitude")
                    mesh_regridder = xe.Regridder(constantsDict["wrf_bounds_" + str(domain)], bounds, 'conservative', filename=tFileSvr)
            
    print(" - 5. Assembling MPI Scripts (Plotting & Verification)")
    
    timestep_one = constantsDict["timesteps_" + domainList[0]][0]
    
    #for key, value in run_directories.items():
    #headDir = value["directory"]

    jobFileName = os.path.dirname(os.path.realpath(__file__)) + "/" + timestep_one.strftime("plotting_%Y%m%d.job")
    print("    -> 5. Constructing Plotting job script for " + directory + ".")
    
    with open(jobFileName, 'w') as target_file:
        target_file.write("#!/bin/bash\n" +
                          "#PBS -N WRF_Plotting_" + timestep_one.strftime("%Y%m%d.job") + "\n" +
                          "#PBS -A " + sDict["request_proj"] + "\n" +
                          "#PBS -l nodes=" + sDict["plotting_cores"] + ":ppn=" + sDict["plotting_procs"] + "\n" + 
                          "#PBS -l walltime=" + sDict["plotting_walltime"] + "\n\n" + 
                          "cd " + os.path.dirname(os.path.realpath(__file__)) + "\n\n"
                          "export PYTHON=~/miniconda3/envs/wrf-work/bin/python\n\n"
                          "mpiexec -machinefile $PBS_NODEFILE -n $PBS_NP $PYTHON plotting.py -i \"" + directory + "\"")
    
    print("    -> 5. Job script is constructed, please complete job script (" + jobFileName + ") to generate plots for " + directory + ".")
    tools.popen("chmod +x " + jobFileName)        

    jobFileName2 = os.path.dirname(os.path.realpath(__file__)) + "/" + timestep_one.strftime("calculation_%Y%m%d.job")
    print("    -> 5. Constructing Calculation job script for " + directory + ".")
    
    with open(jobFileName2, 'w') as target_file:
        target_file.write("#!/bin/bash\n" +
                          "#PBS -N WRF_PostCalc_" + timestep_one.strftime("%Y%m%d.job") + "\n" +
                          "#PBS -A " + sDict["request_proj"] + "\n" +
                          "#PBS -l nodes=" + sDict["calculation_cores"] + ":ppn=" + sDict["calculation_procs"] + "\n" + 
                          "#PBS -l walltime=" + sDict["calculation_walltime"] + "\n\n" + 
                          "cd " + os.path.dirname(os.path.realpath(__file__)) + "\n\n"
                          "export PYTHON=~/miniconda3/envs/wrf-work/bin/python\n\n"
                          "mpiexec -machinefile $PBS_NODEFILE -n $PBS_NP $PYTHON calculation.py -i \"" + directory + "\"")
    
    print("    -> 5. Job script is constructed, please complete job script (" + jobFileName2 + ") to generate calculation netCDF files for " + directory + " (Completing this job will allow the plotting job to make more plots).")
    tools.popen("chmod +x " + jobFileName2)      
        
    print("Script Complete.")
    return 0

def main(argv):
    inDir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["idir="])
    except getopt.GetoptError:
        print("Error: Usage: post_processing.py -i <inputdirectory>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Usage: post_processing.py -i <inputdirectory>")
            sys.exit()
        elif opt in ("-i", "--idir"):
            inDir = arg
         
    if inDir == '':
        print("Error: input directory not defined, exiting.")
        sys.exit()
    else:
        post_processing(inDir)

if __name__ == "__main__":
    main(sys.argv[1:])
    

"""
# ERA-5  
fName1 = "era5-pres-%Y%m%d.nc"
fName2 = "era5-slev-%Y%m%d.nc"
fileTest1 = timestep.strftime(fName1)
fileTest2 = timestep.strftime(fName2)
if not (os.path.exists(headVDir + fileTest1) or os.path.exists(headVDir + fileTest2)):
    print("    -> 3c. Downloading ERA-5 (" + fileTest1 + " / " + fileTest2 + ")")
    #RF NOTE: Need to download other parameters to calculate other variables desired.
    # e.g.: Theta, Theta-E, STP, etc. Talk to Victor about this...
    era_client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'year': timestep.strftime("%Y"),
            'month': timestep.strftime("%m"),
            'day': timestep.strftime("%d"),
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00',
                '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
                '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                51, -125, 20,
                -67,
            ],
        },
        fileTest1)
    era_client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['convective_inhibition', 'convective_available_potential_energy', '2m_dewpoint_temperature'],
            'year': timestep.strftime("%Y"),
            'month': timestep.strftime("%m"),
            'day': timestep.strftime("%d"),
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00',
                '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
                '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                51, -125, 20,
                -67,
            ],
        },                        
        fileTest2)
"""
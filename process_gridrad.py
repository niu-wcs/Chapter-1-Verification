#!/usr/bin/python
# process_gridrad.py
# Robert C Fritzen - Dpt. Geographic & Atmospheric Sciences
#
# Python script file which handles downloading and processing of GridRAD data
#
# This script file should not be called, but initialized from mpirun

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import sys
import os
from mpi4py import MPI
import glob
import getopt
import requests
import ast

from urllib.request import build_opener

import settings
import tools
import mpipool

def download_gridrad_file(file_command):
    splitCmd = file_command.split(";")
    
    vFileDir = splitCmd[0]
    filename = splitCmd[1]
    
    tools.loggedPrint.instance().write("Downloading " + str(filename))

    Yr = filename[15:19]
    Mo = filename[19:21]
    Da = filename[21:23]
    Hr = filename[24:26] 
    Mn = filename[26:28]
    Sc = filename[28:30]
    
    tStr = Yr + Mo + Da + Hr + Mn + Sc
    tStep = datetime.strptime(tStr, '%Y%m%d%H%M%S')    
    
    opener = build_opener()
    
    with tools.cd(vFileDir): 
        trgUrl = "https://data.rda.ucar.edu/ds841.0/" + tStep.strftime("%Y%m") + "/" + filename
        
        infile = opener.open(trgUrl)
        ofile = "nexrad_3d_v3_1_" + Yr + Mo + Da + "T" + Hr + Mn + Sc + "Z.nc"
        with open(ofile, "wb") as outfile:
            outfile.write(infile.read())        
        
    tools.loggedPrint.instance().write("File " + str(filename) + " saved.")
    
def download_gridrad_severe_file(file_command):
    splitCmd = file_command.split(";")
    
    vFileDir = splitCmd[0]
    filename = splitCmd[1]
    
    tools.loggedPrint.instance().write("Downloading " + str(filename))

    Yr = filename[15:19]
    Mo = filename[19:21]
    Da = filename[21:23]
    Hr = filename[24:26] 
    Mn = filename[26:28]
    Sc = filename[28:30]
    
    tStr = Yr + Mo + Da + Hr + Mn + Sc
    tStep = datetime.strptime(tStr, '%Y%m%d%H%M%S')
    
    try:
        extra = splitCmd[2]
        if(extra == "-1day"):
            tStep = tStep - timedelta(days=1)
    except IndexError:
        pass
    
    opener = build_opener()
    
    with tools.cd(vFileDir):
        trgUrl = "https://data.rda.ucar.edu/ds841.6/volumes/" + tStep.strftime("%Y") + "/" + tStep.strftime("%Y%m%d") + "/" + filename
        
        infile = opener.open(trgUrl)
        ofile = "nexrad_3d_v4_2_" + Yr + Mo + Da + "T" + Hr + Mn + Sc + "Z.nc"
        with open(ofile, "wb") as outfile:
            outfile.write(infile.read())          
        
    tools.loggedPrint.instance().write("File " + str(filename) + " saved.")    
    
def process_gridrad_file(file_command):
    splitCmd = file_command.split(";")
    
    vFileDir = splitCmd[0]
    filename = splitCmd[1]  
    
    Vr = filename[11]

    Yr = filename[15:19]
    Mo = filename[19:21]
    Da = filename[21:23]
    Hr = filename[24:26] 
    Mn = filename[26:28]
    Sc = filename[28:30]   
    
    if(int(Vr) == 3):
        gType = "R"
    else:
        gType = "S"
    
    tools.loggedPrint.instance().write("Processing " + str(filename))
    with tools.cd(vFileDir):
        GRID_RAD = xr.open_dataset(filename)
        # Fetch the required attributes / variables
        X = GRID_RAD["Longitude"].values
        Y = GRID_RAD["Latitude"].values
        Z = GRID_RAD["Altitude"].values

        lX = len(X)
        lY = len(Y)
        lZ = len(Z)

        total_grid_points = lX * lY * lZ

        IDX = GRID_RAD["index"]
        ref = np.zeros(total_grid_points)
        wRef = np.zeros(total_grid_points)
        ref[IDX] = GRID_RAD["Reflectivity"]
        wRef[IDX] = GRID_RAD["wReflectivity"]

        ref = ref.reshape(lZ, lY, lX)
        wRef = wRef.reshape(lZ, lY, lX)

        """
         Filtering
        """
        # Set fractional areal coverage threshold for speckle identification
        areal_coverage_thresh = 0.32

        # Copy altitude array to 3 dimensions
        zzz = ((Z.reshape(lZ, 1, 1)).repeat(lY, axis = 1)).repeat(lX, axis = 2)

        # First pass at removing speckles
        fin = np.isfinite(ref)

        # Compute fraction of neighboring points with echo
        cover = np.zeros((lZ, lY, lX))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
        cover = cover/25.0

        # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
        ibad = np.where(cover <= areal_coverage_thresh)
        nbad = len(ibad[0])
        if (nbad > 0): 
            ref[ibad] = float('nan')

        # Attempts to mitigate ground clutter and biological scatterers
        # First check for weak, low-level echo
        inan = np.where(np.isnan(ref)) # Find bins with NaNs 
        nnan = len(inan[0]) # Count number of bins with NaNs
        if (nnan > 0): 
            ref[inan] = 0.0

        # Find weak low-level echo and remove (set to NaN)
        ibad = np.where((ref < 10.0) & (zzz <= 4.0))
        nbad = len(ibad[0])
        if (nbad > 0):
            ref[ibad] = float('nan')

        # Replace NaNs that were removed
        if (nnan > 0): 
            ref[inan] = float('nan')

        # Second check for weak, low-level echo
        inan = np.where(np.isnan(ref)) # Find bins with NaNs 
        nnan = len(inan[0]) # Count number of bins with NaNs
        if (nnan > 0): 
            ref[inan] = 0.0

        refl_max   = np.nanmax(ref, axis=0)
        echo0_max  = np.nanmax((ref > 0.0)*zzz, axis=0)
        echo0_min  = np.nanmin((ref > 0.0)*zzz, axis=0)
        echo5_max  = np.nanmax((ref > 5.0)*zzz, axis=0)
        echo15_max = np.nanmax((ref > 15.0)*zzz, axis=0)

        # Replace NaNs that were removed
        if (nnan > 0): 
            ref[inan] = float('nan')

        # Find weak and/or shallow echo
        ibad = np.where(((refl_max < 20.0) & (echo0_max <= 4.0) & (echo0_min <= 3.0)) | \
                             ((refl_max < 10.0) & (echo0_max <= 5.0) & (echo0_min <= 3.0)) | \
                             ((echo5_max <= 5.0) & (echo5_max > 0.0) & (echo15_max <= 3.0)) | \
                             ((echo15_max < 2.0) & (echo15_max > 0.0)))

        nbad = len(ibad[0])
        if (nbad > 0):
            kbad = (np.zeros((nbad))).astype(int)
            for k in range(0, lZ):
                ref[(k+kbad),ibad[0],ibad[1]] = float('nan')

        # Find clutter below convective anvils
        k4km = ((np.where(ref >= 4.0))[0])[0]
        fin  = np.isfinite(ref)
        ibad = np.where((fin[k4km,:,:] == 0) & \
                        (np.sum(fin[k4km:(lZ-1),:,:], axis=0) > 0) & \
                        (np.sum(fin[0:(k4km-1),:,:], axis=0) > 0))
        nbad = len(ibad[0])
        if (nbad > 0):
            kbad = (np.zeros((nbad))).astype(int)
            for k in range(0,k4km+1):
                ref[(k+kbad),ibad[0],ibad[1]] = float('nan')

        # Second pass at removing speckles
        fin = np.isfinite(ref)

        # Compute fraction of neighboring points with echo
        cover = np.zeros((lZ, lY, lX))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
        cover = cover/25.0

        # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
        ibad = np.where(cover <= areal_coverage_thresh)
        nbad = len(ibad[0])
        if (nbad > 0): 
            ref[ibad] = float('nan')  
            
        ###
        year = int((GRID_RAD.attrs['Analysis_time'])[0:4])

        if gType == "R":
            wmin        = 0.1 # Set absolute minimum weight threshold for an observation (dimensionless)
            wthresh     = 1.33 - 1.0*(year < 2009) # Set default bin weight threshold for filtering by year (dimensionless)
            freq_thresh = 0.6 # Set echo frequency threshold (dimensionless)
            Z_H_thresh  = 18.5 # Reflectivity threshold (dBZ)
            nobs_thresh = 2 # Number of observations threshold
            
            echo_frequency = np.zeros((lZ, lY, lX)) # Create array to compute frequency of radar obs in grid volume with echo

            ipos = np.where(GRID_RAD['Nradobs'].values > 0) # Find bins with obs 
            npos = len(ipos[0]) # Count number of bins with obs

            if (npos > 0):
                echo_frequency[ipos] = (GRID_RAD['Nradecho'].values)[ipos] / (GRID_RAD['Nradobs'].values)[ipos] # Compute echo frequency (number of scans with echo out of total number of scans)

            inan = np.where(np.isnan(ref)) # Find bins with NaNs 
            nnan = len(inan[0]) # Count number of bins with NaNs

            if (nnan > 0): 
                ref[inan] = 0.0

            # Find observations with low weight
            ifilter = np.where((wRef < wmin) | ((wRef < wthresh) & (ref <= Z_H_thresh)) |
                            ((echo_frequency < freq_thresh) &  (GRID_RAD['Nradobs'].values > nobs_thresh)))

            nfilter = len(ifilter[0]) # Count number of bins that need to be removed

            # Remove low confidence observations
            if (nfilter > 0): 
                ref[ifilter] = float('nan')

            # Replace NaNs that were previously removed
            if (nnan > 0): 
                ref[inan] = float('nan')            
        else:
            wthresh     = 1.5												# Set default bin weight threshold for filtering by year (dimensionless)
            freq_thresh = 0.6												# Set echo frequency threshold (dimensionless)
            Z_H_thresh  = 15.0											# Reflectivity threshold (dBZ)
            nobs_thresh = 2												# Number of observations threshold            
            
            echo_frequency = np.zeros((lZ, lY, lX))					# Create array to compute frequency of radar obs in grid volume with echo

            ipos = np.where(GRID_RAD['Nradobs'] > 0)						# Find bins with obs 
            npos = len(ipos[0])											# Count number of bins with obs

            if (npos > 0):
                echo_frequency[ipos] = (GRID_RAD['Nradecho'].values)[ipos] / (GRID_RAD['Nradobs'].values)[ipos]		# Compute echo frequency (number of scans with echo out of total number of scans)

            inan = np.where(np.isnan(ref))				# Find bins with NaNs 
            nnan = len(inan[0])														# Count number of bins with NaNs
            
            if (nnan > 0):
                ref[inan] = 0.0

            # Find observations with low weight
            ifilter = np.where(((wRef < wthresh) & (ref < Z_H_thresh)) | ((echo_frequency < freq_thresh) & (GRID_RAD['Nradobs'] > nobs_thresh)))
            
            nfilter = len(ifilter[0])									# Count number of bins that need to be removed
            
            # Remove low confidence observations
            if (nfilter > 0):
                ref[ifilter] = float('nan')
            
            # Replace NaNs that were previously removed
            if (nnan > 0):
                ref[inan] = float('nan')
             
        tools.loggedPrint.instance().write("Processing of " + str(filename) + " completed, Saving file.")
        
        # Write the output...       
        xs = xr.Dataset(
            coords = {
                'Longitude': (['x'], GRID_RAD["Longitude"].values),
                'Latitude': (['y'], GRID_RAD["Latitude"].values),
                'Altitude': (['z'], GRID_RAD["Altitude"].values),
            },
            attrs = {
                'Analysis_time': GRID_RAD.attrs["Analysis_time"],
            }
        )
        xs["Reflectivity"] = (['z', 'y', 'x'], ref)

        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in xs.data_vars}

        xs.to_netcdf("gridrad_filtered_" + gType + "_" + Yr + Mo + Da + "_" + Hr + Mn + Sc + ".nc",
                        format='NETCDF4',
                        engine='netcdf4',
                        encoding=encoding)

    tools.loggedPrint.instance().write("File gridrad_filtered_" + gType + "_" + Yr + Mo + Da + "_" + Hr + Mn + Sc + ".nc saved.")
    return True
    
def run_script(tDirStr, vWriteDir): 
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
    """
    # Scan directory structure
    tools.loggedPrint.instance().write("Scanning directory structure, checking for list of run days")
    subDirs = glob.glob(tDir + "/Day*/")
    if(len(subDirs) == 0):
        tools.loggedPrint.instance().write("Error: Did not find Day subdirectories, did you target an individual day's directory?")
        sys.exit(0)
    run_directories = {}
    for iDay in subDirs:
        dayStr = iDay[iDay.rfind("Day"):-1]
        dayNumber = int(dayStr[3:])
        run_directories[dayNumber] = {'directory': iDay, 'day_string': dayStr}
    tools.loggedPrint.instance().write("Found " + str(len(run_directories)) + " directories following the naming convention.")    
    # Save some constants.
    first_key = list(run_directories.keys())[0]
    """
    aSet = settings.Settings()
    sDict = aSet.get_full_dict()
    domainList = sDict['domains']  

    timesteps = []
    existing_timesteps = []
    oFList = []    
    
    tDirList = tDirStr.split(";")
    for tDir in tDirList:   
        oFileDir = tDir + "/output/"
  
        # Scan for timesteps to process
        sFile = "subhr_[domain]_[time]"
    
        tools.loggedPrint.instance().write("Scanning for output files (" + str(tDir) + ").")
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
    tools.loggedPrint.instance().write("Timesteps:\n")
    for t in timesteps: tools.loggedPrint.instance().write(str(t))
    # Create a copy of the list that can be edited.
    undownloaded_timesteps = timesteps.copy()
    # Make a deep copy of the list so when we start stripping elements from it, the iterable is not disrupted.
    current_iterable = undownloaded_timesteps.copy() 
    # Fetch a list of files we need to get
    tools.loggedPrint.instance().write("Checking for GridRAD files that are needed...")
    missing_files = []
    # Search for gridrad severe files...
    with open(vWriteDir + "/gridrad_files.txt", "w") as timestep_writer:
        fName = "nexrad_3d_v4_2_%Y%m%dT%H%M%SZ.nc"
        timestep_writer.write("SEVERE\n")
        for t in current_iterable:
            # GridRAD severe works a bit differently, the "directory" denotes the start day, but timesteps proceed until the end
            #  of a convective event (For example, you can have timesteps from day+1 in day's directory), so instead of looping
            #  through the timesteps, we need to approach from a datetime perspective.
            fileTest = t.strftime(fName)
            tools.loggedPrint.instance().write("DEBUG: Testing " + fileTest)
            if not os.path.exists(vWriteDir + fileTest):
                # Check if the file exists on the gridrad severe archive.
                # https://data.rda.ucar.edu/ds841.6/volumes/2012/20120414/nexrad_3d_v4_2_20120414T133000Z.nc
                url = "https://data.rda.ucar.edu/ds841.6/volumes/" + t.strftime("%Y") + "/" + t.strftime("%Y%m%d") + "/" + fileTest       
                #tools.loggedPrint.instance().write("LOG: Checking if URL " + url + " exists...")
                if tools.url_exists(ur):
                    #tools.loggedPrint.instance().write("LOG: " + url + " found.")
                    missing_files.append(vWriteDir + ";" + fileTest))
                    undownloaded_timesteps.remove(t)
                    existing_timesteps.append("S" + t.strftime("%Y%m%dT%H%M%S"))
                    timestep_writer.write(t.strftime("%Y%m%dT%H%M%S") + "\n")
                else:
                    # This could potentially be a case where we need to roll back a day to see if this data are part of a past day's "event"
                    t_dayPrevious = t - timedelta(days=1)
                    urlNew = "https://data.rda.ucar.edu/ds841.6/volumes/" + t_dayPrevious.strftime("%Y") + "/" + t_dayPrevious.strftime("%Y%m%d") + "/" + fileTest
                    #tools.loggedPrint.instance().write("LOG: " + url + " Not found, testing day-1 (" + urlNew + ")...")
                    if tools.url_exists(urlNew):
                        missing_files.append(vWriteDir + ";" + fileTest + ";-1day")
                        undownloaded_timesteps.remove(t)
                        existing_timesteps.append("S" + t.strftime("%Y%m%dT%H%M%S"))
                        timestep_writer.write(t.strftime("%Y%m%dT%H%M%S") + "\n")
                    else:
                        tools.loggedPrint.instance().write("LOG: " + t.strftime("%Y%m%dT%H%M%S") + " not found in GridRad Severe")
            else:
                undownloaded_timesteps.remove(t)
                existing_timesteps.append("S" + t.strftime("%Y%m%dT%H%M%S"))
                timestep_writer.write(t.strftime("%Y%m%dT%H%M%S") + "\n")
        tools.loggedPrint.instance().write("There are " + str(len(missing_files)) + " gridrad severe files we need to download.")
        tools.loggedPrint.instance().write("DEBUG:")
        for m in missing_files:
            tools.loggedPrint.instance().write(m.split(";")[1])
        if(len(missing_files) > 0):
            tools.loggedPrint.instance().write("Downloading missing files.")
            pool.map_limit(download_gridrad_severe_file, missing_files, limits=10)    
        # Search for any remaining missing files...
        # https://data.rda.ucar.edu/ds841.0/201107/nexrad_3d_v3_1_20110701T000000Z.nc
        missing_files = []
        fName = "nexrad_3d_v3_1_%Y%m%dT%H%M%SZ.nc"
        timestep_writer.write("REGULAR\n")
        # Make a deep copy of the list so when we start stripping elements from it, the iterable is not disrupted.
        current_iterable = timesteps.copy()
        for t in current_iterable:
            fileTest = t.strftime(fName)
            if not os.path.exists(vWriteDir + fileTest):
                url = "https://data.rda.ucar.edu/ds841.0/" + t.strftime("%Y%m") + "/" + fileTest
                if tools.url_exists(url):
                    missing_files.append(vWriteDir + ";" + fileTest))
                    if t in undownloaded_timesteps:
                        undownloaded_timesteps.remove(t)
                    existing_timesteps.append("R" + t.strftime("%Y%m%dT%H%M%S"))
                    if "S" + t.strftime("%Y%m%dT%H%M%S") not in existing_timesteps:
                        timestep_writer.write(t.strftime("%Y%m%dT%H%M%S") + "\n")                  
                else:
                    tools.loggedPrint.instance().write("WARN: " + t.strftime("%Y%m%dT%H%M%S") + " not found (" + url + ").")
            else:
                if t in undownloaded_timesteps:
                    undownloaded_timesteps.remove(t)
                existing_timesteps.append("R" + t.strftime("%Y%m%dT%H%M%S"))
                if "S" + t.strftime("%Y%m%dT%H%M%S") not in existing_timesteps:
                    timestep_writer.write(t.strftime("%Y%m%dT%H%M%S") + "\n")
    tools.loggedPrint.instance().write("There are " + str(len(missing_files)) + " files we need to download.")
    tools.loggedPrint.instance().write("DEBUG:")
    for m in missing_files:
        tools.loggedPrint.instance().write(m.split(";")[1])    
    if(len(missing_files) > 0):
        tools.loggedPrint.instance().write("Downloading missing files.")
        pool.map_limit(download_gridrad_file, missing_files, limits=10)
    if(len(undownloaded_timesteps) > 0):
        tools.loggedPrint.instance().write("WARN: There are " + str(len(undownloaded_timesteps)) + " files that did not have gridrad data present.")
        with open(vWriteDir + "/gridrad_nonexisting_files.txt", "w") as no_timestep_writer:
            for m in undownloaded_timesteps:
                no_timestep_writer.write(m.strftime("%Y%m%dT%H%M%S") + "\n")
    # Once we have our verification data inventory, check for already processed files.
    tools.loggedPrint.instance().write("Scanning for files that need to have filtering completed on them.")
    tools.loggedPrint.instance().write("Analyzing " + str(len(existing_timesteps)) + " files.")
    unverified_files = []
    for tStr in existing_timesteps:
        gType = tStr[0]
        tStep = datetime.strptime(tStr[1:], "%Y%m%dT%H%M%S")
        sFile = "gridrad_filtered_" + gType + "_" + tStep.strftime("%Y%m%d_%H%M%S.nc")
        path = vWriteDir + sFile
        if not os.path.exists(path):
            if gType == "R":
                unverified_files.append(vWriteDir + ";" + tStep.strftime("nexrad_3d_v3_1_%Y%m%dT%H%M%SZ.nc"))
            elif gType == "S":
                unverified_files.append(vWriteDir + ";" + tStep.strftime("nexrad_3d_v4_2_%Y%m%dT%H%M%SZ.nc"))
            else:
                tools.loggedPrint.instance().write("WARN: Unknown gridrad file type \'" + gType + "\' for timestep " + tStep)
    tools.loggedPrint.instance().write("There are " + str(len(unverified_files)) + " files that must be processed...")
    # Finally, process any files that have yet to be finished.
    if(len(unverified_files) > 0):
        tools.loggedPrint.instance().write("Processing files.")
        pool.map(process_gridrad_file, unverified_files)
    # Clean up.
    pool.close()
    tools.loggedPrint.instance().write("Removing raw (unfiltered) files to save space.")
    tools.popen("rm " + vWriteDir + "/nexrad_3d_v3_1_*")
    tools.popen("rm " + vWriteDir + "/nexrad_3d_v4_2_*")
    tools.loggedPrint.instance().write("Program Complete.")
    return True
    
def main(argv):
    inDir = ''
    headVDir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:v:", ["idir="])
    except getopt.GetoptError:
        print("Error: Usage: process_gridrad.py -i <inputdirectory> -v <headVerificationDir>")
        print(" - inputDirectory allows for multiple directories, IE: dir1;dir2;dir3")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Usage: process_gridrad.py -i <inputDirectory> -v <headVerificationDir>")
            print(" - inputDirectory allows for multiple directories, IE: dir1;dir2;dir3")
            sys.exit()
        elif opt in ("-i", "--idir"):
            inDir = arg
        elif opt in ("-v"):
            headVDir = arg
         
    if inDir == '':
        print("Error: input directory not defined, exiting.")
        sys.exit()
    elif headVDir == '':
        print("Error: Verification head directory not defined, exiting.")
        sys.exit()
    else:  
        run_script(inDir, headVDir)
    
if __name__ == "__main__":
    main(sys.argv[1:])    
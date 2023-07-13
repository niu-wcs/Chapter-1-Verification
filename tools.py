#!/usr/bin/python
# tools.py
# Robert C Fritzen - Dpt. Geographic & Atmospheric Sciences
#
# Contains micro classes and methods used to help the program

import sys
import os
import os.path
import datetime
import subprocess
import time
import requests

#CD: Current Directory management, see https://stackoverflow.com/a/13197763/7537290 for implementation. This is used to maintain the overall OS CWD while allowing embedded changes.
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
		
#popen: A wrapped call to the subprocess.popen method to test for the debugging flag.
class popen:
    storing = False

    def __init__(self, command, storeOutput = True):
        self.storing = storeOutput
        
        if storeOutput:
            runCmd = subprocess.run(command, shell=True, capture_output=True, text=True)
            #runCmd.wait()
            stdout = runCmd.stdout
            stderr = runCmd.stderr
            self.stored = [stdout, stderr]
            #loggedPrint.instance().write("popen(" + command + "): " + self.stored[0] + ", " + self.stored[1])
        else:
            runcmd = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #loggedPrint.instance().write("popen(" + command + "): Command fired with no return")
            
    def fetch(self):
        if not self.storing:
            return None
        return self.stored
		
#singleton: Class decorator used to define classes as single instances across the program (See https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons)
class Singleton:
    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
		
@Singleton
class loggedPrint:
    f = None
    filePath = None

    def __init__(self):
        curTime = datetime.date.today().strftime("%B%d%Y")
        curDir = os.path.dirname(os.path.abspath(__file__)) 	
        logName = "post_script" + str(curTime) + ".log"	
        logFile = curDir + '/' + logName
        self.filePath = logFile

    def write(self, out):
        self.f = open(self.filePath, "a")
        self.f.write(out + '\n')
        self.f.close()
        print(out)

    def close(self):
        self.f.close()
        
def url_exists(url, cookies=None):
    r = requests.head(url, cookies=cookies)
    return r.status_code == requests.codes.ok
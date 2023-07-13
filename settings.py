#!/usr/bin/python
# settings.py
# Robert C Fritzen - Dpt. Geographic & Atmospheric Sciences
#
# Contains the class responsible for managing settings for the application

import datetime
import time
import os

# Settings: Class responsible for obtaining information from the control file and parsing it to classes that need the information
class Settings():
    settings = {}
    replacementKeys = {}
    myUserID = None

    def loadSettings(self):
        curDir = os.path.dirname(os.path.abspath(__file__))
        controlFile = curDir + "/control.txt"
        with open(controlFile) as f: 
            for line in f: 
                #To-Do: This can be simplified to a single if block, but for the time being, I'm going to leave it as is
                if line.split():
                    tokenized = line.split()
                    if(tokenized[0][0] != '#'):
                        if(tokenized[1].find("{") != -1):
                            #Array-like
                            inStr = tokenized[1]
                            insideSubStr = inStr[inStr.find("{")+1:inStr.find("}")]
                            levels = insideSubStr.split(",")
                            # Check for transformations
                            if(inStr.find("(") != -1):
                                transformType = inStr[inStr.find("(")+1:inStr.find(")")]
                                if(transformType == "int"):
                                    tType = int
                                elif(transformType == "float"):
                                    tType = float
                                else:
                                    raise Exception("Invalid transformType passed to loadSettings, " + transformType + " is not valid")
                                levels = list(map(tType, levels))
                            self.settings[tokenized[0]] = levels
                            #self.logger.write("Applying setting (" + tokenized[0] +", ARRAY-LIKE, TRANSFORM: " + transformType + "): " + str(levels))
                        else:
                            self.settings[tokenized[0]] = tokenized[1]
                            #self.logger.write("Applying setting (" + tokenized[0] +"): " + tokenized[1])
        #Test for program critical settings
        if(not self.settings):
            print("Settings(): Program critical variables missing, check for existence of control.txt, abort.")
            return False
        else:
            self.settings["headdir"] = curDir[:curDir.rfind('/')] + '/'
            return True
        
    def fetch(self, key):
        try:
            return self.settings[key]
        except KeyError:
            print("Key (" + str(key) + ") does not exist")
            return None    
            
    def add_replacementKey(self, key, value):
        self.replacementKeys[key] = value
        print("Settings(): Additional replacement key added: " + str(key) + " = " + str(value))
            
    def assembleKeys(self):	
        # General replacement keys
        #self.replacementKeys["[request_cores]"] = self.settings["request_cores"]
        #self.replacementKeys["[request_procs]"] = self.settings["request_procs"]
        return True
	 
    def replace(self, inStr):
        if not inStr:
            return inStr
        fStr = inStr
        for key, value in self.replacementKeys.items():
            kStr = str(key)
            vStr = str(value)
            fStr = fStr.replace(kStr, vStr)
        return fStr
        
    def whoami(self):
        return self.myUserID
        
    def get_full_dict(self):
        return self.settings        
     
    def __init__(self):	
        if(self.loadSettings() == False):
            print("Settings().init(): Cannot initialize settings class, missing control.txt file")
            sys.exit("Failed to load settings, please check for control.txt")
        
        self.myUserID = os.popen("whoami").read()

        self.assembleKeys()
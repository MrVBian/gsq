import numpy as np
from numpy import array as npa
import matplotlib.pyplot as plt

# ============================================================================ #
#                                Check V-REP lib                               #
# ============================================================================ #
try:
    from simulation import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')

# ============================================================================ #
#                               Connect to server                              #
# ============================================================================ #
print('Program started')
vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
if clientID != -1:
    print('Connected to remote API server')
else:
    print('Failed connecting to remote API server')

# ============================================================================ #
# simCreatePureShape(1, 0, {0.1, 0.1, 0.1}, 1)


emptyBuff = bytearray()
scriptDescription = 'remoteApiCommandServer'
functionName = 'simCreatePureShape'
print('clientID: ', clientID)
print('scriptDescription: ', scriptDescription)
print('functionName: ', functionName)
vrep.simxCallScriptFunction(clientID, scriptDescription, vrep.sim_scripttype_childscript,
                            functionName,
                            [],
                            [1, 0.1, 0.1, 0.1],
                            [],
                            emptyBuff,
                            vrep.simx_opmode_blocking)


# vrep.simxCallScriptFunction(clientID, '', 0,
#                             'simCloseScene',
#                             [],
#                             [],
#                             [],
#                             emptyBuff,
#                             vrep.simx_opmode_blocking)


# vrep.simxCallScriptFunction(clientID, "remoteApiCommandServer",
#                            vrep.sim_scripttype_childscript, 'executeCode_function',
#                            [], [], [code], emptyBuff, vrep.simx_opmode_blocking)

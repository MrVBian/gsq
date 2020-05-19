import os
import numpy as np
import utils
from simulation import vrep

import time
import cv2


class Robot(object):
    def __init__(self, connectIp, connectPort, objectMeshDir = '', objectNumber = 0, workspaceLimits = [[]]):

        ### Connect
        self.ip = connectIp
        self.port = connectPort

        ##### Object
        ### Define colors for object meshes (Tableau palette)
        self.colorSpace = np.asarray([[78.0, 121.0, 167.0], # blue
                                       [89.0, 161.0, 79.0], # green
                                       [156, 117, 95], # brown
                                       [242, 142, 43], # orange
                                       [237.0, 201.0, 72.0], # yellow
                                       [186, 176, 172], # gray
                                       [255.0, 87.0, 51.0], # red
                                       [176, 122, 161], # purple
                                       [118, 183, 178], # cyan
                                       [255, 157, 167], # pink
                                       [0, 0, 0], # black
                                       [0, 0, 255], # pure blue
                                       [0, 255, 0], # pure green
                                       [255, 0, 0], # pure red
                                       [0, 255, 255], # pure cyan
                                       [255, 255, 0], # pure yellow
                                       [255, 0, 255], # magenta
                                       [255, 255, 255], # white
                                      ])/255.0


        ### Read files in object mesh directory
        self.objectMeshDir = objectMeshDir
        self.objectNumber = objectNumber
        try:
            self.meshList = os.listdir(self.objectMeshDir)
            ### Randomly choose objects to add to scene
            self.objectMeshIndex = np.random.randint(0, len(self.meshList), size=self.objectNumber)
            self.objectMeshColor = self.colorSpace[np.asarray(range(self.objectNumber)) % len(self.colorSpace), :]
        except:
            self.meshList = None
            if objectNumber == 0:
                self.cerr("Failed to read mesh, detected objectNumber = 0, no need to read mesh, you can ignore this warning", 1)
            else:
                self.cerr("Failed to read mesh, please check the objectMeshDir", 0)

        ### Workspace limits
        self.workspaceLimits = workspaceLimits


    def cerr(self, context, type):
        if type == 0:
            print("\033[1;31mError\033[0m: %s"%context)
        elif type == 1:
            print("\033[1;33mWarning\033[0m: %s"%context)


    # connect
    def connect(self):

        vrep.simxFinish(-1)  # close all opened connections
        self.clientID = vrep.simxStart(self.ip, self.port, True, True, 3000, 5)
        ### Connect to V-REP
        if self.clientID == -1:
            import sys
            sys.exit('\nV-REP remote API server connection failed (' + self.ip + ':' +
                     str(self.port) + '). Is V-REP running?')
        else:
            print('V-REP remote API server connection success')
        return


    def disconnect(self):

        ### Make sure that the last command sent has arrived
        vrep.simxGetPingTime(self.clientID)
        ### Now close the connection to V-REP:
        vrep.simxFinish(self.clientID)
        return


    def start(self):

        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        print('''==========================================\n========== simulation start ==============\n==========================================\n''')


    def stop(self):

        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        print('''==========================================\n========== simulation stop ===============\n==========================================\n''')


    def restart(self):

        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        print('''==========================================\n========== simulation stop ===============\n==========================================\n''')
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        print('''==========================================\n========== simulation start ==============\n==========================================\n''')


    # undone test
    def getCameraData(self):

        simRet, resolution, rawImage = vrep.simxGetVisionSensorImage(self.clientID, self.cameraHandle, 0, vrep.simx_opmode_blocking)
        colorImg = np.asarray(rawImage)
        colorImg.shape = (resolution[1], resolution[0], 3)
        colorImg = colorImg.astype(np.float)/255
        colorImg[colorImg < 0] += 1
        colorImg *= 255
        colorImg = np.fliplr(colorImg)
        colorImg = colorImg.astype(np.uint8)

        ### Get depth image from simulation
        simRet, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.cameraHandle, vrep.simx_opmode_blocking)
        depthImg = np.asarray(depth_buffer)
        depthImg.shape = (resolution[1], resolution[0])
        depthImg = np.fliplr(depthImg)
        zNear = 0.01
        zFar = 10
        depthImg = depthImg * (zFar - zNear) + zNear

        ### cv2 BGR
        # img = colorImg[:,:, (2, 1, 0)]
        # cv2.imshow('img', img) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # cv2.imwrite('demo.png', img)

        ### cv2 BGR
        # img = colorImg[:,:, (2, 1, 0)]
        # depth = cv2.split(depthImg)[0]
        # depth[depth > 800] = 0
        # depth = depth / 1000.0
        # cv2.imshow('imgOri', depth*256) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return colorImg, depthImg


    ### undone test
    def setCamera(self, cameraName, cameraIntrinsics):

        ### Get handle to camera
        simRet, self.cameraHandle = vrep.simxGetObjectHandle(self.clientID, cameraName, vrep.simx_opmode_blocking)

        ### Get camera pose and intrinsics in simulation
        simRet, cameraPosition = vrep.simxGetObjectPosition(self.clientID, self.cameraHandle, -1, vrep.simx_opmode_blocking)
        simRet, cameraOrientation = vrep.simxGetObjectOrientation(self.clientID, self.cameraHandle, -1, vrep.simx_opmode_blocking)
        cameraTrans = np.eye(4, 4)
        cameraTrans[0:3,3] = np.asarray(cameraPosition)
        cameraOrientation = [-cameraOrientation[0], -cameraOrientation[1], -cameraOrientation[2]]
        cameraRotm = np.eye(4,4)
        cameraRotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cameraOrientation))

        ### Compute rigid transformation representating camera pose
        self.cameraPose= np.dot(cameraTrans, cameraRotm) 
        self.cameraIntrinsics = np.asarray(cameraIntrinsics)
        self.cameraDepthScale = 1

        ### Get background image
        self.colorImg, self.depthImg = self.getCameraData()
        self.depthImg = self.depthImg * self.cameraDepthScale


    ### add objects
    def addObjects(self, dropX=None, dropY=None, dropZ=None):

        self.objectHandles = []
        objHandles = []

        # the name of the scene object where the script is attached to, or an empty string if the script has no associated scene object
        scriptDescription = 'remoteApiCommandServer'
        # the name of the Lua function to call in the specified script
        functionName = 'importShape'

        for objIndex in range(len(self.objectMeshIndex)):
            currMeshFile = os.path.join(self.objectMeshDir, self.meshList[self.objectMeshIndex[objIndex]])
            currShapeName = 'shape_%02d' % objIndex

            if dropX == None:
                dropX = (self.workspaceLimits[0][1] - self.workspaceLimits[0][0] - 0.2) * np.random.random() + self.workspaceLimits[0][0] + 0.1
            if dropY == None:
                dropY = (self.workspaceLimits[1][1] - self.workspaceLimits[1][0] - 0.2) * np.random.random() + self.workspaceLimits[1][0] + 0.1
            if dropZ == None:
                dropZ = (self.workspaceLimits[2][1] - self.workspaceLimits[2][0] - 0.2) * np.random.random() + self.workspaceLimits[2][0] + 0.1

            objectPosition = [dropX, dropY, dropZ]
            objectOrientation = [2 * np.pi * np.random.random(), 2 * np.pi * np.random.random(), 2 * np.pi * np.random.random()]
            objectColor = [self.objectMeshColor[objIndex][0], self.objectMeshColor[objIndex][1], self.objectMeshColor[objIndex][2]]

            rootPath = '/home/bian/project/visual-pushing-grasping/'
            print('rootPath + currMeshFile: ', rootPath + currMeshFile)
            # print('currShapeName: ', currShapeName)
            retResp, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.clientID, scriptDescription, vrep.sim_scripttype_childscript, functionName,[0,0,255,0], objectPosition + objectOrientation + objectColor, [rootPath + currMeshFile, currShapeName], bytearray(), vrep.simx_opmode_blocking)

            if retResp == 8:
                self.cerr("Failed to add new objects to simulation. Please restart.", 0)
                self.stop()
                exit()

            currShapeHandle = retInts[0]
            self.objectHandles.append(currShapeHandle)

        prevObjectPositions = []
        objectPositions = []


    ### move and gripper
    def move(self, targetName, toolPosition, toolOrientation, moveSpeed=50, turnSpeed=50):

        simRet, targetHandle = vrep.simxGetObjectHandle(self.clientID, targetName, vrep.simx_opmode_blocking)
        simRet, targetPosition = vrep.simxGetObjectPosition(self.clientID, targetHandle,-1,vrep.simx_opmode_blocking)
        simRet, targetOrientation = vrep.simxGetObjectOrientation(self.clientID, targetHandle,-1,vrep.simx_opmode_blocking)

        if toolPosition != None:
            moveDirection = np.asarray([toolPosition[0] - targetPosition[0], toolPosition[1] - targetPosition[1], toolPosition[2] - targetPosition[2]])
            moveMagnitude = np.linalg.norm(moveDirection)
            moveStep = 1 / moveSpeed * moveDirection / moveMagnitude
            moveStepNumber = int(np.floor(moveMagnitude * moveSpeed))
            # moveStep = 0.02 * moveDirection / moveMagnitude
            # moveStepNumber = int(np.floor(moveMagnitude / 0.02))

            print('moveStepNumber: ', moveStepNumber)
            for stepIter in range(moveStepNumber):
                ### move
                vrep.simxSetObjectPosition(self.clientID, targetHandle, -1, (targetPosition[0] + moveStep[0], targetPosition[1] + moveStep[1], targetPosition[2] + moveStep[2]), vrep.simx_opmode_blocking)
                simRet, targetPosition = vrep.simxGetObjectPosition(self.clientID, targetHandle, -1, vrep.simx_opmode_blocking)

            vrep.simxSetObjectPosition(self.clientID, targetHandle, -1, (toolPosition[0], toolPosition[1], toolPosition[2]), vrep.simx_opmode_blocking)

        if toolOrientation != None:
            turnDirection = np.asarray([toolOrientation[0] - targetOrientation[0], toolOrientation[1] - targetOrientation[1], toolOrientation[2] - targetOrientation[2]])
            turnMagnitude = np.linalg.norm(turnDirection)
            turnStep = 1 / turnSpeed * turnDirection / turnMagnitude
            turnStepNumber = int(np.floor(turnMagnitude * turnSpeed))

            print('turnStepNumber: ', turnStepNumber)
            for stepIter in range(turnStepNumber):
                ### turn
                vrep.simxSetObjectOrientation(self.clientID, targetHandle, -1, (targetOrientation[0] + turnStep[0], targetOrientation[1] + turnStep[1], targetOrientation[2] + turnStep[2]), vrep.simx_opmode_blocking)
                simRet, targetOrientation = vrep.simxGetObjectOrientation(self.clientID, targetHandle, -1, vrep.simx_opmode_blocking)

            vrep.simxSetObjectOrientation(self.clientID, targetHandle, -1, (toolOrientation[0], toolOrientation[1], toolOrientation[2]), vrep.simx_opmode_blocking)


    def openGripper(self, openCloseJointName, gripperMotorVelocity=0.0, gripperMotorForce=0.0):

        print('openJointName: ', openCloseJointName)
        print('gripperMotorVelocity: ', gripperMotorVelocity)
        print('gripperMotorForce: ', gripperMotorForce)
        # gripperMotorVelocity = 0
        # gripperMotorForce = 0
        simRet, RG2GripperHandle = vrep.simxGetObjectHandle(self.clientID, openCloseJointName, vrep.simx_opmode_blocking)
        simRet, gripperJointPosition = vrep.simxGetJointPosition(self.clientID, RG2GripperHandle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.clientID, RG2GripperHandle, gripperMotorForce, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.clientID, RG2GripperHandle, gripperMotorVelocity, vrep.simx_opmode_blocking)
        while gripperJointPosition < 0.03: # Block until gripper is fully open
            simRet, gripperJointPosition = vrep.simxGetJointPosition(self.clientID, RG2GripperHandle, vrep.simx_opmode_blocking)
            print('gripperJointPosition: ', gripperJointPosition)


    def closeGripper(self, openCloseJointName, gripperMotorVelocity=0.0, gripperMotorForce=0.0):

        print('closeJointName: ', openCloseJointName)
        print('gripperMotorVelocity: ', gripperMotorVelocity)
        print('gripperMotorForce: ', gripperMotorForce)
        simRet, RG2GripperHandle = vrep.simxGetObjectHandle(self.clientID, openCloseJointName, vrep.simx_opmode_blocking)
        simRet, gripperJointPosition = vrep.simxGetJointPosition(self.clientID, RG2GripperHandle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.clientID, RG2GripperHandle, gripperMotorForce, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.clientID, RG2GripperHandle, gripperMotorVelocity, vrep.simx_opmode_blocking)
        gripperFullyClosed = False
        while gripperJointPosition > -0.045: # Block until gripper is fully closed
            simRet, newGripperJointPosition = vrep.simxGetJointPosition(self.clientID, RG2GripperHandle, vrep.simx_opmode_blocking)
            print('gripperJointPosition: ', gripperJointPosition)
            if newGripperJointPosition >= gripperJointPosition:
                return gripperFullyClosed
            gripperJointPosition = newGripperJointPosition
        gripperFullyClosed = True


### connect demo
connectIp = '127.0.0.1'
connectPort = 19997

objectMeshDir = 'objects/blocks'
objectNumber = 10
# Cols: min max, Rows: x y z (define workspace limits in robot coordinate)
workspaceLimits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])

robot = Robot(connectIp, connectPort, objectMeshDir, objectNumber, workspaceLimits)
robot.connect()


### start stop restart demo
robot.start()
# robot.stop()
# robot.restart()


### setCameraParameter demo
cameraName = 'Vision_sensor_persp'
cameraIntrinsics = [[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]]
robot.setCamera(cameraName, cameraIntrinsics)


### add object demo
# robot.addObjects()


### move and gripper demo
openGripperMotorVelocity = 5
openGripperMotorForce = 20
closeOripperMotorVelocity = -0.5
closeGripperMotorForce = 20 

targetName = 'UR5_target'
RG2OpenCloseJointName = 'RG2_openCloseJoint'
speed = 0.02


robot.move(targetName, [-0.625, 0.075, 0.4], None)
robot.move(targetName, [-0.3, -0.2, 0.4], None)
robot.openGripper(RG2OpenCloseJointName, openGripperMotorVelocity, openGripperMotorForce)
robot.move(targetName, [-0.3, -0.2, 0.02], None)
robot.closeGripper(RG2OpenCloseJointName, closeOripperMotorVelocity, closeGripperMotorForce)
robot.move(targetName, [-0.3, -0.2, 0.4], None)
robot.move(targetName, [-0.5, 0, 0.4], None)
robot.move(targetName, [-0.5, 0, 0.02], None)
robot.openGripper(RG2OpenCloseJointName, openGripperMotorVelocity, openGripperMotorForce)
robot.move(targetName, [-0.5, 0, 0.4], None)
robot.closeGripper(RG2OpenCloseJointName, closeOripperMotorVelocity, closeGripperMotorForce)
robot.move(targetName, None, [-0.018, 0.262, 0.012])



### end demo
robot.stop()






# -----------------------------------------------------------------------------
# FILE NAME:         proj1.py
# USAGE:             python3 proj1.py
# NOTES:             Requires NumPy installation
#                    Requires Python3
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon       03-02-2021     1.0 ... setup dev environment, imported NumPy
# Andy Alarcon       03-03-2021     1.1 ... implemented KFreading class & additional matrices
# Andy Alarcon       03-04-2021     1.2 ... Added file input, output and calculations for KF
# Andy Alarcon       03-05-2021     1.3 ... Corrected IMU calibration
# Andy Alarcon       03-06-2021     1.4 ... Added an offset calc for periods of noise
# Andy Alarcon       03-08-2021     1.4 ... Adjusted comments
# -----------------------------------------------------------------------------

import numpy.matlib as m
import numpy as np
import math
import random


def main():

    # Data set read from file
    KFReadings = ReadCommandsFileInput()

    #For all KF readings
    for i in range(len(KFReadings)):
        
        # Calculate Xpred and Pred
        PredictionStage(KFReadings[i])
        # Calculate Kalaman Gain
        CalculateKalmanGain(KFReadings[i])
        # Correction Stage
        CorrectionStage(KFReadings[i])
        #Write current KF results
        WriteDataToFile(KFReadings[i])

        #Set current proccess and state martix to inital for the next KF reading
        if i+1 < len(KFReadings) :
            KFReadings[i+1].matrix_initalX = KFReadings[i].matrix_CurrX
            KFReadings[i+1].matrix_initalP = KFReadings[i].matrix_CurrP

    
    print("Kalman Filter Data Written to Output.txt")
    

 # ----------------------------------------------------------------------------
# FUNCTION NAME:     WriteDataToFile()
# PURPOSE:           writes the x, y, and orientation of the current KF reading
# -----------------------------------------------------------------------------


def WriteDataToFile(KFreading):

    x = np.asarray(KFreading.matrix_CurrX)
    f = open("output.txt", "a+")
    f.write(str(x[0][0]) + '|' + str(x[1][0]) + '|' + str(x[3][0]) + '\n')
    
    f.close() 
  

# ----------------------------------------------------------------------------
# FUNCTION NAME:     CorrectionStage()
# PURPOSE:           This function performs the correction stage
# -----------------------------------------------------------------------------


def CorrectionStage(KFreading):

    #Comput Y mat = H * Xkp
    y_mat = np.dot(KFreading.matrix_H, KFreading.matrix_PredX)

    # tempf_diff = Z - Y
    temp_diff = np.subtract(KFreading.matrix_Z , y_mat)

    #rightS = K[Z-Y]
    rightS = np.dot(KFreading.matrix_K, temp_diff)

    #Calculate current state = Xkp + K[Z-Y]
    KFreading.matrix_CurrX = np.add(KFreading.matrix_PredX, rightS)

    #Update proccess matrix, KH = H * Pkp
    KH_mat = np.dot(KFreading.matrix_H, KFreading.matrix_PredP)
    #KH = K * (H * Pkp)
    KH_mat = np.dot(KFreading.matrix_K, KH_mat)

    #Substract Pkp - K(H*Pkp)
    KFreading.matrix_CurrP = np.subtract(KFreading.matrix_PredP, KH_mat)



# ----------------------------------------------------------------------------
# FUNCTION NAME:     CalculateKalmanGain()
# PURPOSE:           This performs calculates the Kalman gain
# -----------------------------------------------------------------------------


def CalculateKalmanGain(KFreading):

    # Pkp*H^T
    numerator = np.dot(KFreading.matrix_PredP, KFreading.matrix_TransH)

    # Pkp*H^T*H
    denominator = np.dot(numerator, KFreading.matrix_H)

    #Pkp*H^T*H + R
    denominator = np.add(denominator, KFreading.matrix_R)

    #[Pkp*H^T*H + R]^-1
    inv_denominator = np.linalg.inv(denominator)

    # K = Pkp*H^T   *   [Pkp*H^T*H + R]^-1
    k_mat = np.dot(numerator, inv_denominator)

    KFreading.matrix_K = k_mat


# ----------------------------------------------------------------------------
# FUNCTION NAME:     PredictionStage()
# PURPOSE:           This performs the prediction stage for one KFreading
# -----------------------------------------------------------------------------


def PredictionStage(KFreading):


    KFreading.matrix_PredX = np.dot(
        KFreading.matrix_A, KFreading.matrix_initalX)


    # Predict proccess matrix
    leftS = np.dot(KFreading.matrix_A, KFreading.matrix_initalP)
    leftS = np.dot(leftS, KFreading.matrix_TransA)

    KFreading.matrix_PredP = np.add(leftS, KFreading.matrix_QNoise)



# ----------------------------------------------------------------------------
# FUNCTION NAME:     ReadFileInput()
# PURPOSE:           This function reads the data file input and 
#                    returns a list of the data at each time
# -----------------------------------------------------------------------------


def ReadCommandsFileInput():
    KFFileReadings = []

    # Read the lines from the file
    file1 = open('EKF_DATA_circle.txt', 'r')
    Lines = file1.readlines()[1:]

    # Strips each newline character and create a KF object
    for line in Lines:
        line = line.strip()
        x = line.split(",")
        newReading = KFReading(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]),
                               float(x[6]), float(x[7]), float(x[8]), float(x[9]))
        KFFileReadings.append(newReading)

    file1.close()

    return KFFileReadings


#This class is created for every line of data
#Represented as a single KF reading
class KFReading:
    def __init__(self, time, odo_x, odo_y, odo_o, imu_o, imu_cov, gps_x, gps_y, gps_covX, gps_covY):
       #Init variables
        self.time = time
        self.delta_time = 0.001
        self.odo_x = odo_x
        self.odo_y = odo_y
        self.odo_o = odo_o
        self.imu_o = imu_o + (0.32981-0.237156)
        self.imu_cov = imu_cov
        self.gps_x = gps_x
        self.gps_y = gps_y
        self.gps_covX = gps_covX
        self.gps_covY = gps_covY

        #Init additional matrices
        self.matrix_PredX = np.identity(5)
        self.matrix_CurrX = np.identity(5)
        self.matrix_PredP = np.identity(5)
        self.matrix_CurrP = np.identity(5)
        self.matrix_K = np.identity(5)

        # Predicted state equation
        # ---------------------------------------
       
        # Create Matrix A
        cos_theta = np.cos(self.odo_o)
        sin_theta = np.sin(self.odo_o)
        A_mat = m.matrix([[1, 0, self.delta_time*cos_theta, 0, 0],
                          [0, 1, self.delta_time*sin_theta, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, self.delta_time],
                          [0, 0, 0, 0, 1]])
        self.matrix_A = A_mat

        # Create inital Xstate matrx
        self.velocity = 0.44
        tan_theta = np.around(np.tan(self.odo_o), decimals=5)
        self.wvelocity = self.velocity * (tan_theta / 1)
        X_mat = m.matrix([[self.odo_x], [self.odo_y], [self.velocity], [
                         self.odo_o], [self.wvelocity]])
        self.matrix_initalX = X_mat

        # Predicted proccess equation
        # ---------------------------------------

        # Create initial proccess noise covariance matrix
        QN_mat = m.matrix([[.00004, 0, 0, 0, 0],
                           [0, .00004, 0, 0, 0],
                           [0, 0, .0001, 0, 0],
                           [0, 0, 0, .0001, 0],
                           [0, 0, 0, 0, .0001]])
        self.matrix_QNoise = QN_mat

        # Create transpose of A matrix
        self.matrix_TransA = np.transpose(self.matrix_A)

        # Inital covariance matrix P
        PIN_mat = m.matrix([[.01, 0, 0, 0, 0],
                            [0, .01, 0, 0, 0],
                            [0, 0, .01, 0, 0],
                            [0, 0, 0, .01, 0],
                            [0, 0, 0, 0, .01]])
        self.matrix_initalP = PIN_mat

        # Kalman Gain equation
        # ---------------------------------------
        self.matrix_H = np.identity(5)
        self.matrix_TransH = np.transpose(self.matrix_H)
        
        #For certain periods add noise (offset)
        if (self.time >= 500 and self.time <=2000) or (self.time >= 3000):
            offset = random.uniform(-2,2)
        else :
            offset = 0

        # Measurement Error covariance matrix R
        R_mat = m.matrix([[self.gps_covX+0.001, 0, 0, 0, 0],
                          [0, self.gps_covY+0.002, 0, 0, 0],
                          [0, 0, .01, 0, 0],
                          [0, 0, 0, self.imu_cov, 0],
                          [0, 0, 0, 0, .01]])
        self.matrix_R = R_mat

        # Inital measurement error covariance matrix R
        IR_mat = m.matrix([[.04, 0, 0, 0, 0],
                           [0, .04, 0, 0, 0],
                           [0, 0, .01, 0, 0],
                           [0, 0, 0, .01, 0],
                           [0, 0, 0, 0, .01]])
        self.matrix_initalR = IR_mat

        # Correction Stage
        # ---------------------------------------

        # Sensor measurements
        Z_mat = m.matrix([[self.gps_x+offset], [self.gps_y+offset], [
                         self.velocity], [self.imu_o], [self.wvelocity]])
        self.matrix_Z = Z_mat

    


if __name__ == "__main__":
    main()

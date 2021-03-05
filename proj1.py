# -----------------------------------------------------------------------------
# FILE NAME:
# USAGE:             python3
# NOTES:             Run
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon       03-02-2021     1.0 ... setup dev environment, imported NumPy
# Andy Alarcon       03-03-2021     1.1 ... implemented KFreading class & additional matrices
# -----------------------------------------------------------------------------

import numpy.matlib as m
import numpy as np
import math


def main():

    # Data set read from file
    KFReadings = ReadCommandsFileInput()

    #testKF = KFReading(1, -1.9512e-65, 0, 0.001225,
                      # 8.82147e-199, 2.96439e-322, -1.89933e-65, 0, 0, 0)

    
    for i in range(len(KFReadings)):
        
        print("Iteration = " , i)
        print("Inital X State : ")
        print(KFReadings[i].matrix_initalX)

        print("Inital P State : ")
        print(KFReadings[i].matrix_initalP)

        # Calculate Xpred and Pred
        PredictionStage(KFReadings[i])
        # Calculate Kalaman Gain
        CalculateKalmanGain(KFReadings[i])
        # Correction Stage
        CorrectionStage(KFReadings[i])

        WriteDataToFile(KFReadings[i])

        #Set current to inital of next reading
        if i+1 < len(KFReadings) :
            KFReadings[i+1].matrix_initalX = KFReadings[i].matrix_CurrX
            KFReadings[i+1].matrix_initalP = KFReadings[i].matrix_CurrP

        print("-----------------------------------------------------------")




	    

    # # Calculate Xpred and Pred
    # PredictionStage(testKF)
    # # Calculate Kalaman Gain
    # CalculateKalmanGain(testKF)
    # CorrectionStage(testKF)

    #Setup next

 # ----------------------------------------------------------------------------
# FUNCTION NAME:     WriteDataToFile()
# PURPOSE:           writes
# -----------------------------------------------------------------------------


def WriteDataToFile(KFreading):

    x = np.asarray(KFreading.matrix_CurrX)
    f = open("output.txt", "a+")
    f.write(str(x[0][0]) + '|' + str(x[1][0]) + '|' + str(x[3][0]) + '\n')
    f.close() 

    # x = np.asarray(KFreading.matrix_CurrX)
    # print("Array :")
    # print(str(x[0][0]) + '|' + str(x[1][0]) + '|' + str(x[3][0]) + '\n')
  

# ----------------------------------------------------------------------------
# FUNCTION NAME:     CorrectionStage()
# PURPOSE:           This performs the correction stage
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
    print("New State : ")  
    print(KFreading.matrix_CurrX)

    #Update proccess matrix, KH = H * Pkp
    KH_mat = np.dot(KFreading.matrix_H, KFreading.matrix_PredP)
    #KH = K * (H * Pkp)
    KH_mat = np.dot(KFreading.matrix_K, KH_mat)

    #Substract Pkp - K(H*Pkp)
    KFreading.matrix_CurrP = np.subtract(KFreading.matrix_PredP, KH_mat)
    print("New Proccess : ")  
    print(KFreading.matrix_CurrP)



# ----------------------------------------------------------------------------
# FUNCTION NAME:     CalculateKalmanGain()
# PURPOSE:           This performs calculate the Kalman gain
# -----------------------------------------------------------------------------


def CalculateKalmanGain(KFreading):

    # Pkp*H^T
    numerator = np.dot(KFreading.matrix_PredP, KFreading.matrix_TransH)

    # Pkp*H^T*H
    denominator = np.dot(numerator, KFreading.matrix_H)
#    print("Denominator Matrix : ")
#    print(denominator)
#    print("R Matrix : ")
#    print(KFreading.matrix_R)
#    print("Sum of the Denominator : ")

    #Pkp*H^T*H + R
    denominator = np.add(denominator, KFreading.matrix_R)
    # print(denominator)

#    print("inv denominator : ")

    #[Pkp*H^T*H + R]^-1
    inv_denominator = np.linalg.inv(denominator)
#    print(inv_denominator)
#    print("Numerator Matrix : ")
#    print(numerator)
#    print("Dot  : ")

    # K = Pkp*H^T   *   [Pkp*H^T*H + R]^-1
    k_mat = np.dot(numerator, inv_denominator)
#    print(k_mat)
    KFreading.matrix_K = k_mat


# ----------------------------------------------------------------------------
# FUNCTION NAME:     PredictionStage()
# PURPOSE:           This performs the prediction stage for one KFreading
# -----------------------------------------------------------------------------


def PredictionStage(KFreading):

    # Predict state
    #    print("Matrix A : ")
    #    print(KFreading.matrix_A)
    #    print("Inital Matrix X : ")
    #    print(KFreading.matrix_initalX)
    KFreading.matrix_PredX = np.dot(
        KFreading.matrix_A, KFreading.matrix_initalX)
#    print("Matrix A x X : ")
#    print(KFreading.matrix_PredX)

    # Predict proccess matrix
#    print("Inital Matrix P : ")
#    print(KFreading.matrix_initalP)
    leftS = np.dot(KFreading.matrix_A, KFreading.matrix_initalP)
    leftS = np.dot(leftS, KFreading.matrix_TransA)
#    print("Left Matrix : ")
#    print(leftS)
    KFreading.matrix_PredP = np.add(leftS, KFreading.matrix_QNoise)
#    print("Noise Matrix : ")
#    print(KFreading.matrix_QNoise)
#    print("Prediction P matrix : ")
#    print(KFreading.matrix_PredP)


# ----------------------------------------------------------------------------
# FUNCTION NAME:     ReadFileInput()
# PURPOSE:           This function reads the data file input and returns a list of
#                    the data at each time
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


class KFReading:
    def __init__(self, time, odo_x, odo_y, odo_o, imu_o, imu_cov, gps_x, gps_y, gps_covX, gps_covY):
        self.time = time
        self.delta_time = 0.001
        self.odo_x = odo_x
        self.odo_y = odo_y
        self.odo_o = odo_o
        self.imu_o = imu_o
        self.imu_cov = imu_cov
        self.gps_x = gps_x
        self.gps_y = gps_y
        self.gps_covX = gps_covX
        self.gps_covY = gps_covY

        # additional matrices
        self.matrix_PredX = np.identity(5)
        self.matrix_CurrX = np.identity(5)
        self.matrix_PredP = np.identity(5)
        self.matrix_CurrP = np.identity(5)
        self.matrix_K = np.identity(5)

        # Predicted state equation
        # ----------------------------------------------------------------------------------------------
        # Create Matrix A
        #cos_theta = np.around(np.cos(self.odo_o), decimals=5)
        #sin_theta = np.around(np.sin(self.odo_o), decimals=5)
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
        # ----------------------------------------------------------------------------------------------

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
        # --------------------------------------------------------------------------------------------
        self.matrix_H = np.identity(5)
        self.matrix_TransH = np.transpose(self.matrix_H)

        # Measurement Error covariance matrix R
        R_mat = m.matrix([[self.gps_covX, 0, 0, 0, 0],
                          [0, self.gps_covY, 0, 0, 0],
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
        # --------------------------------------------------------------------------------------------

        # Sensor measurements
        Z_mat = m.matrix([[self.gps_x], [self.gps_covY], [
                         self.velocity], [self.imu_o], [self.wvelocity]])
        self.matrix_Z = Z_mat


if __name__ == "__main__":
    main()

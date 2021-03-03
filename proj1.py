# -----------------------------------------------------------------------------
# FILE NAME:
# USAGE:             python3
# NOTES:             Run
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon
# -----------------------------------------------------------------------------

import numpy.matlib as m
import numpy as np
import math


def main():
    testKF = KFReading(1,-1.9512e-65,0,0.001225,8.82147e-199,2.96439e-322,-1.89933e-65,0,0,0)
    print(testKF.matrixA)
    print(testKF.matrixX)
    




class KFReading:
    def __init__(self, time, odo_x, odo_y, odo_o, imu_o, imu_cov, gps_x, gps_y, gps_covX, gps_covY):
        self.time = time
        self.odo_x = odo_x
        self.odo_y = odo_y
        self.odo_o = odo_o
        self.imu_o = imu_o
        self.imu_cov = imu_cov
        self.gps_x = gps_x
        self.gps_y = gps_y
        self.gps_covX = gps_covX
        self.gps_covY = gps_covY

        
        #Create Matrix A
        cos_theta = np.around(np.cos(self.odo_o), decimals=5)
        sin_theta = np.around(np.sin(self.odo_o), decimals=5)
        A_mat = m.matrix([[1, 0, self.time*cos_theta, 0, 0],
                            [0, 1, self.time*sin_theta, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, self.time],
                            [0, 0, 0, 0, 1]])
        self.matrixA = A_mat

        #Creat Xstate matrx
        self.velocity = 0.44
        tan_theta = np.around(np.tan(self.odo_o), decimals=5)
        angularVelocity = self.velocity * (tan_theta / 1)
        X_mat = m.matrix([[self.odo_x, self.odo_y, self.velocity, self.odo_o, angularVelocity]])
        self.matrixX = X_mat



if __name__ == "__main__":
    main()
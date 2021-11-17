import numpy as np
import pandas as pd
import os
from scipy import optimize
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import Label, ttk, filedialog
from tkinter.filedialog import askopenfile


class ComputeCurvature:

    def __init__(self):
        """ Initialize some variables """
        self.xc = 0  # X-coordinate of circle center
        self.yc = 0  # Y-coordinate of circle center
        self.r = 0  # Radius of the circle
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points

    def calc_r(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.xx - xc) ** 2 + (self.yy - yc) ** 2)

    def f(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = np.empty((len(c), self.xx.size))

        ri = self.calc_r(xc, yc)
        df_dc[0] = (xc - self.xx) / ri  # dR/dxc
        df_dc[1] = (yc - self.yy) / ri  # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(
            self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()

        return 1 / self.r  # Return the curvature

    def Radius_of_curvature(self):
        radius = self.r
        return radius


# df = pd.read_csv("10mM_capacitive.csv")
# columns = list(df.columns)[1:]

files = os.listdir(r"C:\Users\Madhukant\Downloads\csv folder")
xyz = r"C:\Users\Madhukant\Downloads\csv folder"

#dictionary = {}
for file in files:
    # data['manager'].index.tolist()[200:300]
    data = dict()


    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(xyz, file))
        df = df[475:525]

        columns = list(df.columns)[1:]
        print(columns)

        col_1 = df[df.columns[0]]
        x = max(col_1.tolist())
        print(x)

        column_1 = np.array(col_1)
        column_1_arr = np.isnan(column_1)
        not_nan_array = ~ column_1_arr
        array1 = column_1[not_nan_array]
        for i in range(1, 2):
            print("i", i)
            print("generated data")

            col_2 = df[df.columns[i]]
            y = max(col_2.tolist())
            print(y)

            column_2 = np.array(col_2)
            column_2_arr = np.isnan(column_2)
            not_nan_array1 = ~ column_2_arr
            array2 = column_2[not_nan_array1]
            x = array1
            y = array2

            comp_curv = ComputeCurvature()
            curvature = comp_curv.fit(x, y)
            radius = comp_curv.Radius_of_curvature()
            print("radious",radius)


            #Plot the result
            theta_fit = np.linspace(-np.pi, np.pi, 180)
            x_fit = comp_curv.xc + comp_curv.r * np.cos(theta_fit)
            y_fit = comp_curv.yc + comp_curv.r * np.sin(theta_fit)
            plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2)
            plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.title('curvature = {:.3e}'.format(curvature,comp_curv.)
            plt.title('Radius = {}'.format(comp_curv.Radius_of_curvature()))
            plt.show()

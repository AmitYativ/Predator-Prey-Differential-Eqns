#!/usr/bin/env python
# Author: Amit Yativ

from pylab import *
from scipy import integrate
from operator import itemgetter
from prettytable import PrettyTable
import numpy
import matplotlib.pyplot as p

""" Program Description:

This program is written in Python and utilizes methods
of approximating ordinary differential equations such
as RK4, Forward Euler, and Adams Predictor Corrector
Method to approximate periodic solutions of the Lotka-Volterra
Equations. This depicts the change of population through time
within a predator-prey relationship, such as rabbits (prey)
and foxes (predators), portraying the dynamics of biological
systems in which the predator and prey species interact.

"""

# ------- Printing Set-up ---------------------
# Prey Table
prey_table = PrettyTable()
prey_table.align = "l" ; prey_table.right_padding_width = 0
prey_table.left_padding_width = 0  ; prey_table.junction_char = 'o'
prey_table.vertical_char = '|' ; prey_table.horizontal_char = '-'
prey_table.field_names = ["x(t)","t"]

# Predator Table
pred_table = PrettyTable()
pred_table.align = "l" ; pred_table.right_padding_width = 0
pred_table.left_padding_width = 0  ; pred_table.junction_char = 'o'
pred_table.vertical_char = '|' ; pred_table.horizontal_char = '-'
pred_table.field_names = ["y(t)","t"]

np.set_printoptions(precision=8, suppress=True)
# ----------------------------------------------

def main():
    #Input values
    t0 = 0
    tn = 100
    h  = 0.01
    n  = int((tn - t0)/h) - 1
    x0 = 10
    y0 = 5
    t = np.arange(t0,tn,h)

    # ----------------- Euler's Method ----------------------------------
    # (1) Compute Results
    arr1 = euler2D(t0, tn, n, x0, y0)

    # (2) Print Results
    print('\nOur results using the Euler Method are: ')
    print('\n Rabbits (Prey) (Euler Method):')
    for i in range(0,t.size,int((n+1)/10)):
        prey_table.add_row([arr1[0][i], t[i]])
    print(prey_table)
    print('\nFoxes (Predator) (Euler Method):')
    for i in range(0,t.size,int((n+1)/10)):
        pred_table.add_row([arr1[1][i], t[i]])
    print(pred_table)
    print('---------------------------------------------------------------')

    # ---------------------------------------------------------------
    prey_table.clear_rows()  ;  pred_table.clear_rows()
    # ----------------- RK4 Method ----------------------------------
    # (1) Compute Results
    arr2 = RK42D(t0, tn, n, x0, y0)

    # (2) Print Results
    print('\nOur results using the Runge Kutta Order 4 Method are: ')
    print('\nRabbits (Prey) (RK4 Method):')
    for i in range(0,t.size,int((n+1)/10)):
        prey_table.add_row([arr2[0][i], t[i]])
    print(prey_table)
    print('\nFoxes (Predator) (RK4 Method):')
    for i in range(0,t.size,int((n+1)/10)):
        pred_table.add_row([arr2[1][i], t[i]])
    print(pred_table)
    print('\n---------------------------------------------------------------')

    # ---------------------------------------------------------------
    prey_table.clear_rows()   ;   pred_table.clear_rows()
    # -------------- Adams 4th Predictor Corrector Method -----------
    # (1) Compute Results
    arr3 = adamsPredictCorrect(t0, tn, x0, y0, h)

    # (2) Print Results
    print('\nOur results using the Adams Pred. Corr. Order 4 Method are: ')
    print('\nRabbits (Prey) (Adams Pred. Corr. Order 4 Method):')
    for i in range(0, t.size, int((n + 1) / 10)):
        prey_table.add_row([arr3[0][i], t[i]])
    print(prey_table)
    print('\nFoxes (Predator) (Adams Pred. Corr. Order 4 Method):')
    for i in range(0, t.size, int((n + 1) / 10)):
        pred_table.add_row([arr3[1][i], t[i]])
    print(pred_table)
    print('\n---------------------------------------------------------------')

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # ----------------- SciPy Integration Method ---------------------
    # (1) Compute Results
    X = np.array(integrate.odeint(F2, [x0,y0], t))
    rabbits,foxes = X.T
    # -----------------------------------------------------------------

    # (*) Graph Results
    graphIt(t,arr1,'Euler Method')
    graphIt(t,arr2,'Runge-Kutta Order 4 Method')
    graphIt(t,arr3,'Adams Predictor Corrector Order 4 Method')
    graphIt(t,[rabbits,foxes], 'SciPy Integration Method')

# --------------------------------- Lotka-Volterra System + Set-Up ---------------------------------
# Coefficients

a = 0.9
b = 0.1
c = 0.1
d = 0.4

# Lotka - Volterra System
def F(t, x, y):
    return (a * x - b * x * y,c * x * y - d * y) #(Prey,Predator)

def F2(X, t = 0):
    return (a * X[0] - b * X[0] * X[1],c * X[0] * X[1] - d * X[1]) #(Prey,Predator)


"""  Lotka-Volterra Coefficients:
     a is the natural growing rate of rabbits, when there's no fox
     b is the natural dying rate of rabbits, due to predation
     c is the natural dying rate of fox, when there's no rabbit
     d is the factor describing how many caught rabbits let create a new fox
"""

# ---------------------------------------------------------------------------------------------------
# --------------------------------- Algorithms ------------------------------------------------------
#Forward Euler’s Method
def euler2D(t0,tn,n,x0,y0):
    h = abs(tn-t0)/n
    t = linspace(t0,tn,n+1)
    x = zeros(n+1)
    y = zeros(n+1)
    x[0] = x0
    y[0] = y0
    for k in range(0,n):
        (dx,dy) = F(t[k],x[k],y[k])
        x[k+1] = x[k] + h*dx
        y[k+1] = y[k] + h*dy
    return np.array([x,y])


# Runge-Kutta Method - 4th Order
def RK42D(t0,tn,n,x0,y0):
    h = abs(tn-t0)/n
    t = linspace(t0,tn,n+1)
    x = zeros(n+1)
    y = zeros(n+1)
    x[0] = x0
    y[0] = y0
    for k in range(0,n):
        (dx1,dy1) = F(t[k],x[k],y[k])
        (dx2,dy2) = F(t[k]+h/2,x[k]+dx1*h/2,y[k]+dy1*h/2)
        (dx3,dy3) = F(t[k]+h/2,x[k]+dx2*h/2,y[k]+dy2*h/2)
        (dx4,dy4) = F(t[k]+h, x[k]+dx3*h, y[k]+dy3*h)
        x[k+1] = x[k] + h*(dx1 +2*dx2 +2*dx3 +dx4)/6
        y[k+1] = y[k] + h*(dy1 +2*dy2 +2*dy3 +dy4)/6
    return np.array([x,y])


# For vector
def vec(m):
    z = [0] * m; return (z)

# First item of list
def Extract_1(lst):
    return [item[0] for item in lst]
# Second item of list
def Extract_2(lst):
    return [item[1] for item in lst]


# Adams Predictor-Corrector
def adamsPredictCorrect(t0, tn, init_cond_x, init_cond_y, h):
    N = int((tn - t0) / h)  - 1
    y = vec(N + 1) ;  x = vec(N + 1)
    y[0] = round(init_cond_y, 8)  ;  x[0] = round(init_cond_x, 8)
    t = linspace(t0, tn, N + 1)
    for k in range(0, N):
        if k in range(0, 3):
            (dx1, dy1) = F(t[k], x[k], y[k])
            (dx2, dy2) = F(t[k] + h / 2, x[k] + dx1 * h / 2, y[k] + dy1 * h / 2)
            (dx3, dy3) = F(t[k] + h / 2, x[k] + dx2 * h / 2, y[k] + dy2 * h / 2)
            (dx4, dy4) = F(t[k] + h, x[k] + dx3 * h, y[k] + dy3 * h)
            x[k + 1] = x[k] + h * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6
            y[k + 1] = y[k] + h * (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6

        # Set Up Values
        (dx, dy) = F(t[k], x[k], y[k])
        (dx_m1,dy_m1) = F(t[k - 1], x[k - 1], y[k - 1])
        (dx_m2,dy_m2) = F(t[k - 2], x[k - 1], y[k - 2])
        (dx_m3,dy_m3) = F(t[k - 3], x[k - 1], y[k - 3])

        #Y(t) and X(t) Predictions
        y[k + 1] = y[k] + h * ( 55.0 * dy - 59.0 * dy_m1 + 37.0 \
                                * dy_m2 - 9.0 * dy_m3) / 24.0
        x[k + 1] = x[k] + h * (55.0 * dx - 59.0 * dx_m1 + 37.0 \
                               * dx_m2 - 9.0 * dx_m3) / 24.0

        # Evaluate function with predictions
        (dx_p1,dy_p1) = F(t[k + 1], x[k + 1], y[k + 1])

        # Y(t) and X(t) Corrections
        y[k + 1] = round(y[k] + h * (9.0 * dy_p1 + 19.0 * dy - 5.0 \
                                     * dy_m1 + dy_m2) / 24.0,8)
        x[k + 1] = round(x[k] + h * (9.0 * dx_p1 + 19.0 * dx - 5.0 \
                                     * dx_m1 + dx_m2) / 24.0, 8)
    return np.array([x,y])

def graphIt(t,arr, method_used):
    p.plot(t, arr[0], 'r-', label='Rabbits (Prey)')
    p.plot(t, arr[1], 'b-', label='Foxes (Predator)')
    p.grid()
    p.legend(loc='upper left', fontsize = 'small')
    p.xlabel('Time')
    p.ylabel('Population]')
    p.title('Evolution of Fox and Rabbit Populations Via {} Method: '.format(method_used))
    p.show()

if __name__ == '__main__':
    main()

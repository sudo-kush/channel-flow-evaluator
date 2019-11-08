#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:27:12 2019

@author: student
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLineEdit, QDialog, 
                             QVBoxLayout, QAction, QSizePolicy, QPushButton, QHBoxLayout, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvas
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
class psi: 
    def __init__(self, lx1, lx2, ly, dx, dy, w, tolerance):
        self.lx1, self.lx2, self.ly, self.dx, self.dy, self.w = lx1, lx2, ly, dx, dy, w
        self.tolerance = tolerance
    def create_tri(lx1,lx2,ly,dx,dy, w, tolerance):
        beta = dx/dy 
        lx = 2*lx1+lx2 + 2*dx
        nx = int(lx/dy)
        ny = int(ly/dy)
        PSI = sp.zeros((ny,nx))
        al = int(lx1 / dx) # left segment
        ar = int(((lx1+lx2)+2*dx) / dx) # right segment
        PSI[0, 0:al] = 1.0 # bottom1
        PSI[0, ar:] = 1.0 # bottom2
        PSI[-1, :] = 1.0 # top
        PSI[:, 0] = 1.0 # left 
        PSI[:, -1] = 1.0 # right
        A = 2*(1+beta**2)*sp.diag(sp.ones(nx))-w*sp.diag(sp.ones(nx-1), -1)-w*sp.diag(sp.ones(nx-1),1)
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        A[0, 1] = 0.0
        A[-1, -2] = 0.0
        A = sp.sparse.csr_matrix(A)
        b = sp.zeros(nx) # right hand side vector 
        b[0] = 1.0
        b[-1] = 1.0 
        itr = 1;
        maxitr = 5000
        while itr < maxitr:
            tempPSI = sp.copy(PSI)
            # Sweep by row
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    b[j] = 2*(1+beta**2)*(1-w)*PSI[i,j]+w*beta**2*(PSI[i+1,j]+PSI[i-1,j])
                    # Solve Ax = b
                    u = sp.sparse.linalg.spsolve(A, b)
                    PSI[i,:] = u
                    # Check for Convergence
            if sp.linalg.norm(tempPSI-PSI) <= tolerance:
                break
            itr = itr + 1
        return itr, PSI
        
        
class grid:
    def __init__(self, lx1, lx2, ly, dx, dy):
        super(psi, self).__init__(lx1, lx2, ly, dx, dy)
        self.lx1, self.lx2, self.ly, self.dx, self.dy = lx1, lx2, ly, dx, dy
        
    def create_mesh(lx1, lx2, ly, dx, dy):
        lx = 2*lx1+lx2 + 2*dx
        X, Y = sp.meshgrid(sp.linspace(0,lx,int(lx/dx)), sp.linspace(0,ly,int(ly/dy)))
        return X, Y

class MainWindow(QMainWindow):
     def __init__(self):
        QMainWindow.__init__(self)
        
        self.menuFile = self.menuBar().addMenu("&File")
        self.actionQuit = QAction("&Quit", self)
        self.actionQuit.triggered.connect(self.close)
        self.menuHelp = self.menuBar().addMenu("&Help")
        self.setWindowTitle("Channel Flow Evaluator")
        self.form=Form()
        self.setCentralWidget(self.form)
        


class Form(QDialog):

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        
        
        self.function_edit = QLineEdit("0.25")
        self.function_edit.selectAll()
        self.parameter_edit = QLineEdit("1.0e-3")
        self.parameter_edit.selectAll()
        
        self.plot = MatplotlibCanvas()
        self.function_edit.setFocus()
        # run button
        self.run =  QPushButton("Run")
        self.run.clicked.connect(self.clear_in)
        self.run.clicked.connect(self.updateUI)
        # clear button
        self.clear_data = QPushButton("Clear")
        self.clear_data.clicked.connect(self.clear_out)
        # layout

        layout=QVBoxLayout()
        self.plot=MatplotlibCanvas()
        layout1=QVBoxLayout()        
        layout1.addWidget(QLabel("Change in x and y (dx & dy)"))
        layout1.addWidget(self.function_edit)
        layout1.addWidget(QLabel("Tolerance"))
        layout1.addWidget(self.parameter_edit)

        layout2=QHBoxLayout()
        layout2.addWidget(self.run)
        layout2.addWidget(self.clear_data)
        layout.addWidget(self.plot)
        layout.addLayout(layout1)
        layout.addLayout(layout2)
        self.setLayout(layout)
        
    def updateUI(self):
        
        self.dx = float(self.function_edit.text())
        self.t = float(self.parameter_edit.text())
        #Create grid create_mesh(lx1,lx2,ly,dx,dy)
        self.X, self.Y = grid.create_mesh(1.5,2.5,5.0,self.dx,self.dx)
        #Create triangular matrix create_tri(lx1, lx2, ly, dx, dy, w, tolerance)
        self.itr, self.PSI = psi.create_tri(1.5,2.5,5.0,self.dx,self.dx,1.0,self.t)
        self.plot.redraw(self.X,self.Y,self.PSI)
    
    def clear_in(self):
        self.plot.figure.clear()
        self.plot.axes.figure.canvas.draw_idle()
    
    def clear_out(self):
        self.parameter_edit.clear()
        self.function_edit.clear()
        self.plot.figure.clear()
        self.plot.axes.figure.canvas.draw_idle()

class MatplotlibCanvas(FigureCanvas):
    import matplotlib.pyplot as plt
    def __init__(self):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        self.fig.suptitle("Stream Function", fontsize=18)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
   
    def redraw(self, x, y, PSI):
        import matplotlib.pyplot as plt
        self.axes.clear()
        self.plt.contourf(x,y,PSI,20,cmap=plt.cm.jet)
        self.plt.colorbar()
        self.plt.suptitle("Stream Function", fontsize=18)
        self.plt.xlabel('X',fontsize=18)
        self.plt.ylabel('Y', fontsize=18)
        self.draw()    



       
app = QApplication(sys.argv)
widget = MainWindow()
widget.show()
app.exec_()    


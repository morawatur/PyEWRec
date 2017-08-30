import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

fName = 'uw_out.txt'
textFile = open(fName, 'rb')

x = np.arange(0, 8, 1)
y = np.arange(0, 8, 1)
z = np.loadtxt(textFile)
X, Y = np.meshgrid(x, y)

textFile.close()

refIdx = 1
inFocusIdx = 11
nImages = 20
dfMin = -inFocusIdx * 2.0
dfMax = (nImages - inFocusIdx) * 2.0
dfValues = np.arange(dfMin, dfMax, 2.0)

with PdfPages('output.pdf') as pdf:
    for page in range(19):
        Z1 = z[page*64:(page+1)*64, 0].reshape(8, 8)
        Z2 = z[page*64:(page+1)*64, 1].reshape(8, 8)
        dfDiff = dfValues[page + 1] - dfValues[refIdx - 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.jet, antialiased=False, shade=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
        plt.title('{0}->{1}: {2} um, x-shift'.format(refIdx, page+2, dfDiff))
        pdf.savefig()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.jet, antialiased=False, shade=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
        plt.title('{0}->{1}: {2} um, y-shift'.format(refIdx, page+2, dfDiff))
        pdf.savefig()
        plt.close()
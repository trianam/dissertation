import numpy as np
import math
from matplotlib.path import Path
import matplotlib.patches as patches

class Polygon:
    def __init__(self, **pars):
        if 'maxEmptyLen' in pars:
            maxEmptyLen = pars['maxEmptyLen']
        else:
            maxEmptyLen = 0.1

        #if 'vertexes' in pars:
        self._vertexes = pars['vertexes']
        
        edges=[]
        for i in range(1,len(self._vertexes)):
            edges.append([self._vertexes[i-1], self._vertexes[i]])
        edges.append([self._vertexes[i], self._vertexes[0]])
        self._edges=np.array(edges)

        if 'invisible' in pars:
            self._invisible = pars['invisible']
        else:
            self._invisible = False
            
        allPoints = []
        if (not 'distributePoints' in pars) or (pars['distributePoints'] == True):

            for edge in self._edges:
                allPoints.append(edge[0])
                allPoints.append(edge[1])

            while edges:
                edge = edges.pop(0)
                if np.linalg.norm(edge[1]-edge[0]) > maxEmptyLen:
                    mid = self._comb2(edge[0],edge[1])
                    allPoints.append(mid)
                    edges.append([edge[0],mid])
                    edges.append([mid, edge[1]])
                    
        self.allPoints = np.array(allPoints)

        vert = self._vertexes.tolist()
        vert.append(vert[0])
        cod=np.ones(len(vert),int)*Path.MOVETO
        cod[1:len(vert)-1]=Path.LINETO
        cod[len(vert)-1]=Path.CLOSEPOLY

        self._mplPath = Path(vert, cod)

        
    _comb2 = lambda self,a,b: 0.5*a + 0.5*b
        
    def plotAllPoints(self, plotter):
        plotter.plot(self.allPoints[:,0], self.allPoints[:,1], self.allPoints[:,2], 'ob')

    def intersectSegment(self, a, b):

        for edge in self._edges:
            diffba = b-a
            diffEdge = edge[1]-edge[0]
            diffEa = edge[0] - a

            A = np.array([
                [diffba[0], -diffEdge[0]],
                [diffba[1], -diffEdge[1]]])
            B = np.array([diffEa[0], diffEa[1]])
            
            try:
                x = np.linalg.solve(A,B)
                # check if
                #          0 <= k <= 1,
                #          0 <= w <= 1,
                if (x[0] >= 0.) and (x[0] <= 1.) and (x[1] >= 0.) and (x[1] <= 1.):
                    return True
            except np.linalg.linalg.LinAlgError:
                pass

        return False

    def isInside(self, point):
        if not self._invisible:
            return self._mplPath.contains_point(point)
        else:
            return False
    
    def plot(self, plotter):
        if self._invisible == False:
            patch = patches.PathPatch(self._mplPath, facecolor='palegoldenrod', lw=2)
            plotter.add_patch(patch)
            

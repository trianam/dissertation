import numpy as np
import polygon

class Scene:
    def __init__(self):
        self._polygons = []
        
    def addPolygon(self, polygon):
        self._polygons.append(polygon)

    def addBoundingBox(self, a, b, maxEmptyLen=1, invisible=True):
        self._polygons.append(polygon.Polygon(vertexes=np.array([a,[a[0], b[1]],b,[b[1], a[0]]]), invisible=invisible, maxEmptyLen=maxEmptyLen))

    def allPoints(self):
        points = []
        for polygon in self._polygons:
            points.extend(polygon.allPoints)
        return np.array(points)

    def segmentIntersect(self, a, b):
        for polygon in self._polygons:
            if polygon.intersectSegment(a,b):
                return True
        return False


    def polygons(self):
        return self._polygons

    def isInside(self, point):
        for p in self._polygons:
            if p.isInside(point):
                return True
            
        return False
    
    def plot(self, plotter):
        for poly in self._polygons:
            poly.plot(plotter)




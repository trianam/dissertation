import random
import math
import numpy as np
import scipy.interpolate as si

class Path:
    """
    Represents a state of system in the lagrangian space (path
    configurations X constraint).
    """
    _maxVlambdaPert = 100.
    _maxVertexPert = 0.01
    _initialVlambda = 0.
    _changeVlambdaProbability = 0.05
    _numPointsSplineMultiplier = 10
    _numSigmaGauss = 9
    
    def __init__(self, initialVertexes, scene, optimizeVal):
        """
        optimizeVal can be: 'length', 'meanAngle', 'maxAngle', 'meanCurvature', 'maxCurvature', 'maxCurvatureLength', 'maxDerivative2'
        """
        self._vertexes = initialVertexes
        self._scene = scene
        self._optimizeVal = optimizeVal
        self._dimR = self._vertexes.shape[0]
        self._dimC = self._vertexes.shape[1]
        self._numPointsSpline = self._numPointsSplineMultiplier * self._dimR
        self._spline, splineD1, splineD2 = self._splinePoints(self._vertexes)
        self._vlambda = self._initialVlambda
        self._initialLength = self._calculateTotalLength(self._vertexes)
        self._currentEnergy, self._currentLength, self._currentMeanAngle, self._currentMaxAngle, self._meanCurvature, self._maxCurvature, self._maxCurvatureLength, self._maxDerivative2, self._currentConstraints = self._initializePathEnergy(self._vertexes, self._spline, splineD1, splineD2, self._vlambda)
        

    @property
    def vertexes(self):
        return self._vertexes

    @property
    def spline(self):
        return self._spline

    @property
    def energy(self):
        return self._currentEnergy
    
    @property
    def length(self):
        return self._currentLength
    
    @property
    def meanAngle(self):
        return self._currentMeanAngle
    
    @property
    def maxAngle(self):
        return self._currentMaxAngle
    
    @property
    def meanCurvature(self):
        return self._currentMeanCurvature
    
    @property
    def maxCurvature(self):
        return self._currentMaxCurvature
    
    @property
    def maxCurvatureLength(self):
        return self._currentMaxCurvatureLength
    
    @property
    def maxDerivative2(self):
        return self._currentMaxDerivative2
    
    @property
    def constraints(self):
        return self._currentConstraints
    
    @property
    def vlambda(self):
        return self._vlambda
    
    @property
    def optimizeVal(self):
        return self._optimizeVal
    
    
    def tryMove(self, temperature, neighbourMode):
        """
        Move the path or lambda multipiers in a neighbouring state,
        with a certain acceptance probability.
        Pick a random vertex (except extremes), and move
        it in a random direction (with a maximum perturbance).
        Use a lagrangian relaxation because we need to evaluate
        min(measure(path)) given the constraint that all quadrilaters
        formed by 4 consecutive points in the path must be collision
        free; where measure(path) is, depending of the choose method,
        the length of the path or the mean
        of the supplementary angles of each pair of edges of the path.
        If neighbourMode=0 then move the node uniformly, if
        neighbourMode=1 then move the node with gaussian probabilities
        with mean in the perpendicular direction respect to the
        previous-next nodes axis.
        """

        moveVlambda = random.random() < self._changeVlambdaProbability
        if moveVlambda:
            newVlambda = self._vlambda
            newVlambda = newVlambda + (random.uniform(-1.,1.) * self._maxVlambdaPert)

            newEnergy = self._calculatePathEnergyLambda(newVlambda)

            #attention, different formula from below
            if (newEnergy > self._currentEnergy) or (math.exp(-(self._currentEnergy-newEnergy)/temperature) >= random.random()):
                self._vlambda = newVlambda
                self._currentEnergy = newEnergy
        
        else:
            newVertexes = np.copy(self._vertexes)
            movedV = random.randint(1,self._dimR - 2) #don't change extremes

            if(neighbourMode == 0):
                moveC = random.randint(0,self._dimC - 1)
                newVertexes[movedV][moveC] = newVertexes[movedV][moveC] + (random.uniform(-1.,1.) * self._maxVertexPert)
            else:
                a = self._vertexes[movedV-1] - self._vertexes[movedV+1]
                b = np.array([1,0])
                
                alfa = math.acos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) - (math.pi/2)
                randAng = self._truncGauss(math.pi/2, math.pi/(2*self._numSigmaGauss), 0, math.pi)
                if random.randint(0,self._dimC - 1) == 1:
                    randAng = randAng + math.pi

                randAng = randAng + alfa

                randDist = random.uniform(-1.,1.) * self._maxVertexPert

                newVertexes[movedV] = self._vertexes[movedV] + np.array([randDist * math.cos(randAng), randDist * math.sin(randAng)])
#                newVertexes[movedV][0] = newVertex[0]
#                newVertexes[movedV][1] = newVertex[1]
                            
                
            newSpline,newEnergy,newLength,newMeanAngle,newMaxAngle,newMeanCurvature,newMaxCurvature,newMaxCurvatureLength,newMaxDerivative2,newConstraints = self._calculatePathEnergyVertex(newVertexes, movedV)

            #attention, different formula from above
            if (newEnergy < self._currentEnergy) or (math.exp(-(newEnergy-self._currentEnergy)/temperature) >= random.random()):
                self._vertexes = newVertexes
                self._spline = newSpline
                self._currentEnergy = newEnergy
                self._currentLength = newLength
                self._currentMeanAngle = newMeanAngle
                self._currentMaxAngle = newMaxAngle
                self._currentMeanCurvature = newMeanCurvature
                self._currentMaxCurvature = newMaxCurvature
                self._currentMaxCurvatureLength = newMaxCurvatureLength
                self._currentMaxDerivative2 = newMaxDerivative2
                self._currentConstraints = newConstraints

    def _truncGauss(self, mu, sigma, bottom, top):
        v = random.gauss(mu,sigma)
        while not (bottom <= v <= top):
            v = random.gauss(mu,sigma)
        return v
        
    def _initializePathEnergy(self, vertexes, spline, splineD1, splineD2, vlambda):
        length = self._calculateTotalLength(vertexes)

        meanAngle = 0.
        maxAngle = 0.
        for i in range(1, self._dimR - 1): #from 1 to dimR-2
            currAngle = self._calculateAngle(vertexes[i-1], vertexes[i], vertexes[i+1])
            meanAngle = meanAngle + currAngle
            if currAngle > maxAngle:
                maxAngle = currAngle
            
        meanAngle = meanAngle / (self._dimR - 2)

        meanCurvature = self._calculateMeanCurvature(spline, splineD1, splineD2)
        maxCurvature = self._calculateMaxCurvature(spline, splineD1, splineD2)
        maxCurvatureLength = self._calculateMaxCurvatureLength(spline, splineD1, splineD2, vertexes)
        maxDerivative2 = self._calculateMaxDerivative2(splineD2)
        
        constraints = self._calculateConstraints(spline)

        if self._optimizeVal == 'length':
            energy = length + vlambda * constraints
        elif self._optimizeVal == 'meanAngle':
            energy = meanAngle + vlambda * constraints
        elif self._optimizeVal == 'maxAngle':
            energy = maxAngle + vlambda * constraints
        elif self._optimizeVal == 'meanCurvature':
            energy = meanCurvature + vlambda * constraints
        elif self._optimizeVal == 'maxCurvature':
            energy = maxCurvature + vlambda * constraints
        elif self._optimizeVal == 'maxCurvatureLength':
            energy = maxCurvatureLength + vlambda * constraints
        elif self._optimizeVal == 'maxDerivative2':
            energy = maxDerivative2 + vlambda * constraints

        return (energy, length, meanAngle, maxAngle, meanCurvature, maxCurvature, maxCurvatureLength, maxDerivative2, constraints)
                

    def _calculatePathEnergyLambda(self, vlambda):
        """
        calculate the energy when lambda is moved.
        """
        return (self._currentEnergy - (self._vlambda * self._currentConstraints) + (vlambda * self._currentConstraints))
    
    def _calculatePathEnergyVertex(self, vertexes, movedV):
        """
        calculate the energy when a vertex is moved and returns it.
        """
        spline, splineD1, splineD2 = self._splinePoints(vertexes)
        constraints = self._calculateConstraints(spline)

        length = 0.
        meanAngle = 0.
        maxAngle = 0.
        meanCurvature = 0.
        maxCurvature = 0.
        maxCurvatureLength = 0.
        maxDerivative2 = 0.
        if self._optimizeVal == 'length':
            length = self._calculateTotalLengthSimp(vertexes, movedV)
            energy = length + self._vlambda * constraints
        elif self._optimizeVal == 'meanAngle':
            meanAngle = self._calculateMeanAngle(vertexes, movedV)
            energy = meanAngle + self._vlambda * constraints
        elif self._optimizeVal == 'maxAngle':
            maxAngle = self._calculateMaxAngle(vertexes, movedV)
            energy = maxAngle + self._vlambda * constraints            
        elif self._optimizeVal == 'meanCurvature':
            meanCurvature = self._calculateMeanCurvature(spline, splineD1, splineD2)
            energy = meanCurvature + self._vlambda * constraints
        elif self._optimizeVal == 'maxCurvature':
            maxCurvature = self._calculateMaxCurvature(spline, splineD1, splineD2)
            energy = maxCurvature + self._vlambda * constraints
        elif self._optimizeVal == 'maxCurvatureLength':
            maxCurvatureLength = self._calculateMaxCurvatureLength(spline, splineD1, splineD2, vertexes)
            energy = maxCurvatureLength + self._vlambda * constraints
        elif self._optimizeVal == 'maxDerivative2':
            maxDerivative2 = self._calculateMaxDerivative2(splineD2)
            energy = maxDerivative2 + self._vlambda * constraints
            
        return (spline, energy, length, meanAngle, maxAngle, meanCurvature, maxCurvature, maxCurvatureLength, maxDerivative2, constraints)

    def _calculateTotalLength(self, vertexes):
        length = 0.
        for i in range(1, self._dimR):
            length = length + np.linalg.norm(np.subtract(vertexes[i], vertexes[i-1]))
        return length
            
    def _calculateTotalLengthSimp(self, vertexes, movedV):
        length = self._currentLength
        
        length = length - self._calculateLength(self._vertexes[movedV], self._vertexes[movedV-1]) + self._calculateLength(vertexes[movedV], vertexes[movedV-1])
        length = length - self._calculateLength(self._vertexes[movedV+1], self._vertexes[movedV]) + self._calculateLength(vertexes[movedV+1], vertexes[movedV])

        return length
    
    def _calculateMeanAngle(self, vertexes, movedV):
        meanAngle = self._currentMeanAngle
        if movedV >= 2:
            meanAngle = meanAngle + (self._calculateAngle(vertexes[movedV-2], vertexes[movedV-1], vertexes[movedV]) - self._calculateAngle(self._vertexes[movedV-2], self._vertexes[movedV-1], self._vertexes[movedV])) / (self._dimR - 2)

        meanAngle = meanAngle + (self._calculateAngle(vertexes[movedV-1], vertexes[movedV], vertexes[movedV+1]) - self._calculateAngle(self._vertexes[movedV-1], self._vertexes[movedV], self._vertexes[movedV+1])) / (self._dimR - 2)

        if movedV < self._dimR-2:
            meanAngle = meanAngle + (self._calculateAngle(vertexes[movedV], vertexes[movedV+1], vertexes[movedV+2]) - self._calculateAngle(self._vertexes[movedV], self._vertexes[movedV+1], self._vertexes[movedV+2])) / (self._dimR - 2)

        return meanAngle

    def _calculateMaxAngle(self, vertexes, movedV):
        maxAngle = 0.
        for i in range(1, self._dimR - 1): #from 1 to dimR-2
            currAngle = self._calculateAngle(vertexes[i-1], vertexes[i], vertexes[i+1])
            if currAngle > maxAngle:
                maxAngle = currAngle

        return maxAngle

    def _calculateMeanCurvature(self, spline, splineD1, splineD2):
        meanCurvature = 0.
        for i in range(0, len(spline)):
            d1Xd2 = np.cross(splineD1[i], splineD2[i])
            Nd1Xd2 = np.linalg.norm(d1Xd2)
            Nd1 = np.linalg.norm(splineD1[i])
            currCurv = Nd1Xd2 / math.pow(Nd1,3)
            
            meanCurvature += currCurv

        meanCurvature = meanCurvature / len(spline)

        return meanCurvature

    def _calculateMaxCurvature(self, spline, splineD1, splineD2):
        maxCurvature = 0.
        for i in range(0, len(spline)):
            d1Xd2 = np.cross(splineD1[i], splineD2[i])
            Nd1Xd2 = np.linalg.norm(d1Xd2)
            Nd1 = np.linalg.norm(splineD1[i])
            currCurv = Nd1Xd2 / math.pow(Nd1,3)

            if currCurv > maxCurvature:
                maxCurvature = currCurv

        return maxCurvature

    def _calculateMaxCurvatureLength(self, spline, splineD1, splineD2, vertexes):
        length = self._calculateTotalLength(vertexes)
        normLength = length/self._initialLength * 100 #for making the ratio indipendent of the initial length
        
        maxCurvature = 0.
        for i in range(0, len(spline)):
            d1Xd2 = np.cross(splineD1[i], splineD2[i])
            Nd1Xd2 = np.linalg.norm(d1Xd2)
            Nd1 = np.linalg.norm(splineD1[i])
            currCurv = Nd1Xd2 / math.pow(Nd1,3)

            if currCurv > maxCurvature:
                maxCurvature = currCurv

        ratioCurvLen = 0.1 #0: all length; 1: all maxCurvature
        return ratioCurvLen*maxCurvature + (1-ratioCurvLen)*normLength

    def _calculateMaxDerivative2(self, splineD2):
        maxDerivative2 = 0.
        for i in range(0, len(splineD2)):
            currDer2 = np.linalg.norm(splineD2[i])
            if currDer2 > maxDerivative2:
                maxDerivative2 = currDer2

        return maxDerivative2


    def _calculateConstraints(self, spline):
        """
        calculate the constraints function. Is the ratio of the points
        of the calculated spline that are inside obstacles respect the
        total number of points of the spline.
        """
        pointsInside = 0
        for p in spline:
            if self._scene.isInside(p):
                pointsInside = pointsInside + 1

        constraints = pointsInside / self._numPointsSpline

        return constraints

    def _splinePoints(self, vertexes):
        degree=4
        
        x = vertexes[:,0]
        y = vertexes[:,1]

        t = np.linspace(0, 1, len(vertexes) - degree + 1, endpoint=True)
        t = np.append([0]*degree, t)
        t = np.append(t, [1]*degree)

        tck = [t,[x,y], degree]
        

        u=np.linspace(0,1,self._numPointsSpline,endpoint=True)

        out = si.splev(u, tck)
        outD1 = si.splev(u, tck, 1)
        outD2 = si.splev(u, tck, 2)

        spline = np.stack(out).T
        splineD1 = np.stack(outD1).T
        splineD2 = np.stack(outD2).T

        return (spline, splineD1, splineD2)

    def _calculateLength(self, a, b):
        return np.linalg.norm(np.subtract(a, b))
        
    def _calculateAngle(self, a, b, c):
        #return 1. + (np.dot(np.subtract(a,b), np.subtract(c,b)) / (np.linalg.norm(np.subtract(a,b)) * np.linalg.norm(np.subtract(c,b))))
        return 1. + (np.dot(np.subtract(b,a), np.subtract(b,c)) / (np.linalg.norm(np.subtract(b,a)) * np.linalg.norm(np.subtract(b,c))))

    def plot(self, plotter, plotStartEnd=True, plotInnerVertexes=False, plotEdges=True, plotSpline=True):
        if plotEdges:
            plotter.plot(self._vertexes[:,0], self._vertexes[:,1], 'r--')
        if plotStartEnd:
            plotter.plot(self._vertexes[0,0], self._vertexes[0,1], 'ro')
            plotter.plot(self._vertexes[-1,0], self._vertexes[-1,1], 'ro')
        if plotInnerVertexes:
            plotter.plot(self._vertexes[1:-1,0], self._vertexes[1:-1,1], 'ro')
        if plotSpline:
            plotter.plot(self._spline[:,0], self._spline[:,1], 'r', lw=2)


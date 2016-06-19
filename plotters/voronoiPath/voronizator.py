import numpy as np
import numpy.linalg
import scipy as sp
import scipy.spatial
import networkx as nx
import numpy.linalg
import polygon
import path

class Voronizator:
    def __init__(self, scene):
        self._graph = nx.Graph()
        self._pathStart = np.array([])
        self._pathEnd = np.array([])
        self._scene = scene
        self._sites = scene.allPoints()
        
    def makeVoroGraph(self, prune=True):
        vor = sp.spatial.Voronoi(self._sites)
        vorVer = vor.vertices
        for ridge in vor.ridge_vertices:
            for i in range(len(ridge)):
                if (ridge[i] != -1) and (ridge[(i+1)%len(ridge)] != -1):
                    a = vorVer[ridge[i]]
                    b = vorVer[ridge[(i+1)%len(ridge)]]
                    if (not prune) or (not self._scene.segmentIntersect(a,b)):
                        self._graph.add_edge(tuple(a), tuple(b), weight=np.linalg.norm(a-b))
        i = 0
        for node in self._graph.nodes():
            self._graph.node[node]['index'] = i
            i = i + 1

    def createShortestPath(self, start, end, attachMode='near', minEdgeLen=0., maxEdgeLen=0., prune=True, optimizeVal='length'):
        if attachMode=='near':
            self._attachToGraphNear(start, end, prune)
        elif attachMode=='all':
            self._attachToGraphAll(start, end, prune)
        else:
            self._attachToGraphNear(start, end, prune)
            
        self._pathStart = start
        self._pathEnd = end

        if tuple(start) in self._graph.nodes():
            self._graph.node[tuple(start)]['index'] = 's'
        if tuple(end) in self._graph.nodes():
            self._graph.node[tuple(end)]['index'] = 'e'

        try:
            length,shortestPath=nx.bidirectional_dijkstra(self._graph, tuple(start), tuple(end))
        except (nx.NetworkXNoPath, nx.NetworkXError):
            shortestPath = []

        if (minEdgeLen > 0.) or (maxEdgeLen > 0.):
            i = 1
            while (i<len(shortestPath)):
                a = np.array(shortestPath[i-1])
                b = np.array(shortestPath[i])
                if (np.linalg.norm(b-a) < minEdgeLen) and (i>1) and (i<len(shortestPath)-1): #don't adjust extremes
                        shortestPath[i] = (0.5*a+0.5*b).tolist()
                        shortestPath.pop(i-1)
                        i = i-1
                        
                elif (np.linalg.norm(b-a) > maxEdgeLen):
                    shortestPath.insert(i, (0.5*a+0.5*b).tolist())
                else:
                    i = i+1

        return path.Path(np.array(shortestPath), self._scene, optimizeVal)

    def plotSites(self, plotter):
        if self._sites.size > 0:
            plotter.plot(self._sites[:,0], self._sites[:,1], 'o')
            
    def plotGraph(self, plotter, vertexes=True, edges=True, labels=False, pathExtremes=False, showOnly=[]):
        if vertexes:
            for ver in self._graph.nodes():
                if not showOnly or self._graph.node[ver]['index'] in showOnly:
                    if (ver!=tuple(self._pathStart) and ver!=tuple(self._pathEnd)):
                        plotter.plot([ver[0]], [ver[1]], 'og')
                        if labels and ('index' in self._graph.node[ver]):
                                plotter.text(ver[0], ver[1], self._graph.node[ver]['index'], color='red')
                    elif pathExtremes==True:
                        plotter.plot([ver[0]], [ver[1]], 'or')
                        if labels and ('index' in self._graph.node[ver]):
                                plotter.text(ver[0], ver[1], self._graph.node[ver]['index'], color='red')

        if edges:
            for edge in self._graph.edges():
                if not showOnly or (self._graph.node[edge[0]]['index'] in showOnly and self._graph.node[edge[1]]['index'] in showOnly):
                    if pathExtremes==True or (edge[0]!=tuple(self._pathStart) and edge[0]!=tuple(self._pathEnd) and edge[1]!=tuple(self._pathStart) and edge[1]!=tuple(self._pathEnd)):
                        plotter.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k')

    def _attachToGraphNear(self, start, end, prune):
        firstS = True
        firstE = True
        minAttachS = None
        minAttachE = None
        minDistS = 0.
        minDistE = 0.
        for node in self._graph.nodes():
            if (not prune) or (not self._scene.segmentIntersect(start,np.array(node))):
                if firstS:
                    minAttachS = node
                    minDistS = np.linalg.norm(start-np.array(node))
                    firstS = False
                else:
                    currDist = np.linalg.norm(start-np.array(node))
                    if currDist < minDistS:
                        minAttachS = node
                        minDistS = currDist
                    
            if (not prune) or (not self._scene.segmentIntersect(end,np.array(node))):
                if firstE:
                    minAttachE = node
                    minDistE = np.linalg.norm(end-np.array(node))
                    firstE = False
                else:
                    currDist = np.linalg.norm(end-np.array(node))
                    if currDist < minDistE:
                        minAttachE = node
                        minDistE = currDist

        if minAttachS != None:
            self._graph.add_edge(tuple(start), minAttachS, weight=minDistS)
        if minAttachE != None:
            self._graph.add_edge(tuple(end), minAttachE, weight=minDistE)

    def _attachToGraphAll(self, start, end, prune):
        for node in self._graph.nodes():
            if (not prune) or (not self._scene.segmentIntersect(start,np.array(node))):
                self._graph.add_edge(tuple(start), node, weight=np.linalg.norm(start-np.array(node)))
            if (not prune) or (not self._scene.segmentIntersect(end,np.array(node))):
                self._graph.add_edge(tuple(end), node, weight=np.linalg.norm(end-np.array(node)))
    

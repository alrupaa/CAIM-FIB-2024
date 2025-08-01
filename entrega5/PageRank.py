#!/usr/bin/python

from collections import namedtuple
import time
import sys

damping = 0.85
alfa = 0.1 #mirar el valor de alfa optimo
pageRank = []

class Edge:
    def __init__ (self, origin=None):
        self.origin = origin # write appropriate value
        self.weight = 0 # write appropriate value
        self.originIndex = 0

    def __repr__(self):
        return "edge: {0} {1}".format(self.origin, self.weight)
        
    ## write rest of code that you need for this class

class Airport:  #A:  importancia anterior del qeu te apunta * peso con el que te apunta / peso total de salida del qeu te apunta
    def __init__ (self, iden=None, name=None):
        self.code = iden
        self.name = name         
        self.routes = [] #lista de EDGES entrantes 
        self.routeHash = dict() #edge.origin -> edge
        self.outweight = 0  # suma pesos salientes

    def __repr__(self):
        return f"{self.code}\t{self.pageIndex}\t{self.name}"

edgeList = [] # list of Edge
edgeHash = dict() # hash of edge to ease the match
airportList = [] # list of Airport
airportHash = dict() # hash key IATA code -> Airport

def readAirports(fd):
    print("Reading Airport file from {0}".format(fd))
    airportsTxt = open(fd, "r")
    cont = 0
    for line in airportsTxt.readlines():
        a = Airport()
        try:
            temp = line.split(',')
            if len(temp[4]) != 5 :
                raise Exception('not an IATA code')
            a.name=temp[1][1:-1] + ", " + temp[3][1:-1] # name, country
            a.code=temp[4][1:-1] #iata code
        except Exception as inst:
            pass
        else:
            cont += 1
            airportList.append(a)
            airportHash[a.code] = a
    airportsTxt.close()
    print(f"There were {cont} Airports with IATA code")


def readRoutes(fd):
    print("Reading Routes file from {fd}")
    routesTxt = open(fd,"r")
    for line in routesTxt.readlines():
        e = Edge()
        try:
            temp = line.split(',')
            if len(temp[2]) != 3 or len(temp[4]) != 3 or temp[2] not in airportHash or temp[4] not in airportHash:
                raise Exception('not an IATA code or not a valid route')
        except Exception as inst:
            pass
        else:
            a = edgeHash.get((temp[2], temp[4]), None)
            if a is not None:
                a.weight += 1        
            else:
                e.origin = temp[2]
                e.weight = 1
                e.originIndex = airportList.index(airportHash[temp[2]])
                edgeList.append(e)
                edgeHash[(temp[2],temp[4])] = e
                airportHash[temp[4]].routes.append(e)
                airportHash[temp[4]].routeHash[temp[2]] = e
            airportHash[temp[2]].outweight += 1
    routesTxt.close()
    print(f"There were {len(edgeList)} routes")

    # write your code

def computePageRanks():
    print("Computing PageRanks...")
    n = len(airportList)
    P = [1/n for a in airportList]
    condition =  True
    iterations = 0
    while condition:
        condition = False
        iterations += 1
        Q = [0 for a in airportList]

        for i in range(0, n):
            Q[i] = (1 - damping) / n

        sinSalidas = 0

        for i in range(0, n):
            a = airportList[i]

            if a.outweight == 0:
                sinSalidas += P[i]
            suma = 0
            for edge in a.routes:
                suma += P[edge.originIndex] * edge.weight / airportList[edge.originIndex].outweight
            Q[i] += damping * suma

        for i in range(0, n):
            Q[i] += damping * sinSalidas / n
            if abs(P[i] - Q[i]) > alfa: 
                condition = True

        P = Q

    global pageRank 
    pageRank = P
    return iterations

def outputPageRanks():
    print("Printing PageRanks...")
    sorted_airports = sorted(zip(airportList, pageRank), key=lambda x: x[1], reverse=True)
    for airport, pageRanks in sorted_airports:
        print(f"{airport.code}\t{pageRanks}")

def debug():
    sorted_airports = sorted(zip(airportList, pageRank), key=lambda x: x[1], reverse=True)
    for idx, (airport, _) in enumerate(sorted_airports):
        if len(airport.routes) == 0:
            print(f"{airport.code}\t{idx}")

    total_page_rank = sum(pageRank)
    if abs(total_page_rank - 1.0) > alfa:
        print(f"Warning: Total PageRank is not 1 but {total_page_rank}")
    else:
        print("Total PageRank is correctly equal to 1")
    

def debugAlpha():
    global alfa
    previous_ranks = None
    for i in range(25):
        alfa = 0.1 / (10 ** i)
        start_time = time.time()
        iterations = computePageRanks()
        end_time = time.time()
        current_ranks = [airport.code for airport, _ in sorted(zip(airportList, pageRank), key=lambda x: x[1], reverse=True)]
        if previous_ranks is not None:
            same_position_count = sum(1 for prev, curr in zip(previous_ranks, current_ranks) if prev == curr)
            print(f"Alpha: {0.1}/10^{i}, Iterations: {iterations}, Same position count: {same_position_count}, Time: {end_time - start_time} seconds")
        previous_ranks = current_ranks

def mainProgram():
    time1 = time.time()
    iterations = computePageRanks()
    time2 = time.time()
    outputPageRanks()
    print("#Iterations:", iterations)
    print("Time of computePageRanks():", time2-time1)
    
def main(argv=None):
    readAirports("airports.txt")
    readRoutes("routes.txt")
    #mainProgram()
    #debug()
    debugAlpha()


if __name__ == "__main__":
    sys.exit(main())




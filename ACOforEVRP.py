from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

import osmnx as ox
import networkx as nx

class Instancia:
    def __init__(self,nombre):
        archivo = open(nombre, "r")
        lineas = [i.rstrip() for i in archivo.readlines()]
        parametros = [lineas[i].split(":",1) for i in range(11)]
        self.parametros = {i[0]:i[1] for i in parametros}
        self.llaves = list(self.parametros.keys())
        iCoordenadas = lineas.index('NODE_COORD_SECTION')
        iDemanda = lineas.index('DEMAND_SECTION')
        iEstaciones = lineas.index('STATIONS_COORD_SECTION')
        iDeposito = lineas.index('DEPOT_SECTION')
        iFin = lineas.index('EOF')
        correccion = int(lineas[iDeposito+1])
        self.deposito = 0
        self.coordenadas = [tuple(map(float,i.split(" "))) for i in lineas[iCoordenadas+1:iDemanda]]
        self.coordenadas = {n-correccion:(x,y) for n,x,y in self.coordenadas}
        self.demandas = [tuple(map(float,i.split(" "))) for i in lineas[iDemanda+1:iEstaciones]]
        self.demandas = {n-correccion: d for n,d in self.demandas}
        self.estaciones = {int(i)-correccion for i in lineas[iEstaciones+1:iDeposito]}
        archivo.close()
        
class colonia:
    def __init__(self, instancia, graph = None, poblacion =10, iterMax=200, alpha = 1, 
                 beta=2, rho = 0.2, q0 = 0, visualizarRuta =False, semilla = 0, vecinos = 8, periodo = 20, periodoTau = 500000):
        self.instancia = instancia
        self.graph = graph
        self.poblacion = poblacion
        self.iterMax = iterMax
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.visualizarRuta = visualizarRuta
        self.semilla = semilla
        self.vecinos = vecinos
        self.q0 = q0
        self.periodo = periodo
        self.periodoTau = periodoTau

    def distancia(self,i,j):
        return self.__distancias[(i,j)]

    def __inicializarDatos(self):
        # incializar acá las distanicas y la sij
        self.vehiculos = int(self.instancia.parametros["VEHICLES"])
        self.numNodos = int(self.instancia.parametros["DIMENSION"])
        self.clientes = {i for i in range(1,self.numNodos)}
        self.numEstaciones = int(self.instancia.parametros["STATIONS"]) 
        self.capacidad = float(self.instancia.parametros["CAPACITY"])
        self.capBateria = float(self.instancia.parametros["ENERGY_CAPACITY"])
        self.ratioConsumo = float(self.instancia.parametros["ENERGY_CONSUMPTION"])
        self.n = self.numNodos + self.numEstaciones
        self.estaciones = {i for i in range(self.numNodos,self.n)}
        self.__inicializarInstancia()
        self.__demandas = self.instancia.demandas
        self.__coordenadas = self.instancia.coordenadas
        self.__consumos = self.__distancias*self.ratioConsumo

    def __inicializarInstancia(self):
        n = self.n
        graph_area = ("Colo Colo 417, Concepción, Chile")
        G = ox.graph_from_address(graph_area, network_type='drive',dist = 2000)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        distancias = np.zeros((n,n),dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    distancias[i,j] = np.inf
                else:
                    t1 = self.instancia.coordenadas[i]
                    t2 = self.instancia.coordenadas[j]
                    origen = ox.get_nearest_node(G, t1)
                    destino =ox.get_nearest_node(G, t2)
                    aux = nx.shortest_path_length(G, origen, destino, weight='length')
                    distancias[i,j] = aux/1000
        self.G = G
        self.__distancias = distancias

    def consumo(self,i,j):
        return self.__consumos[i,j]

    def __actualizar(self,Tau,sbest,cbest):
        rho = self.rho
        Tau *= (1-rho)
        m = len(sbest)
        
        #actualización global
        for c in range(m-1):
            Tau[sbest[c],sbest[c+1]] += 1/cbest
        Tau[sbest[-1],sbest[0]] += 1/cbest
        Tau[Tau < self.TauMin] = self.TauMin
        Tau[Tau > self.TauMax] = self.TauMax
        return Tau

    def __candidatos(self,actual,restantes,bateria,carga):
        i = actual
        deposito, estacion = [0,min(self.estaciones, key = lambda x: self.consumo(actual,x))]
        candidatos = list()
        tieneCarga = False
        for j in restantes:
            if self.__demandas[j] < carga:
                tieneCarga  = True
                aux = bateria - self.consumo(i,j)
                if aux - self.consumo(i,deposito) > 0 or aux - self.consumo(i,estacion) > 0:
                    candidatos.append(j)
        if not tieneCarga:
            return [0]
        if len(candidatos) == 0:
            return [0,estacion]
        return candidatos

    def __costoRuta(self,ruta):
        m = len(ruta)
        energia = 0
        for k in range(m-1):
            i,j = ruta[k],ruta[k+1]
            energia += self.consumo(i,j)
        return energia

    def __subrutas(self,ruta):
        sub = []
        conjunto = []
        for i in ruta[1:]:
            if i == 0:
                conjunto.append(sub)
                sub = []
            else:
                sub.append(i)
        return conjunto

    def __siguiente(self,candidatos,probabilidad):
        # siguientes no solo elige si no que es el encargado de actualizar el costo que significará la decision
        if candidatos != [0]:
            if np.random.random() < self.q0:
                indice = probabilidad.argmax()
                return candidatos[indice]
            else:
                return int(np.random.choice(candidatos,size = 1,p = probabilidad))
        else:
            return 0

    def __explorar(self,atractivo):
        vecinos = self.vecinos
        bateria = self.capBateria
        carga = self.capacidad
        ruta = [0] # parte del cero
        restantes = self.clientes.copy()
        energia = 0

        while len(restantes) > 0:
            posicion = ruta[-1]
            candidatos = self.__candidatos(posicion,restantes,bateria,carga)
            if len(candidatos) >= vecinos:
                candidatos.sort(key=lambda x: self.consumo(posicion,x))
                candidatos = candidatos[0:vecinos]
            if candidatos == [0]:
                elegido = 0
                energia += self.consumo(posicion,0)
                carga = self.capacidad
                bateria = self.capBateria
            else:
                probabilidad = atractivo[posicion, candidatos]
                probabilidad = probabilidad/probabilidad.sum()
                elegido =  self.__siguiente(candidatos,probabilidad)                
                if elegido in self.clientes:
                    carga -= self.__demandas[elegido]
                    energia += self.consumo(posicion,elegido)
                    bateria -= self.consumo(posicion,elegido)
                else: 
                    energia += self.consumo(posicion,elegido)
                    bateria = self.capBateria
            ruta.append(elegido)
            restantes.difference_update({elegido})
        ruta.append(0)
        energia += self.consumo(ruta[-2],ruta[-1])
        return ruta, energia

    def __vecinoMasCercano(self):
        bateria = self.capBateria
        carga = self.capacidad
        ruta = [0] # parte del cero
        restantes = self.clientes.copy()
        energia = 0

        while len(restantes) > 0:
            posicion = ruta[-1]
            candidatos = self.__candidatos(posicion,restantes,bateria,carga)
            candidatos.sort(key=lambda x: self.consumo(posicion,x))
            elegido = candidatos[0] 
            if elegido == 0: 
                energia += self.consumo(posicion,0)
                carga = self.capacidad
                bateria  =self.capBateria
            elif elegido in self.clientes:
                carga = carga - self.__demandas[elegido]
                energia += self.consumo(posicion,elegido)
                bateria = bateria - self.consumo(posicion,elegido)
            else: 
                energia += self.consumo(posicion,elegido)
                bateria = self.capBateria
            ruta.append(elegido)
            restantes.difference_update({elegido})
        ruta.append(0)
        energia += self.consumo(ruta[-2],ruta[-1])
        return ruta, energia

    def __imprimir(self,solucion):
        solucion = [str(i) for i in solucion]
        return "-".join(solucion)

    def graficarLog(self):
        local, mejor = zip(*self.log)
        plots = plt.plot(local,'c-', mejor, 'b-')
        plt.legend(plots, ('mejor generacional', 'mejor global'), frameon=True)
        plt.ylabel('Costo')
        plt.xlabel('Generaciones')
        plt.title("Generaciones vs uso de energia - EVRP")
        plt.xlim((0, len(local)))
        plt.show()

    def visualizar(self,solucion):
        coord_x = [self.__coordenadas[i][0] for i in self.__coordenadas]
        coord_y = [self.__coordenadas[i][1] for i in self.__coordenadas]

        colores = ['crimson','gold','navy' ,'forestgreen','indigo'  ,'teal' ,'dodgerblue' ,'indianred' ,'orangered' ]

        m = len(solucion)
        arcosSolucion = [(solucion[i],solucion[i+1]) for i in range(m-1)]
        fig = plt.figure(figsize = (12,12))
        c = -1
        for i,j in  arcosSolucion:
            if i == 0:
                c += 1
            plt.plot([coord_x[i],coord_x[j]],[coord_y[i],coord_y[j]], color = colores[c] ,zorder=1)

        plt.scatter(coord_x[0],coord_y[0], color ='green',marker = 's',s = 275,zorder=2)
        plt.scatter(coord_x[self.numNodos:],coord_y[self.numNodos:], color ='green',marker = 's',s = 275,zorder=2)
        plt.scatter(coord_x[1:self.numNodos],coord_y[1:self.numNodos], color ='olive',marker = 'o', s = 275,zorder=3)
        
        for i in range(self.n):
            plt.annotate(str(i) ,xy = (coord_x[i],coord_y[i]),xytext = (coord_x[i],coord_y[i]), color = 'black',zorder=4)
        
        plt.xlabel('Coordenada en X')
        plt.ylabel('Coordenada en y')
        plt.title('EVRP en Concepción')
        fig.savefig('Gráfico de la mejor instancia')
        plt.show()
        plt.close()

    def visualizarCiudad(self,solucion):
        
        m = len(solucion)
        G = self.G
        arcosSolucion = [(solucion[i],solucion[i+1]) for i in range(m-1)]
        rutas = [ox.shortest_path(G, ox.get_nearest_node(G, self.__coordenadas[i]), ox.get_nearest_node(G, self.__coordenadas[j])) for i,j in arcosSolucion]
        colores = ["r","c","y"]*100
        rc = []
        for i,j in arcosSolucion:
            if i == 0:
                c = colores.pop()
            rc.append(c)
        fig, ax = ox.plot_graph_routes(self.G, rutas, route_colors=rc, route_linewidth=6, node_size=0)

        return None

    def ejecutar(self):
        np.random.seed(0)
        self.__inicializarDatos()

        global s_mejor, costoMejor, tiempo
        tiempoInicio = perf_counter()
        s_mejor, costoMejor = self.__vecinoMasCercano()
        log = list()
        n = self.n
        eta = 1/self.__distancias
        avg = (self.n-self.numEstaciones-1)/2
        m = 2
        self.TauMax = 1/(self.rho*costoMejor)
        self.TauMin = self.TauMax*(1-(0.05)**(1/n))/((avg-1)*(0.05)**(1/n))
        Tau = np.full((n,n),self.TauMax) 

        print(f"{'iter':^5}|{'mejor':^20}|{'actual':^20}")
        
        for iter in range(self.iterMax):
            atractivo = (Tau ** self.alpha) * (eta) ** self.beta 
            costoLocal = np.inf #iteration best
            for _ in range(self.poblacion): #cada individuo tiene que construir su camino
                ruta, costo = self.__explorar(atractivo)
                if costo < costoLocal:
                    costoLocal = costo
                    s = ruta
            if costoMejor > costoLocal:
                s_mejor,costoMejor = s.copy(),costoLocal
            if iter%self.periodoTau == 0:
                Tau = np.full((n,n),self.TauMax) 
            elif iter%self.periodo == 0:
                Tau = self.__actualizar(Tau,s_mejor,costoMejor)
            else: 
                Tau = self.__actualizar(Tau,s,costoLocal)
            log.append((costoLocal,costoMejor))
            print(f"{iter+1:^5}|{costoMejor:^20}|{costoLocal:^20}")

        self.log = log
        tiempo = perf_counter() - tiempoInicio
        print(f"mejor: {costoMejor}")
        print(self.__costoRuta(s_mejor))
        print(f"solucion: {self.__imprimir(s_mejor)}")
        print(self.__subrutas(s_mejor))
        print(f"tiempo: {tiempo}")
        return None  

def main():
    ins = Instancia("instancias/ccp.evrp")
    modelo = colonia(ins, poblacion = 30 , iterMax= 1000)
    modelo.ejecutar()
    modelo.graficarLog()
    modelo.visualizar(s_mejor)
    modelo.visualizarCiudad(s_mejor)
    if modelo.visualizarRuta:
        modelo.graficar
        
if __name__ == "__main__":
    main()
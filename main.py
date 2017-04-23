#/usr/bin/python

import logging,itertools
import networkx as nx
import matplotlib.pyplot as plt
from random import randint

"""============================================"""

logging.basicConfig()
LOG = logging.getLogger(__name__)
hdlr = logging.FileHandler('./'+__name__+'.log')
LOG.addHandler(hdlr)
LOG.setLevel(logging.INFO)      

NUMBER_OF_NODES=200
PPROB_OF_EDGE=0.1
K_NEAREST=4
G=None # Given Directed Graph  
E=None # Edges [(u,v)]
N=None # Nodes [List]
D=None # Active Transactions (cost,[path nodes])
E_P= None # edge to path link (e,[paths])

def Init():
    G= nx.fast_gnp_random_graph(NUMBER_OF_NODES,PPROB_OF_EDGE,directed=True)
    E= G.edges()
    N= G.nodes()
    #nx.draw(G)
    LOG.debug('Nodes: '+ str(N))
    LOG.debug('Edges: '+ str(E))
    LOG.info('Graph Created')
    #plt.show()
    return (G,E,N)


def Init_Tansactions(G,N,E):
    D=[]
    min_node=min(N)
    max_node=max(N)
    for u in N:
        for v in N:
            #v=randint(min_node,max_node)
            if v==u:
                continue
            try:
                l=nx.bidirectional_dijkstra(G,u, v)

            except:
                continue
            if l not in D:
                    D.append(l)
        
    LOG.debug('Edges: '+ str(E))
    LOG.info('Transactions Generated')

    return D

def Edge_Tran(D,E):
    E_Cost={}
    for e in E:
        E_Cost[e]=[]
        for d in D:
            m=d[1]
            t=[(m[i],m[i+1]) for i in range(len(m)-1)]
  
            if e in t:
                    E_Cost[e].append(d)

    return E_Cost

def Update_Trans(E_P, E):
    
    for e_p in E_P.keys():
        if e_p not in E:
            #print e_p
            del E_P[e_p]
    return E_P




def Edge_Cost(n,p):
    cost=0
    for p_n in p:
        if p_n[1].index(n) != len(p_n[1])-1:
            cost=cost+p_n[0]
    return cost

def Edge_Benefit(n,p):
    benefit=0
    for p_n in p:
        if p_n[1].index(n) == len(p_n[1])-1:
            benefit=benefit+1
    return benefit


def Play_Game(G,N,E_P):
    for n in N:
        for e_p in E_P:
            """"
            
            check where edge is incident on n
            """
            if n in e_p and e_p[0]!=n:
               u_t= Edge_Benefit(n, E_P[e_p]) - Edge_Cost(n, E_P[e_p])
               if u_t < 0:
                #print e_p
                G.remove_edge(e_p[0], e_p[1])
    
    return (G,G.edges(),N)




if __name__ == '__main__':

    G,E,N=Init()


    """"
    
    Creating copy of given graph
    """

    G1=G.copy()
    E1=None

    while (1):
        if E1 == E:
            break
        E1=E
        D=Init_Tansactions(G1,N, E)
        E_P=Edge_Tran(D, E)
        LOG.info('Playing Game')
        G1,E,N=Play_Game(G1,N,E_P)
        LOG.info('Edges:' + str(len(E)))



    f, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure(1)
    nx.draw(G,ax=ax1)
    nx.draw(G1,ax=ax2)
    plt.savefig('./Simple_Strategy.pdf')
    plt.show()

    


    
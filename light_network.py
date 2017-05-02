#/usr/bin/python

import logging,itertools
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from operator import itemgetter


"""============================================"""

""""

Assumption:
No node can send transactions worth more than his total bitcoins.
"""

logging.basicConfig()
LOG = logging.getLogger(__name__)
hdlr = logging.FileHandler('./'+__name__+'.log')
LOG.addHandler(hdlr)
LOG.setLevel(logging.INFO)      


NUMBER_OF_NODES=100
PPROB_OF_EDGE=0.01 # very low otherwise there will be always path
PER_NODE_TRANSACS=50
MAX_BITCOINS=100
G=None # Given Directed Graph  
E=None # Edges [(u,v)]
N=None # Nodes [List]
D=None # Active Transactions [n x n]
f=None # Trnsactions fee [n x n]
B=None # Number of bitcoins for each node [list]
D_NP= None # Number of Transaction which have no path [list]
C_Val= None #Channel Value [src Fij, dst Fij, f, z]
Path_E= None

def Init():
    G= nx.gnp_random_graph(NUMBER_OF_NODES,PPROB_OF_EDGE,directed=True)
    E= G.edges()
    N= G.nodes()
    #G=nx.to_numpy_matrix(G) #changing to numpy array 
    #nx.draw(G)
    LOG.debug('Nodes: '+ str(N))
    LOG.debug('Edges: '+ str(E))
    LOG.info('Graph Created')
    #plt.show()

    return (G,E,N)


def Init_Bitcoins(N):

    B=np.zeros(len(N))
    for u in N:
    	B[u]=MAX_BITCOINS
    	if u in range(10,13):
    		B[u]=MAX_BITCOINS*80000
    LOG.info('Bitcoins Initiated')
    return B

def Init_Tansactions(B,N):
    D=np.zeros((len(N),len(N)))
    f=np.zeros((len(N),len(N)))
    B_temp= np.copy(B)
    min_node=min(N)
    max_node=max(N)


    for u in N:
    	for i in range(0,PER_NODE_TRANSACS):
    		v=randint(min_node,max_node)
    		if u==v:
    			D[u][u]=0
    		else:
    			t_amount=randint(0,MAX_BITCOINS/4)
    			if  D[u][v]==0 and B_temp[u]-t_amount>0:
    				D[u][v]=t_amount
    				B_temp[u]=B_temp[u]-t_amount
    				if B_temp[u]-(t_amount/MAX_BITCOINS)>0:
    					f[u][v]=(t_amount/MAX_BITCOINS*2)
    					B_temp[u]=B_temp[u]-(t_amount/MAX_BITCOINS)
    LOG.info('Transactions Generated')
    
    return D,f




def Check_Paths(G,N,D):
	D_NP=[]
	for i in range(len(N)):
		for j in range(0,len(N)):
			if D[i][j]>0:
				try:
					l=nx.shortest_path(G,i, j)
					D_NP.append([i,j])
				except:	
					D_NP.append([i,j])
					continue
					
	LOG.info('Uncompleted Transactions Stored')
	return D_NP

def Channel_Val_Util(G,N,E,D,f,D_NP):
	G_temp=G.copy()
	C_Val=np.zeros((len(N),len(N),5))
	U=np.zeros((len(N),len(N)))
	P=np.zeros((len(N),len(N)))
	P=P.tolist()
	for i in N:
		for j in N:
			P[i][j]=[]
			if i!=j:
				G_temp.add_edge(i, j)
				for k in D_NP:
					try:
						l=nx.shortest_path(G_temp,k[0], k[1])
						if i==k[0]:
							C_Val[i][j][0]=C_Val[i][j][0]+D[k[0]][k[1]]
						elif j==k[1]:
							C_Val[i][j][1]=C_Val[i][j][1]+D[k[0]][k[1]]
						elif i in l and j in l:
							
							C_Val[i][j][2]=C_Val[i][j][2]+D[k[0]][k[1]]
							C_Val[i][j][3]=C_Val[i][j][3]+f[k[0]][k[1]]
							C_Val[i][j][4]=C_Val[i][j][4]+len(l)
							P[i][j].append(l)
							#print P[i][j]
						D_NP.remove(k)

					except:
						continue
				
				U[i,j]= C_Val[i][j][3]-(C_Val[i][j][4])+C_Val[i][j][0]+len(G.neighbors(j))

		

	LOG.info('Channel Value Calculated')
	return C_Val,U,P
	


def Play_Add(G,N,B,U,C_Val):
	for i in N:
			max_u= np.sort(U[i,:])
			max_u= max_u[::-1]
			#print max_u
			m=0
			while(B[i]>0 and m<NUMBER_OF_NODES):
				j= U[i,:].tolist().index(max_u[m])
				cost=C_Val[i][j][0]+C_Val[i][j][2]
				if U[i,j]>0 and cost<B[i]:
					G.add_edge(i, j)
					B[i]=B[i]-cost
				m=m+1


	LOG.info('Add Played')
	return G,B
				
def Play_Remove(G,Path_E,U,C_Val,B):
	for i in N:
		for j in N:
			if C_Val[i][j][0]==0 and C_Val[i][j][3]>0:
				 	try:
				 		G.remove_edge(i,j)
				 		if U[i,j]>0 and cost<B[i]:
				 			B[i]=B[i]+C_Val[i][j][3]
				 	except:
				 		continue
	LOG.info('Remove Played')
	return G
	

if __name__ == '__main__':
	LOG.info('Creating Graph')
	G,E,N=Init()
	G_1=G.copy()
	B= Init_Bitcoins(N)
	D,f= Init_Tansactions(B, N)
	""" "
	
	Setup above this point
	"""

	E=0
	while E!=G_1.edges():
		E=G_1.edges()
		D_NP= Check_Paths(G_1, N, D)
		C_Val,U,Path_E=Channel_Val_Util(G_1, N, G_1.edges(), D,f, D_NP)
		G_1,B_1=Play_Add(G_1, N, B, U, C_Val)
		G_1=Play_Remove(G_1,Path_E,U,C_Val,B_1)



	print sorted(G.degree_iter(),key=itemgetter(1),reverse=True)
	print sorted(G_1.degree_iter(),key=itemgetter(1),reverse=True)



	f, (ax1, ax2) = plt.subplots(1, 2)
	plt.figure(1)
	nx.draw(G,ax=ax1)
	nx.draw(G_1,ax=ax2)
	plt.savefig('./Node_Add_Strategy.pdf')
	plt.show()


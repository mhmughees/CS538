#/usr/bin/python

import logging,itertools,os
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from operator import itemgetter


"""============================================"""

""""
TODO:
	After removing edge check the changes in utility of i
	Relation of removed link to the channel
	Effect of transaction fee
	
Assumption:
No node can send transactions worth more than his total bitcoins.
"""

logging.basicConfig()
LOG = logging.getLogger(__name__)
try:
    os.remove('./ligh_network.log')
except OSError:
    pass
hdlr = logging.FileHandler('./ligh_network.log')
LOG.addHandler(hdlr)
LOG.setLevel(logging.INFO)      


NUMBER_OF_NODES=100
PPROB_OF_EDGE=0.005 # very low otherwise there will be always path
PER_NODE_TRANSACS=100
MAX_BITCOINS=50
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
    			t_amount=randint(0,MAX_BITCOINS/8)
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
	P=np.zeros((len(N),len(N)))
	P=P.tolist()
	for i in N:
		for j in N:
			P[i][j]=[]
			edge_flag=0
			if i!=j: 
				if G_temp.has_edge(i,j)==False:
					edge_flag=1
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
						#D_NP.remove(k)

					except:
						continue
				if edge_flag==1:
					G_temp.remove_edge(i, j)
				


				
				

		

	LOG.info('Channel Value Calculated')
	return C_Val,P
	

def normalized_utility(C_Val):
	U=np.zeros((len(N),len(N)))
	C_Val_temp= np.copy(C_Val)
	for i in N:
		for j in N:
			C_Val_temp[i][j][3]=C_Val[i][j][3]/ (np.amax(C_Val[:][:][3]) if np.amax(C_Val[:][:][3])!=0 else 1)
			C_Val_temp[i][j][4]=C_Val[i][j][4]/ (np.amax(C_Val[:][:][4]) if np.amax(C_Val[:][:][4])!=0 else 1)
			C_Val_temp[i][j][0]=C_Val[i][j][0]/ (np.amax(C_Val[:][:][0]) if np.amax(C_Val[:][:][0])!=0 else 1)
			neighbors= len(G.neighbors(j))/len(N)
			U[i,j]= C_Val_temp[i][j][3]-(C_Val_temp[i][j][4])+C_Val[i][j][0]+neighbors 

	return U


def Play_Add(G,N,B,U,C_Val,Path_E):
	for i in N:
			max_u= np.sort(U[i,:])
			max_u= max_u[::-1]
			#print max_u
			m=0
			while(B[i]>0 and m<NUMBER_OF_NODES):
				j= U[i,:].tolist().index(max_u[m])
				#print j
				cost=C_Val[i][j][0]+C_Val[i][j][2]
				if U[i,j]>0 and cost<B[i]:
					if Path_E[i][j]==['hahahahaha']:
						print i,j,C_Val[i][j][3],(C_Val[i][j][4]),C_Val[i][j][0],len(G.neighbors(j)),U[i,j]
					G.add_edge(i, j)
					B[i]=B[i]-cost
				m=m+1


	LOG.info('Add Played')
	return G,B
				
def Play_Remove(G,Path_E,U,C_Val,B):

	for i in N:
		pred=G.predecessors(i)
		for j in pred:

			cost=C_Val[j][i][0]+C_Val[j][i][2]
			if C_Val[j][i][1]==0 and cost > B[i]:
				G.remove_edge(j, i)
				B[j]+cost
			elif cost < B[i]:
				B[i]=B[i]-cost

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
	r=0
	while E!=G_1.edges():
		LOG.info('Game Round: '+str(r))
		r=r+1
		E=G_1.edges()
		D_NP= Check_Paths(G_1, N, D)
		C_Val,Path_E=Channel_Val_Util(G_1, N, G_1.edges(), D,f, D_NP)
		U=normalized_utility(C_Val)
		G_1,B_1=Play_Add(G_1, N, B, U, C_Val,Path_E)
		G_1=Play_Remove(G_1,Path_E,U,C_Val,B_1)



	print G.in_degree(N)
	print G_1.in_degree(N)
	print U[:,0]

	nx.write_gexf(G, "G.gexf")
	nx.write_gexf(G_1, "G_1.gexf")

	f, (ax1, ax2) = plt.subplots(1, 2)
	plt.figure(1)
	nx.draw(G,ax=ax1)
	nx.draw(G_1,ax=ax2)
	plt.savefig('./Node_Add_Strategy.pdf')
	plt.show()


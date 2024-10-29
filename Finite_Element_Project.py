import numpy as np
import matplotlib.pyplot as plt


M=200
N=M+2
h=(1)/(N-1)

x=np.linspace(0,1,N)

alpha=0.1
a=0.1

def f(x:float)->float:
    return a*alpha*x+1

def u_exact(x:float)->float:
    return alpha*x+1

"""
######### Sol exacte par résolution d'EDO pour f(x)=1 ####################
a=1
alpha=0
delta= (1/2)*( ( (1 - np.exp(-np.sqrt(a))) / (2*np.sinh(a)) ) - 1)
gamma= (alpha / np.sqrt(a)) + (1/2)*( ( (1 - np.exp(-np.sqrt(a)))/(2*np.sinh(a)) ) -1 )

def f(x):
    return 1

def u_exact(x):
    return gamma*(np.exp(x*np.sqrt(a))) + delta*np.exp(-x*np.sqrt(a)) +1/a
"""

############################################################################

Uext=[]
for i in range(len(x)):
    Uext.append(u_exact(x[i]))
    
plt.plot(x,Uext)
plt.title("Sol. Exacte")
plt.show()

def Mat(N):

    ### Ok après vérif. à la main
    d=np.ones(N)*(2)
    d[0]=1
    d[-1]=1
    d1=np.ones(N-1)*(-1)
    A=(1/h)*(np.diag(d)+np.diag(d1,1)+np.diag(d1,-1))
    
    ### Ok après vérif. à la main par assemblage
    dd=np.ones(N)*(2)
    dd1=np.ones(N-1)*(1)
    B=(h/6)*(np.diag(dd)+np.diag(dd1,1)+np.diag(dd1,-1))

    return A+a*B

A=Mat(N)
print("A",A,"\n")

############# Méthodes d'intégrations ############

def mid_point(M:int)->list:
    F = np.zeros(M)
    h = 1/(M+1)
    F[0]=  -alpha
    for i in np.arange(M):
        F[i] += (f((i+1)*h)*h)
    F[-1]+= alpha
    return F

def trapeze(M:int)->list:
    F=np.zeros(M)
    h=1/(M+1)
    x=np.linspace(0,1,M)
    F[0]=(h/2)*f(x[0]) - alpha
    for i in range(1,M-2):
        F[i]=(h)*f(x[i])
    F[-1]=(h/2)*f(x[-1]) + alpha
    return F

def simpson(M:int)->list:
    F=np.zeros(M)
    h=1/(M+1)
    x=np.linspace(0,1,M)
    F[0]=(h/6)*(f(x[0]) + 2*f((x[0]+x[1])/2)) - alpha
    for i in range(1,M-2):
        F[i]=(h/6)*(f((x[i]+x[i+1]) /2) - f((x[i-1]+x[i]) /2) + f(x[i]))
    F[-1]=(h/6)*( 2*f( (x[-2] +x[-1] )/ 2 ) + f(x[-1])) + alpha
    return F
    
#########################################

def Norm_H1(u:list,v:list,A:list,alpha:float)->list:
    UU=[]
    for i in range(len(u)):
        UU.append((u[i]-v[i])**2 +(A-alpha)**2)
        #UU.append((u[i]-v[i])**2)
    sum=0
    for j in range(len(u)-1):
        sum+= (h/6)*(UU[j] + UU[j+1] + 4*(((u[j]-v[j]) + u[j+1]-v[j+1])/2)**2)
    return np.sqrt(sum)

############ Phase de test du code ##########

Fh=simpson(N)

plt.plot(x,Fh)
plt.title("L(vh)")
plt.show()

Uh=np.dot(np.linalg.inv(A),Fh)

plt.plot(x,Uh)
plt.plot(x,Uext)
plt.title("Sol. Approchée par méthode de Simpson")
plt.legend(["Sol. approchée", "Sol. exacte"])
plt.show()


### Norme L2 de u-uh ######
def err(u:list,v:list)->float:
    sum=0
    UU=[]
    for i in range(len(u)):
        #sum+=abs(u[i]-v[i])**2
        UU.append((u[i]-v[i])**2)
    for j in range(len(u)-1):
        sum+= (h/6)*(UU[j] + UU[j+1] + 4*(((UU[j]+UU[j+1])/2)**2))
    return np.sqrt(sum)

#E=err(Uext,Uh)



###############################################################################

################################ Boucle #######################################

NN=[60,70,90,100,200,500,1000,1200,1500,1800,2000]
#NN=[2000,1800,1500,1200,1000,500,200,100,90,70,60]

UU=[] #Stocke les Uh
EE=[] #Stock les erreurs
HH=[] #Stock les h
YY=[] #Stock les maillages
AA=[] #Stock les alpha
EEH1=[] #Stock les normes H1

for i in range(len(NN)):
    y=np.linspace(0,1,NN[i])
    YY.append(y)
    h=1/(NN[i]+1)
    A=Mat(NN[i])
    Fh=simpson(NN[i])
    Uext=[]
    for j in range(len(y)):
        Uext.append(u_exact(y[j]))
    U=Uext
    Uh=np.dot(np.linalg.inv(A),Fh)
    #Uh=np.linalg.solve(A,Fh)
    UU.append(Uh)
    EE.append(err(U,Uh))
    
    HH.append(1/(NN[i]+1))
    AA.append((Uh[-1]-Uh[0]))
    EEH1.append(Norm_H1(Uh, U,AA[i],alpha))
    
    plt.plot(y,Uh)
plt.title("Graphes des sol. approchés")
plt.legend(["60 noeuds","70 noeuds","90 noeuds","100 noeuds","200 noeuds","500 noeuds","1000 noeuds","2000 noeuds"])
plt.show()

######################## Affiche le comportement de l'erreur ##################
plt.plot(HH,EE)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("h")
plt.ylabel("Erreur")
plt.title("Comportement de l'erreur en base log")
plt.show()

########################## Affiche les alpha approchés ########################
plt.plot(HH,AA)
plt.title("Alpha approché")
plt.ylabel("Alpha approché")
plt.xlabel("h")
plt.show()

###############################################################################
######### Affiche la norme H1 ##########
plt.plot(NN,EEH1)
plt.xlabel("N")
plt.yscale('log')
plt.xscale('log')
plt.title("Norme H1 ||U - Uh||")
plt.show()

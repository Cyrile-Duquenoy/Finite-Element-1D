import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate # pour l'intégration du second membre
pi=np.pi

M=500
x=np.linspace(0,1,M)
h=1/(M+1)
print(x)

MM=[20,50,70,100,200]

########################### SOlution Exacte au probleme #######################

                            ####### Remarque ######
                            ## Alpha dépend de a ##
                            #######################

alpha = 0
a = 2


def u_exact(x,alpha):
    return 1/a


"""
def u_exact(x,alpha):
    return (x**2)/2 - (x**3)/3 + alpha*x
"""


def f(x):
    return 1


"""
def f(x):
    return (-a/3)*x**3 + (a/2)*x**2 + (alpha*a+2)*x-1
"""

def Fext(M):
    F=[]
    for i in range(len(x)):
        F.append(f(x[i]))
    return F

Uext=[]
for i in range(len(x)):
    Uext.append(u_exact(x[i],alpha))
    
print(Uext)

Fext=Fext(M)

plt.plot(x,Uext)
plt.title("Solution exacte pour alpha=0.5")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.show()

plt.plot(x,Fext)
plt.title("Fonction source f")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

############################## a(phi_i,phi_j) #################################

def Mat(M):

    #diagonale
    d=np.ones(M)*(2/h)
    d[0]= 1/h - (a*h)/3
    d[M-1]= 1/h + (a*h)/3  
    
    
    #diagonale inferieure/supérierue
    d1=np.zeros(M-1)
    for i in range(1,len(d1)-1):
        d1[i]=-1/h + a/(h**2)*( (x[i]/2)*(x[i]**2 - x[i-1]**2) - (1/3)*(x[i]**3 -x[i-1]**3) -h*x[i]*x[i-1] + (x[i-1]/2)*(x[i]**2 -x[i-1]**2)  )
    d1[0]=-1/h + a/(h**2)*( (x[1]/2)*(x[1]**2 - x[0]**2) - (1/3)*(x[1]**3 -x[0]**3) -h*x[1]*x[0] + (x[0]/2)*(x[1]**2 -x[0]**2)  )
    d1[len(d1)-1]=-1/h + a/(h**2)*( (x[len(x)-1]/2)*(x[len(x)-1]**2 - x[len(x)-2]**2) - (1/3)*(x[len(x)-1]**3 -x[len(x)-2]**3) -h*x[len(x)-1]*x[len(x)-2] + (x[len(x)-2]/2)*(x[len(x)-1]**2 -x[len(x)-2]**2)  )
    
    A=np.diag(d)+np.diag(d1,1)+np.diag(d1,-1)
    return A

Ah=Mat(M)

print(Ah)
###############################################################################

#Apprixamtion de l(v_h)
def Fpy(N): 
    F=np.zeros(N)
    h=1/(N+1)
    X = np.arange(0,N+2)*h   
    for i in np.arange(1,N+1):
        F[i-1] = integrate.quad(lambda X : f(X)*(X-(i-1)*h)/h,(i-1)*h,i*h)[0] 
        F[i-1] += integrate.quad(lambda X: f(X)*((i+1)*h-X)/h,i*h,(i+1)*h)[0]
    return F

def Fquad(M):
    F = np.zeros(M)
    h = 1/(M+1)
    x = np.arange(0,M+2)*h
    for i in np.arange(M):
        F[i] = f((i+1)*h)*h
    return F

def simpson(N):
    F=[]
    h=1/(N+1)
    sum=0
    for i in range(0,N):
        xx=x[i]
        a=f(xx+i*h)*((xx+i*h-(i-1)*h)/h)
        b=4*(f(xx+( i+ 1/2  )/h))*((xx+( i+ 1/2  )/h -(i-1)*h)/h)
        c=f(xx+(i+1)/h)*((xx+(i+1)/h -(i-1)*h)/h)
        sum+=(a+b+c)
        F.append((h/6)*sum)
    return F

Fh=Fquad(M)
print("Fh", Fh,"\n")

plt.plot(x,Fh)
plt.title("Approximation de l(v)")
plt.show()

###############################################################################

####################### Resolution du système #################################

#Uh=np.linalg.solve(Ah,Fh)
Uh=np.dot(np.linalg.inv(Ah),Fh)

plt.title("Sol. approchée par Methode Galerkin")
plt.plot(x,Uh)
#plt.plot(x,Uext)
plt.show()

print(Uh)

def err(U,Uh):
    E=0
    for i in range(len(U)):
        E+=abs(U[i]-Uh[i])**2
    return np.sqrt(E)

Err=err(Uext,Uh)

print(Err)
        
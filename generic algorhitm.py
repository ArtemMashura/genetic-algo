import random
import math
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

data = []
with open("sppnw3.txt") as f:
    for line in f:
        data.append([int(x) for x in line.split()])
M = data[0][0]
N = data[0][1]
C = []
A = []
for i in range(N+1):
    A.append([]); rab=[]
    if i > 0:
        C.append (data[i][0])
        t = len(data[i])
        for j in range(t):
            if j>1:
                z = int(data[i][j]) - 1
                rab.append(z)
        for k in range (M):
            if k in (rab):
                A[i-1].append(1)
            else:
                A[i-1].append(0)

A.pop()
exp_pers_items = M / N
shtraf =  sum(C) / len(C)
mu = 50 #int(input('Number of parents in each age is '))
lam = 2 #int(input('Number of children in each age is '))
mut_level_0 = 5 #float(input('Mutation level is '))
P_mut_min = 0.4
P_mut_max = 0.99 #float(input('Mutation probability is '))
P_cross = 0.9 #float(input('Crossover probability is '))
best=[]
best_owerjob = []
tmax = int(input('Number of evolution age '))


def fitness(C, A, agent):
    f = 0; owerstack = np.zeros(M, dtype=int)
    for j in range(len(C)):
        if agent[j] > 0:
            f = f + C[j]
            for k in range(M):
                if A[j][k]==1:
                    owerstack[k] += 1
    owerjob = 0
    for k in range(M):
        owerjob = owerjob + abs(owerstack[k]-1)
    f = f + shtraf * owerjob            
    return f, owerjob        
    
def generate (mu):
    xi = []
    for j in range(mu):
        xi.append([])
        for i in range(N):
            if random.random() < exp_pers_items:
                xi[j].append(1)
            else:
                xi[j].append(0)  
    return xi

def selection_r (Z):
#Rank selection
    parents = []
    P_det = np.zeros(len(X), dtype=float)
    s = P_det
    f = np.zeros(len(X), dtype=int)
    for i in range (mu):
        f[i] = (mu-i)**2
    s_f = sum(f)
    for i in range (mu):
        P_det[i] = f[i] / s_f
        if i>0:
            s[i] = s[i-1] + P_det[i]
        else:
            s[i] = P_det[i]
    count = 0
    for d in range (lam):
        num = random.random()
        while s[count] < num:
            count = count+1
        parents.append(Z[count][0:N])
    return parents


def crossover(parents):
    #Twopoint crossover
    if random.random() < P_cross:
        b_point = random.randint(0,mu)
        c_point = random.randint(b_point, mu)
        child_1 = []
        child_2 = []
        for i in range(N):
            if i < b_point and i > c_point:
                child_1.append(parents[0][i])
                child_2.append(parents[1][i])
            else:
                child_1.append(parents[1][i])
                child_2.append(parents[0][i])
    else:
        child_1 = parents[0]
        child_2 = parents[1]     
    return(child_1, child_2)

def mutation(children, P_mut, mut_level):
    Y = []
    for j in range(len(children)):
        d = children[j]
        if random.random() < P_mut:
            for i in range(mut_level):
                num_mut = random.randint(0,len(d)-1)
                d[num_mut] = 1 - d[num_mut]
        Y.append(d)
    return Y

def new_population (Z, H): 
    W = []
    for z in range(mu-lam):
         W.append(Z[z][0:N])
    for z in range(lam):
         W.append(H[z])
    return W

def local_search(child):
    star = child; f_old = fitness(C,A,child)
    for i in range(N):
        star[i] = 1 - star[i]
        f_new = fitness(C, A, star)
        if f_new > f_old:
            star[i] = 1 - star[i]
        else:
            f_old = f_new
    return star
                

#generating the firt population
X = generate(mu)
#print(X)
for t in range(tmax):
    if t % (tmax / 10) == 0 and t != 0:
        print(t / (tmax / 100))
    mut_level = math.ceil(mut_level_0*(tmax-t+1)/tmax)
    P_mut = P_mut_min+(P_mut_max-P_mut_min)*t/tmax
    F = []
    for i in range(mu):
        agent = X[i]
        arr, owerjob = fitness(C,A,agent)
        F.append(arr)
    Z = []; 
    for i in range(mu):
        Zi = X[i]
        Zi.append(F[i])
        Z.append(Zi)
    Z.sort(key = itemgetter(N),reverse = False)
    best.append(Z[0])
    best_owerjob.append(owerjob)
    parents = selection_r(Z)
    children = crossover(parents)
    Y = mutation(children, P_mut, mut_level)
    H=[]
    for j in range (lam):
        child = Y[j]
        H.append(local_search(child))    
    X = new_population (Z, Y)
q=len(best)
x = np.arange(0, q, 1)
y = []
for i in range(q):
        k = len(best[i])
        y.append(best[i][k-1])
final_owerjob = y.index(min(y))
print(final_owerjob)
print('Best solution selected', min(y))
print(best_owerjob[final_owerjob])
print(f'Number of parents in each age is {mu}')
plt.style.use('classic')
plt.plot(x,y)
plt.show()

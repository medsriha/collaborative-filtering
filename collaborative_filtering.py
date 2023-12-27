import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib
matplotlib.use("svg")



import matplotlib.pyplot as plt
plt.style.use("ggplot")

movies = pd.read_csv("ml-latest-small/movies.csv")
movies.head()


ratings = pd.read_csv("ml-latest-small/ratings.csv")
print (len(ratings))
ratings.head()


import math
def process(ratings, movies, P):
    """ Given a dataframe of ratings and a random permutation, split the data into a training 
        and a testing set, in matrix form. 
        
        Args: 
            ratings (dataframe) : dataframe of MovieLens ratings
            movies (dataframe) : dataframe of MovieLens movies
            P (numpy 1D array) : random permutation vector
            
        Returns: 
            (X_tr, X_te, movie_names)  : training and testing splits of the ratings matrix (both 
                                         numpy 2D arrays), and a python list of movie names 
                                         corresponding to the columns of the ratings matrices. 
    """
    splitpoint = int(math.floor(9.0*len(P)/10))

    
    users = sorted(set(ratings['userId']))
    items = sorted(set(ratings['movieId']))
    movies = movies.set_index('movieId')
    idMovieDict = movies.loc[:,['title']].to_dict()['title']
    
    train = ratings.iloc[P[:splitpoint],:]
    
    test = ratings.iloc[P[splitpoint:],:]

    def pivotAndArray(df):
        pivot = pd.pivot_table(df, values = 'rating', index=['userId'], columns=['movieId'], margins=False)
        
        pivotdf = pd.DataFrame(pivot).fillna(0)

        for item in items:
            if item not in pivotdf:
                pivotdf[item] = np.zeros(len(pivotdf))

        for user in users:
            if user not in pivotdf.index:
                pivotdf.loc[user,:] = np.zeros(len(pivotdf.columns))

        pivotdf = pivotdf.loc[users,items]
        
        Xarr = pivotdf.values
        return Xarr
        
    X_tr = pivotAndArray(train)
    X_te = pivotAndArray(test)
    movie_names = [idMovieDict[item] for item in items]
     
    return X_tr, X_te, movie_names
    

X_tr, X_te, movieNames = process(ratings, movies, np.arange(len(ratings)))
print (X_tr.shape, X_te.shape, movieNames[:5])
pd.DataFrame(data={'row':np.nonzero(X_tr)[0], 'col': np.nonzero(X_tr)[1]}).to_csv('X_tr_test.csv', sep=',')
pd.DataFrame(data={'row':np.nonzero(X_te)[0], 'col': np.nonzero(X_te)[1]}).to_csv('X_te_test.csv', sep=',')




def error(X, U, V):
    """ Compute the mean error of the observed ratings in X and their estimated values. 
        Args: 
            X (numpy 2D array) : a ratings matrix as specified above
            U (numpy 2D array) : a matrix of features for each user
            V (numpy 2D array) : a matrix of features for each movie
        Returns: 
            (float) : the mean squared error of the observed ratings with their estimated values
        """
    def binarize(x):
        x = float(x)
        if x==0.0:
            return 0.0
        else:
            return 1.0
    W = pd.DataFrame(X)
    W = np.array(W.applymap(binarize))
    h0 = U.dot(V.T)
    sqerr = (h0 - X)**2.0
    sqerr = np.multiply(sqerr,W)
    sqerr = np.array(pd.DataFrame(sqerr))
    mse = sqerr.sum().sum()/W.sum().sum()
    return mse

def train(X, X_te, k, U, V, niters=51, lam=10, verbose=False):
    """ Train a collaborative filtering model. 
        Args: 
            X (numpy 2D array) : the training ratings matrix as specified above
            X_te (numpy 2D array) : the testing ratings matrix as specified above
            k (int) : the number of features use in the CF model
            U (numpy 2D array) : an initial matrix of features for each user
            V (numpy 2D array) : an initial matrix of features for each movie
            niters (int) : number of iterations to run
            lam (float) : regularization parameter
            verbose (boolean) : verbosity flag for printing useful messages
            
        Returns:
            (U,V) : A pair of the resulting learned matrix factorization
    """
    k = U.shape[1]
    lam = float(lam)
    def binarize(x):
        x = float(x)
        if x==0.0:
            return 0.0
        else:
            return 1.0
    W = pd.DataFrame(X)
    W = np.array(W.applymap(binarize))
    
    if verbose:
        print ("Iter \t|Train Err \t\t|Test Err")

    for i_num in range(niters):
        

        for i in range(U.shape[0]):
            Wi = W[i]
            Wik = np.array([Wi for ki in range(k)]).T
                

            Vz = np.multiply(V,Wik)

            firstterm = np.dot(Vz.T, Vz) + lam*np.eye(k)
            secondterm = np.dot(Vz.T,X[i])
            
            U[i] = np.linalg.solve(firstterm, secondterm)     
            

        for j in range(V.shape[0]):
            Wj = W[:,j]
            Wjk = np.array([Wj for ki in range(k)]).T
            

            Uz = np.multiply(U,Wjk)
    

            firstterm = np.dot(Uz.T, Uz) + lam*np.eye(k)
            secondterm = np.dot(Uz.T,X[:,j])
            
            V[j] = np.linalg.solve(firstterm, secondterm)
            


        if verbose and (i_num % 5 == 0):
            print (i_num,"\t|", error(X, U, V),"\t|",error(X_te, U, V))
               
    return (U,V)


k=5
np.random.seed(1)
U = np.random.rand(X_tr.shape[0],k)

np.random.seed(1)
V = np.random.rand(X_tr.shape[1],k)
print (U.shape)
print (V.shape)
print (U, V)

train(X_tr, X_te, k, U, V, niters=2, lam=10, verbose=True)

vsum = sum(np.dot(V[j,None].T,V[j,None]) for  j in range(V.shape[0])) + np.eye(5)*10
print (vsum)
v1 = pd.DataFrame(V[0])
v2 = pd.DataFrame(V[0]).transpose()
print (v1.shape)
print (v2.shape)
print (v1.dot(v2))

vdf = pd.DataFrame(V)

np.array([V[0] for k in range(5)]).T
sum(V[0]**2)

def binarize(x):
    x = int(x)
    if x==0:
        return 0.0
    else:
        return 1.0
test = pd.DataFrame(np.array([[1,0,3],[4,0.0,6],[7,0,0]]))
test2 = pd.DataFrame(np.array([[7,7,7],[7,7,7],[7,7,7]]))
bintest = test.applymap(binarize)
print (test)
print (bintest)
print (type(test))
print (type(bintest))
print (np.multiply(test2,bintest))
print (sum(np.array([[7,7,7],[7,7,7],[7,7,7]])))


a = np.array([[1,0,3],[4,0.0,6],[7,0,0]])
a[1,2]
b = np.array([1,1,1])
np.dot(b,b.T)



def recommend(X, U, V, movieNames):
    """ Recommend a new movie for every user.
        Args: 
            X (numpy 2D array) : the training ratings matrix as specified above
            U (numpy 2D array) : a learned matrix of features for each user
            V (numpy 2D array) : a learned matrix of features for each movie
            movieNames : a list of movie names corresponding to the columns of the ratings matrix
        Returns
            (list) : a list of movie names recommended for each user
    """
    def binarizeFlipped(x):
        x = int(x)
        if x==0:
            return 1.0
        else:
            return 0.0
    W = pd.DataFrame(X)
    W = np.array(W.applymap(binarizeFlipped))
    
    X_new = np.multiply(U.dot(V.T),W)
    X_new= pd.DataFrame(X_new, columns=movieNames)
    recs = X_new.idxmax(axis=1)
    
    return recs.tolist()

recommendations = recommend(X_tr, U, V, movieNames)
# 10 top recommendations
print (recommendations[:10])


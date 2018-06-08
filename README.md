## Collaborative Filtering by Matrix Factorization
In collaborative filtering we wish to factorize our ratings matrix into two smaller feature matrices whose product is equal to the original ratings matrix. Specifically, given some partially filled ratings matrix $X\in \mathbb{R}^{m\times n}$, we want to find feature matrices $U \in \mathbb{R}^{m\times k}$ and $V \in \mathbb{R}^{n\times k}$ such that $UV^T = X$. In the case of movie recommendation, each row of $U$ could be features corresponding to a user, and each row of $V$ could be features corresponding to a movie, and so $u_i^Tv_j$ is the predicted rating of user $i$ on movie $j$. This forms the basis of our hypothesis function for collaborative filtering: 

$$h_\theta(i,j) = u_i^T v_j$$

In general, $X$ is only partially filled (and usually quite sparse), so we can indicate the non-presence of a rating with a 0. Notationally, let $S$ be the set of $(i,j)$ such that $X_{i,j} \neq 0$, so $S$ is the set of all pairs for which we have a rating. The loss used for collaborative filtering is squared loss:

$$\ell(h_\theta(i,j),X_{i,j}) = (h_\theta(i,j) - X_{i,j})^2$$

The last ingredient to collaborative filtering is to impose an $l_2$ weight penalty on the parameters, so our total loss is:

$$\sum_{i,j\in S}\ell(h_\theta(i,j),X_{i,j}) + \lambda_u ||U||_2^2 + \lambda_v ||V||_2^2$$

For this assignment, we'll let $\lambda_u = \lambda_v = \lambda$. 

## MovieLens rating dataset
To start off, let's get the MovieLens dataset. This dataset is actually quite large (24+ million ratings), but we will instead use their smaller subset of 100k ratings. You will have to go fetch this from their website. 

* You can download the archive containing their latest dataset release from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip (last updated October 2016). 
* For more details (contents and structure of archive), you can read the README at http://files.grouplens.org/datasets/movielens/ml-latest-README.html
* You can find general information from their website description located at http://grouplens.org/datasets/movielens/. 

We will only be looking at the ratings data specifically. However, it is good to note that there is additional data available (i.e. movie data and user made tags for movies) which could be used to improve the ability of the recommendation system. 

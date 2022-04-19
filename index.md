Hello and welcome to this blog. Edit the `index.md` file to change this content. All pages on the blog, including this one, use [Markdown](https://guides.github.com/features/mastering-markdown/). You can include images:

![Image of fast.ai logo](images/logo.png)

## This is a title
$x + y = b$ in tttttt

$$ 1 + 3 = 4$$
And you can include links, like this [link to fast.ai](https://www.fast.ai). Posts will appear after this file. 

## Algorithm

### Procedure

**2-Way spectral clustering**

1. Compute the Laplacian matrix $L$ of the graph.
2. Compute the eigenvectors and eigenvalues of $L$. Sort the eigenvalues by $\lambda_1\leq \lambda_2 \leq \dots$.
3. Choose splitting points
   1. Naive approach: split at 0. Assign labels to vertices based on the sign of coordinates of $\lambda_2$
   2. More expensive approaches: ...

**k-way spectral clustering**

One approach: Apply 2-way repeatedly until a total of $k$ clusters have been found.

Another approach: 

1. Compute the first $k$ eigenvectors $\lambda_1,\dots, \lambda_k$ of $L$.
2. Let $U\in R^{n\times k}$ be the matrix containing $\lambda_1,\dots, \lambda_k$ as columns.
3. K-means ...

### Theoretical Guarantee

For symmetric matrix $L$, we have
$$
\lambda_2=\min_{x:x^T\lambda_1=0} \frac{x^TLx}{x^Tx}
$$
where $\lambda_1$ is the first eigenvector of $L$, namely, $\lambda_1=\begin{bmatrix} 1&\dots 1\end{bmatrix}$.

**Proof**



Since,
$$
x^TLx = \sum_{(i,j)\in E}(x_i-x_j)^2
$$

And $x$ is orthogonal to $\lambda_1$, giving $\sum_i x_i=0$, we have
$$
\lambda_2=\min_{\sum_i x_i=0} \frac{\sum_{(i,j)\in E}(x_i-x_j)^2}{\sum_i x_i^2}
$$

Therefore, the approach of using $\lambda_2$ to cluster vertices into 2 sets in 2-way spectral clustering is aimed to minimizing the distances between the two sets $A$ and $B$, where $i\in A$ and $j\in B$.

**Approximation guarantee** 


The **conductance** of a subset $S\subset V$ can be defined as

$$
\phi(S)=\frac{|E(S, \bar{S})|}{d\min\left\{|S|, |\bar{S}|\right\}}
$$

where $E(S, \bar{S})$ denotes the set of edges of $G$ crossing from $S$ to its complement.

The conductance of the graph $G$ is defined as

$$
\phi(G) = \min_{S\subset V}\phi(S)
$$

Cheeger's Inequality states that for any graph $G$,

$$
\frac{\lambda_2}{2} \leq \phi(G) \leq \sqrt{2\lambda_2}
$$

Therefore, using this approach, the 2-way spectral clustering algorithm is able to find a cut that has at most twice the conductance as the optimal one of conductance $\phi(G)$.

### Time Complexity

1. $O(n^3)$ → find the eigenvalues and eigenvectors

2. Fast approximate spectral clustering https://people.eecs.berkeley.edu/~jordan/papers/yan-etal-long.pdf

![image-20220418215442089](C:%5CUsers%5Chelli%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220418215442089.png)

​		$O(k^3)+O(knt)$, where $t$ is the number of iterations

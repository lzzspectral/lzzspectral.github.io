Hello and welcome to this blog. Edit the `index.md` file to change this content. All pages on the blog, including this one, use [Markdown](https://guides.github.com/features/mastering-markdown/). You can include images:

![Image of fast.ai logo](images/logo.png)

## This is a title
$x + y = b$ in tttttt

$$ 1 + 3 = 4$$
And you can include links, like this [link to fast.ai](https://www.fast.ai). Posts will appear after this file. 

## Introduction \& Graph Cut

Suppose we have an undirected graph $G(V,E)$, and we would like to perform a bi-partitioning task on the graph, i.e., dividing vertices in $V$ into two disjoint groups $A$ and $B$.

How to define a good partition?

1. Maximize the number of within-group
connections.
2. Minimize the number of between-group connections.

![Example of a good partition]{images/good partition.png}

Definition (Graph Cuts): Set of edges with one endpoint in each group:

$$cut(A,B)=\Sigma_{i\in A, j\in B}w_{ij}$$

where, if the graph is weighted $w_{ij}$ is the weight, otherwise, all $w_{ij}\in \{0,1\}$.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.5\linewidth]{graph cuts.png}
    \label{fig:my_label}
\end{figure}

\vspace{0.5cm}

In this blog, we mainly focus on the unweighted graph.

\vspace{0.5cm}
So, what is a good graph cut criterion?

    \begin{enumerate}
      \item \textbf{Minimum-cut}: Minimize weight of connections between groups.
      $$\arg \min_{A,B}cut(A,B)$$
      Drawbacks:
        \begin{itemize}
        \item Only considers external cluster connections.
        \item Does not consider internal cluster connectivity.
        \end{itemize}
      \item \textbf{Conductance}: Connectivity between groups relative to the density of each group.
      $$\Phi(A,B)=\frac{cut(A,B)}{\min(vol(A),vol(B))}$$
      
      where $vol(A)$ = total degree of the nodes in $A$ = number of edge end points in A. And we are trying to find the minimal conductance.
      
      Advantages: Conductance produces more balanced partitions. The reason is as following: The total number of edges in the graph $G$ is fixed. After bi-partitioning, we split the total degree of all nodes in $G$ into two parts, and take the smaller one among these two as the denominator of the conductance. Therefore, a smaller conductance tends to give us a more balanced partition.
    \end{enumerate}

\section{Graph Laplacian matrix}
There are three main matrix representations that we care about: Adjacency matrix, degree matrix, and laplacian matrix.
\begin{itemize}
    \item Adjacency matrix ($A$):
    $A$ is a $n \times n$ matrix, where $n = |V|$, and it defines as:
    
    $$A = [a_{ij}], a_{ij} = 1 \text{ if there is an edge between $i$ and $j$, otherwise, $a_{ij} = 0$.} $$
    There are some important properties of an ajacency matrix $A$: (1) Symmetric matrix, (2) Has $n$ real eigenvalues, (3) Eigenvectors are real-valued and orthogonal.
    
    \textbf{Spectral Graph Theory}: Analyze the “spectrum” of matrix representing $G$.
    
    Definition (Spectrum): Eigenvectors $x^{(i)}$ of a graph ('s ajacency matrix $A$), ordered by the magnitude (strength) of their corresponding eigenvalues $\lambda_i$: $\Lambda = \{\lambda_1,\lambda_2,\cdots,\lambda_n\}$, where $\lambda_1\leq \lambda_2 \leq \cdots \leq \lambda_n$.
    
    \item Degree matrix ($D$): $D$ is a $n \times n$ matrix, where $n = |V|$, and it defines as:
    
    $$D = [d_{ii}], d_{ii} = \text{ degree of node } i, \text{ other elements of $D$ are all $0$.}$$
    \item Laplacian matrix ($L$): $L$ is defined as, another $n \times n$ matrix, the difference between $D$ and $A$, i.e., $L = D - A$.
    
    Note that there is a trivial eigenpair for a laplacian matrix $L$: $x=(1,\cdots,1)$ then $L\cdot x=0$ and so $\lambda=\lambda_1 \text{(smallest eigenvalue)} = 0$.
    
    There are some important properties of a laplacian matrix $L$: (1) Eigenvalues are non-negative real numbers, (2) Eigenvectors are real (and always orthogonal)
\end{itemize}

Let's denote $\lambda_2$ as an optimization problem, where $\lambda_2$ is the second smallest eigenvalue for a symmetric matrix $M$. We have the following fact:

$$\lambda_2=\min_{x:x^Tw_1=0}\frac{x^TMx}{x^Tx}$$

where $w_1$ is the eigenvector corresponding to the smallest eigenvalue $v_1$.

So now we can consider what is the meaning of $\min x^TLx$ on the graph $G$.

$$
\begin{aligned}
x^TLx &=\Sigma^n_{i,j=1}L_{ij}x_ix_j \\
&=\Sigma^{n}_{i,j=1}(D_{ij}-A_{ij})x_ix_j \\
&=\Sigma_iD_{ii}x^2_i-\Sigma_{(i,j)\in E}2x_ix_j\\
& =\Sigma_{(i,j)\in E}(x^2_+x^2_j-2x_ix_j) \\
&=\Sigma_{(i,j)\in E}(x_i-x_j)^2 \\
\end{aligned}
$$

We also know that $x$ has another two properties:

    \begin{itemize}
      \item $x$ is unit vector, i.e., $\Sigma_ix^2_i=1$
      \item $x$ is orthogonal to the eigenvector $[1,\cdots, 1]$. Therefore, $\Sigma_ix_i\cdot 1=\Sigma_ix_i=0$
    \end{itemize}
    
Up to this point, we can convert the original fact about $\lambda_2$ to be:

% $$\lambda_2=\min_{\text{All labelings of nodes $i$ so that $\Sigma_i x_i = 0$}}\frac{\Sigma_{(i,j)\in E}(x_i-x_j)^2}{\Sigma_ix^2_i}$$

$$
\begin{aligned}
\lambda_2&=\min_{\text{All labelings of nodes $i$ so that $\Sigma_i x_i = 0$}}\frac{\Sigma_{(i,j)\in E}(x_i-x_j)^2}{\Sigma_ix^2_i} \\
&=\min_{\text{All labelings of nodes $i$ so that $\Sigma_i x_i = 0$}}\Sigma_{(i,j)\in E}(x_i-x_j)^2
\end{aligned}
$$

So, we want to assign values $x_i$ to nodes $i$ such that few edges cross $0$. (we want $x_i$ and $x_j$ to subtract each other)

\begin{figure}[H]
    \centering
    \includegraphics[width=0.5\linewidth]{balance to minimize.png}
    \label{fig:my_label}
\end{figure}

\section{Rayleigh Theorem}

$$\min_{y\in\mathbb{R}^n:\Sigma_iy_i=0, \Sigma_iy^2_i=1}f(y)=\Sigma_{(i,j)\in E}(y_i-y_j)^2=y^TLy$$

\begin{itemize}
    \item $\lambda_2 = \min_yf(y)$: The minimum value of $f(y)$ is given by the second smallest eigenvalue $\lambda_2$ of the Laplacian matrix $L$.
    \item $x = \arg\min_yf(y)$: The optimal solution for $y$ is given by the eigenvector $x$ corresponding to $\lambda_2$, referred to as the Fiedler vector.
    \item \textbf{We can use the sign of $\mathbf{x_i}$ to determine cluster assignment of node $\mathbf{i}$}.
\end{itemize}



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

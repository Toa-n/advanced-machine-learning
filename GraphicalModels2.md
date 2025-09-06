---
title: Graphical Models
markmap:
  colorFreezeLevel: 4
  initialExpandLevel: 2
---
# Graphical Models

## Introduction
- **Definition**: A fusion of probability theory and graph theory to represent and analyze probabilistic models.
- **Purpose**:
    - To visualize the structure of a probabilistic model.
    - To discover properties of the model, like conditional independence.
    - To find efficient ways to compute probabilities (inference).
- **Key Idea**: The graph structure (nodes and edges) defines the dependencies and independences between random variables. The *missing* edges are crucial as they represent simplifying assumptions.

## Key Components & Types
- **Nodes**: Represent random variables (can be discrete, continuous, binary).
    - Unobserved (unknown) nodes are usually empty circles.
    - Observed (known) nodes are shaded circles.
- **Edges**: Represent probabilistic relationships between variables.
- **Two Main Types**:
    - **Directed Graphical Models (Bayesian Networks)**: Edges have a direction, indicating a (causal) dependency.
    - **Undirected Graphical Models (Markov Random Fields)**: Edges are undirected, representing general correlations.

# Directed Graphical Models (Bayesian Networks)
- Based on a **Directed Acyclic Graph (DAG)**.

## Factorization & Joint Probability
- **Problem**: Compute the joint probability distribution over all variables in the network.
- **Formula (General)**: The joint probability is the product of the conditional probabilities of each node given its parents.
    - $p(\mathbf{x}) = \prod_k p(x_k | \text{pa}_k)$
    - where $\text{pa}_k$ are the parents of node $x_k$.
- **Intuition**: The network structure allows us to break down a complex joint probability into smaller, local conditional probabilities.
- **Notes**:
    - This factorization significantly reduces complexity. For $n$ binary variables, brute-force requires $O(2^n)$ terms, while factorization requires $O(n \cdot 2^k)$ terms, where $k$ is the maximum number of parents for any node.

### Example: Chain of Nodes ($a \rightarrow b \rightarrow c$)
- **Formula**: $p(a, b, c) = p(c | b) p(b | a) p(a)$

### Example: Convergent Connection ($a \rightarrow c \leftarrow b$)
- **Formula**: $p(a, b, c) = p(c | a, b) p(a) p(b)$
- **Note**: This assumes $a$ and $b$ are marginally independent.

## Conditional Independence
- **Definition**: A variable $X$ is conditionally independent of $Y$ given $V$ if knowing $V$ makes $X$ independent of $Y$.
- **Notation**: $X \perp Y | V$
- **Formula**: $p(X | Y, V) = p(X | V)$
- **Key Idea**: We can determine conditional independencies directly from the graph structure using D-Separation.

### D-Separation ("Directed Separation")
- **Problem**: Determine if a set of nodes $A$ is conditionally independent of a set $B$ given a set $C$.
- **Algorithm**: A path from any node in $A$ to any node in $B$ is **blocked** if it contains a node $n$ where:
    1.  The arrows on the path meet **head-to-tail** ($\rightarrow n \rightarrow$) or **tail-to-tail** ($\leftarrow n \rightarrow$) at $n$, AND $n$ is in the conditioning set $C$.
    2.  The arrows on the path meet **head-to-head** ($\rightarrow n \leftarrow$) at $n$, AND neither $n$ nor any of its descendants are in the conditioning set $C$.
- **Conclusion**: If all paths from $A$ to $B$ are blocked by $C$, then $A$ is d-separated from $B$ by $C$, which means $A \perp B | C$.

### Three Fundamental Cases for a Path ($a - n - b$)
1.  **Tail-to-Tail (Divergent)**: $a \leftarrow n \rightarrow b$
    - $a$ and $b$ are **NOT** independent.
    - They become **conditionally independent** if $n$ is observed (i.e., given $n$).
2.  **Head-to-Tail (Chain)**: $a \rightarrow n \rightarrow b$
    - $a$ and $b$ are **NOT** independent.
    - They become **conditionally independent** if $n$ is observed.
3.  **Head-to-Head (Convergent)**: $a \rightarrow n \leftarrow b$
    - $a$ and $b$ **ARE** independent.
    - They become **conditionally dependent** if $n$ or any of its descendants are observed.

### Explaining Away
- A phenomenon specific to **head-to-head** connections.
- **Intuition**: If we observe a common effect ($n$), and one of its causes ($a$) is confirmed, our belief in the other cause ($b$) decreases. The confirmed cause "explains away" the effect.

### Bayes Ball Algorithm
- **An intuitive algorithm for determining d-separation.**
- **Algorithm**: To test $X \perp Y | V$:
    - Place "balls" at each node in $X$. Let them travel according to rules, trying to reach $Y$.
    - **Rules (at a node W)**:
        - **Unobserved ($W \notin V$)**:
            - A ball from a parent passes through to all children.
            - A ball from a child is bounced back to all parents and also passed through to all other children.
        - **Observed ($W \in V$)**:
            - A ball from a parent is bounced back to all parents.
            - A ball from a child is blocked.
- **Conclusion**: If no ball from $X$ can reach $Y$, then $X \perp Y | V$.

### Markov Blanket
- **Definition**: The minimal set of nodes that renders a node $x_i$ conditionally independent of all other nodes in the graph.
- **Composition**: The Markov Blanket of $x_i$ includes its:
    - **Parents**
    - **Children**
    - **Co-parents** (other parents of its children)
- **Note**: The "co-parents" part is due to the explaining away phenomenon.

# Undirected Graphical Models (Markov Random Fields - MRFs)
- Given by an undirected graph.

## Conditional Independence
- **Rule**: Much simpler than in Bayesian Networks.
- **Algorithm**: Two sets of nodes $A$ and $B$ are conditionally independent given a set $C$ if **all paths** from $A$ to $B$ pass through at least one node in $C$.
- **Intuition**: If $C$ "separates" $A$ from $B$ in the graph, then $A \perp B | C$.

## Factorization
- More complex than in Bayesian Networks. Relies on the concept of cliques.
- **Clique**: A subset of nodes where every two distinct nodes are connected by an edge.
- **Maximal Clique**: A clique that cannot be extended by adding any other node from the graph.
- **Problem**: Define the joint probability distribution.
- **Formula**: The joint distribution is a product of potential functions $\psi_C$ over the maximal cliques $C$ of the graph.
    - $p(\mathbf{x}) = \frac{1}{Z} \prod_C \psi_C(\mathbf{x}_C)$
    - **Potential Function $\psi_C(\mathbf{x}_C)$**: A non-negative function measuring the "compatibility" or "affinity" of the variables in a clique.
    - **Partition Function $Z$**: A normalization constant to ensure the distribution sums/integrates to 1.
        - $Z = \sum_\mathbf{x} \prod_C \psi_C(\mathbf{x}_C)$
- **Boltzmann Distribution**: A common way to define potential functions using an energy function $E$.
    - $\psi_C(\mathbf{x}_C) = \exp(-E(\mathbf{x}_C))$
- **Notes**:
    - The presence of the partition function $Z$ is a major limitation, as it can be computationally expensive to calculate (requires summing over all possible states of $\mathbf{x}$).
    - Unlike BNs, MRFs are not automatically normalized.

## Converting Directed to Undirected Graphs
- **Process**: **Moralization**
- **Algorithm**:
    1.  **Marry the parents**: For each node, add an undirected edge between all pairs of its parents.
    2.  **Drop the arrows**: Convert all original directed edges to undirected edges.
    3.  The resulting graph is the **moral graph**.
- **Note**: This process can lose conditional independence information that the original directed graph contained (e.g., the independence in a head-to-head structure is lost).

# Inference in Graphical Models
- **Goal**: Evaluate the probability distribution over some variables, given the values of others (observations). E.g., compute $p(\text{Query} | \text{Evidence})$.

## Factor Graphs
- **Motivation**: A representation that makes the factorization of the joint probability explicit, unifying directed and undirected models for inference.
- **Structure**: A bipartite graph with two types of nodes:
    - **Variable nodes** (circles): Represent variables.
    - **Factor nodes** (squares): Represent the factors in the joint probability distribution.
- **Key Property**: Any directed or undirected **tree** converts to a factor graph that is also a tree. A directed **polytree** (which has loops if converted to an undirected graph) converts to a tree-structured factor graph. This allows tree-based inference algorithms to be applied more generally.

## Exact Inference on Trees (Sum-Product Algorithm)
- **Goal**: Efficiently compute all marginals $p(x_i)$ in a tree-structured factor graph.
- **Key Idea**: Message passing. Exploits the distributive law to avoid redundant computations.
- **Algorithm (Two Message Types)**:
    - **Variable-to-Factor message $\mu_{x \rightarrow f}(x)$**:
        - **Formula**: $\mu_{x \rightarrow f}(x) = \prod_{h \in \text{ne}(x) \setminus f} \mu_{h \rightarrow x}(x)$
        - **Intuition**: A variable node tells a factor node what it knows based on information from all *other* connected factors. It is the product of incoming messages.
    - **Factor-to-Variable message $\mu_{f \rightarrow x}(x)$**:
        - **Formula**: $\mu_{f \rightarrow x}(x) = \sum_{\mathbf{X}_f \setminus x} \left[ f(\mathbf{X}_f) \prod_{y \in \text{ne}(f) \setminus x} \mu_{y \rightarrow f}(y) \right]$
        - **Intuition**: A factor node summarizes information from all *other* connected variables, combines it with its own factor, and passes the result to a target variable node.
- **Procedure**:
    1.  Pick an arbitrary root node.
    2.  Pass messages from the leaves up to the root.
    3.  Pass messages from the root back down to the leaves.
- **Marginal Calculation**:
    - **Formula**: $p(x) \propto \prod_{f \in \text{ne}(x)} \mu_{f \rightarrow x}(x)$
    - **Intuition**: The marginal probability of a variable is proportional to the product of all incoming messages from its neighboring factors.

## Finding the Most Probable State (Max-Sum Algorithm)
- **Goal**: Find the configuration $x_{\max}$ that maximizes the joint probability $p(x)$ (the MAP estimate).
- **Key Idea**: A modification of the Sum-Product algorithm. For numerical stability, it operates on log-probabilities.
    - $\sum$ is replaced by $\max$.
    - $\prod$ is replaced by $\sum$.
- **Algorithm**:
    - The message update rules are analogous to Sum-Product, using `max` and `+`.
    - **Backtracking**: To recover the maximizing configuration $x_{\max}$, each maximization step also stores the `argmax`. After the forward pass to the root, a backward pass traces these `argmax` choices back to the leaves.

## Exact Inference on General Graphs (Junction Tree Algorithm)
- **Goal**: Perform exact inference on graphs with loops.
- **Key Idea**: Convert the loopy graph into a tree of cliques (a "junction tree") and then run a message-passing algorithm on this new tree.
- **Algorithm Steps**:
    1.  **Moralization**: If the graph is directed, make it undirected ("marry the parents").
    2.  **Triangulation**: Add edges (chords) to the graph to ensure no cycle of length > 3 exists without a chord.
    3.  **Clique Identification**: Find the maximal cliques of the triangulated graph.
    4.  **Junction Tree Construction**: Build a new graph where nodes are the maximal cliques. Find a maximum spanning tree of this clique graph.
- **Running Intersection Property**: Triangulation ensures that if a variable appears in two nodes (cliques) of the tree, it appears in all nodes on the path between them. This is crucial for consistent message passing.
- **Limitation**: The computational cost is exponential in the size of the largest clique. The algorithm is intractable for graphs with large cliques.

## Approximate Inference (Loopy Belief Propagation)
- **Goal**: Perform approximate inference on general graphs with loops when exact methods are too costly.
- **Algorithm**: Apply the standard Sum-Product message-passing rules directly to the loopy graph. Messages are passed around for a fixed number of iterations or until they converge.
- **Notes**:
    - It is an approximate method.
    - Convergence is not guaranteed.
    - Often works surprisingly well in practice.

# Applications of Bayesian Networks

## Hidden Markov Models (HMMs)
- **Model**: A dynamic Bayesian Network representing a sequence of hidden states ($z_n$) and corresponding observations ($x_n$). Models the probabilities $p(z_n | z_{n-1})$ (transition) and $p(x_n | z_n)$ (emission).
- **Inference Tasks**:
    - **Likelihood Estimation**: $p(X)$ - Solved by the **Forward-Backward Algorithm**, which is a special case of the **Sum-Product algorithm**.
    - **Most Probable State Sequence**: $\arg\max_Z p(Z | X)$ - Solved by the **Viterbi Algorithm**, which is a special case of the **Max-Sum algorithm**.
    - **Parameter Learning**: Solved by the **Baum-Welch algorithm** (an instance of EM).

## Kalman Filters
- **Model**: An HMM with continuous latent states and observations.
- **Assumptions**: Assumes linear-Gaussian dynamics, meaning the transition and emission probabilities are defined by linear functions with added Gaussian noise.

# Applications of Markov Random Fields

## MRFs in Computer Vision
- **Model**: Often used for low-level vision tasks (denoising, segmentation, etc.) on a grid-like graph where nodes are pixels.
- **Energy Formulation**: Inference is framed as an energy minimization problem. The goal is to find the labeling $\mathbf{x}$ that minimizes the total energy $E(\mathbf{x},\mathbf{y})$.
    - **Formula**: $E(\mathbf{x}, \mathbf{y}) = \sum_i \phi(x_i, y_i) + \sum_{i,j} \psi(x_i, x_j)$
        - **Unary Potential $\phi(x_i, y_i)$**: The "data term". Encodes the cost of assigning label $x_i$ to pixel $i$ based on its observed data $y_i$.
        - **Pairwise Potential $\psi(x_i, x_j)$**: The "smoothness term". Encodes the cost for two neighboring pixels $i$ and $j$ to have different labels.
- **Potts Model**: A common pairwise potential that adds a fixed penalty if neighboring labels are different.
    - **Formula**: $\psi(x_i, x_j) = \theta_\psi \cdot \delta(x_i \neq x_j)$

## Energy Minimization with Graph Cuts
- **Goal**: An efficient algorithm to find the globally optimal solution (minimum energy) for certain MRFs.
- **Key Idea**: Transforms the MRF energy minimization problem into a minimum cut problem on a specially constructed source-sink graph.
- **Algorithm**:
    1.  Construct a graph with a source $s$ and a sink $t$.
    2.  Add nodes for each pixel/variable in the MRF.
    3.  Add edges (t-links) from $s$ and to $t$ with weights determined by the unary potentials.
    4.  Add edges between pixel nodes (n-links) with weights determined by the pairwise potentials.
    5.  Find the minimum $s-t$ cut in this graph using a max-flow/min-cut algorithm.
- **Requirement**: The energy function must be **submodular**. This is the discrete equivalent of a convex function, and it guarantees that a min-cut finds the global minimum. For binary labeling problems, this condition is generally met.
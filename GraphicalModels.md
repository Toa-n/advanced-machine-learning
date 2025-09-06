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
    - `p(x) = ∏_k p(x_k | pa_k)`
    - where `pa_k` are the parents of node `x_k`.
- **Intuition**: The network structure allows us to break down a complex joint probability into smaller, local conditional probabilities.
- **Notes**:
    - This factorization significantly reduces complexity. For `n` binary variables, brute-force requires `O(2^n)` terms, while factorization requires `O(n * 2^k)` terms, where `k` is the maximum number of parents for any node.

### Example: Chain of Nodes (a -> b -> c)
- **Formula**: `p(a, b, c) = p(c | b) * p(b | a) * p(a)`

### Example: Convergent Connection (a -> c <- b)
- **Formula**: `p(a, b, c) = p(c | a, b) * p(a) * p(b)`
- **Note**: This assumes `a` and `b` are marginally independent.

## Conditional Independence
- **Definition**: A variable `X` is conditionally independent of `Y` given `V` if knowing `V` makes `X` independent of `Y`.
- **Notation**: `X ⊥ Y | V`
- **Formula**: `p(X | Y, V) = p(X | V)`
- **Key Idea**: We can determine conditional independencies directly from the graph structure using D-Separation.

### D-Separation ("Directed Separation")
- **Problem**: Determine if a set of nodes `A` is conditionally independent of a set `B` given a set `C`.
- **Algorithm**: A path from any node in `A` to any node in `B` is **blocked** if it contains a node `n` where:
    1.  The arrows on the path meet **head-to-tail** (`-> n ->`) or **tail-to-tail** (`<- n ->`) at `n`, AND `n` is in the conditioning set `C`.
    2.  The arrows on the path meet **head-to-head** (`-> n <-`) at `n`, AND neither `n` nor any of its descendants are in the conditioning set `C`.
- **Conclusion**: If all paths from `A` to `B` are blocked by `C`, then `A` is d-separated from `B` by `C`, which means `A ⊥ B | C`.

### Three Fundamental Cases for a Path (a - n - b)
1.  **Tail-to-Tail (Divergent)**: `a <- n -> b`
    - `a` and `b` are **NOT** independent.
    - They become **conditionally independent** if `n` is observed (i.e., given `n`).
2.  **Head-to-Tail (Chain)**: `a -> n -> b`
    - `a` and `b` are **NOT** independent.
    - They become **conditionally independent** if `n` is observed.
3.  **Head-to-Head (Convergent)**: `a -> n <- b`
    - `a` and `b` **ARE** independent.
    - They become **conditionally dependent** if `n` or any of its descendants are observed.

### Explaining Away
- A phenomenon specific to **head-to-head** connections.
- **Intuition**: If we observe a common effect (`n`), and one of its causes (`a`) is confirmed, our belief in the other cause (`b`) decreases. The confirmed cause "explains away" the effect.

### Bayes Ball Algorithm
- **An intuitive algorithm for determining d-separation.**
- **Algorithm**: To test `X ⊥ Y | V`:
    - Place "balls" at each node in `X`. Let them travel according to rules, trying to reach `Y`.
    - **Rules (at a node W)**:
        - **Unobserved (W ∉ V)**:
            - A ball from a parent passes through to all children.
            - A ball from a child is bounced back to all parents and also passed through to all other children.
        - **Observed (W ∈ V)**:
            - A ball from a parent is bounced back to all parents.
            - A ball from a child is blocked.
- **Conclusion**: If no ball from `X` can reach `Y`, then `X ⊥ Y | V`.

### Markov Blanket
- **Definition**: The minimal set of nodes that renders a node `x_i` conditionally independent of all other nodes in the graph.
- **Composition**: The Markov Blanket of `x_i` includes its:
    - **Parents**
    - **Children**
    - **Co-parents** (other parents of its children)
- **Note**: The "co-parents" part is due to the explaining away phenomenon.

# Undirected Graphical Models (Markov Random Fields - MRFs)
- Given by an undirected graph.

## Conditional Independence
- **Rule**: Much simpler than in Bayesian Networks.
- **Algorithm**: Two sets of nodes `A` and `B` are conditionally independent given a set `C` if **all paths** from `A` to `B` pass through at least one node in `C`.
- **Intuition**: If `C` "separates" `A` from `B` in the graph, then `A ⊥ B | C`.

## Factorization
- More complex than in Bayesian Networks. Relies on the concept of cliques.
- **Clique**: A subset of nodes where every two distinct nodes are connected by an edge.
- **Maximal Clique**: A clique that cannot be extended by adding any other node from the graph.
- **Problem**: Define the joint probability distribution.
- **Formula**: The joint distribution is a product of potential functions `ψ_C` over the maximal cliques `C` of the graph.
    - `p(x) = (1/Z) * ∏_C ψ_C(x_C)`
    - **Potential Function `ψ_C(x_C)`**: A non-negative function measuring the "compatibility" or "affinity" of the variables in a clique.
    - **Partition Function `Z`**: A normalization constant to ensure the distribution sums/integrates to 1.
        - `Z = Σ_x ∏_C ψ_C(x_C)`
- **Boltzmann Distribution**: A common way to define potential functions using an energy function `E`.
    - `ψ_C(x_C) = exp(-E(x_C))`
- **Notes**:
    - The presence of the partition function `Z` is a major limitation, as it can be computationally expensive to calculate (requires summing over all possible states of `x`).
    - Unlike BNs, MRFs are not automatically normalized.

## Converting Directed to Undirected Graphs
- **Process**: **Moralization**
- **Algorithm**:
    1.  **Marry the parents**: For each node, add an undirected edge between all pairs of its parents.
    2.  **Drop the arrows**: Convert all original directed edges to undirected edges.
    3.  The resulting graph is the **moral graph**.
- **Note**: This process can lose conditional independence information that the original directed graph contained (e.g., the independence in a head-to-head structure is lost).

# Inference in Graphical Models
- **Goal**: Evaluate the probability distribution over some variables, given the values of others (observations). E.g., compute `p(Query | Evidence)`.

## Exact Inference on a Chain
- **Problem**: Compute the marginal probability `p(x_n)` for a node in a chain.
- **Algorithm**: **Sum-Product (or Message Passing)**. Avoids naively summing over all variables.
    - **Intuition**: Split the computation into two parts, passing "messages" from the ends of the chain towards the query node `x_n`.
    - **Forward message `μ_α(x_n)`**: `μ_α(x_n) = Σ_{x_{n-1}} ψ_{n-1,n}(x_{n-1}, x_n) * μ_α(x_{n-1})`
    - **Backward message `μ_β(x_n)`**: `μ_β(x_n) = Σ_{x_{n+1}} ψ_{n,n+1}(x_n, x_{n+1}) * μ_β(x_{n+1})`
- **Final Marginal**: The marginal `p(x_n)` is the product of the incoming messages, normalized.
    - `p(x_n) = (1/Z) * μ_α(x_n) * μ_β(x_n)`
- **Normalization Constant `Z`**: Can be computed at any node `m` by summing the product of messages.
    - `Z = Σ_{x_m} μ_α(x_m) * μ_β(x_m)`

## Exact Inference on a Tree
- The message passing algorithm can be generalized to tree-structured graphs.
- **Algorithm**:
    1.  Designate the query node as the root of the tree.
    2.  Messages are passed from the leaves up to the root.
    3.  Messages are passed from the root back down to the leaves.
    4.  Once all messages are computed, the marginal at any node can be calculated as the product of all incoming messages.
- **Note**: The computational cost is linear in the number of nodes, making it very efficient for trees.
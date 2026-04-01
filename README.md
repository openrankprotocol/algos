# Graph-Based Trust and Ranking Algorithms

A Rust implementation of various graph-based algorithms for computing trust, reputation, and ranking scores in networks. This repository contains implementations of classic algorithms like PageRank and HITS, as well as trust-specific algorithms like EigenTrust and Transitive Trust.

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
  - [PageRank](#pagerank)
  - [HITS (Hubs and Authorities)](#hits-hubs-and-authorities)
  - [EigenTrust](#eigentrust)
  - [Transitive Trust](#transitive-trust)
  - [Graph Neural Network (GNN)](#graph-neural-network-gnn)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

## Overview

This project implements several algorithms that operate on graph structures to compute various forms of scores, rankings, or trust values for nodes. These algorithms are particularly useful in:

- **Reputation systems**: Determining the trustworthiness of peers in a network
- **Web page ranking**: Identifying authoritative pages in a hyperlink graph
- **Decentralized trust**: Computing trust scores without a central authority
- **Sybil resistance**: Defending against fake identity attacks in peer-to-peer networks

## Algorithms

### PageRank

**File:** `src/page_rank.rs`

PageRank is Google's original algorithm for ranking web pages based on the link structure of the web. It models a "random surfer" who randomly clicks on links, with a probability of jumping to a random page.

#### How It Works

1. **Initialization**: Each node starts with a seed score (can be uniform or based on pre-trust values)

2. **Normalization**: The adjacency matrix is row-normalized so that outgoing weights from each node sum to 1

3. **Iterative Update**: For each iteration:
   - Compute the weighted sum of incoming scores for each node
   - Apply dampening: `new_score = d × pre_trust + (1-d) × incoming_scores`
   - Where `d` is the dampening factor (0.2 in this implementation)

4. **Convergence**: After sufficient iterations (50 by default), scores stabilize

#### Key Parameters

- `DAMPENING_AMOUNT` (0.2): Probability of teleporting to a pre-trusted node
- `NUM_ITER` (50): Number of iterations to run
- `pre_trust`: Initial trust distribution for teleportation

#### Formula

```
PR(i) = d × pre_trust(i) + (1-d) × Σ(PR(j) × weight(j→i))
```

---

### HITS (Hubs and Authorities)

**File:** `src/hubs_and_auth.rs`

HITS (Hyperlink-Induced Topic Search) computes two scores for each node: a **hub score** (how well a node points to good authorities) and an **authority score** (how much a node is pointed to by good hubs).

#### How It Works

1. **Initialization**: Each node starts with initial hub and authority scores

2. **Authority Update**: A node's authority score is the sum of hub scores of nodes pointing to it
   ```
   auth(i) = Σ hub(j) for all j that link to i
   ```

3. **Hub Update**: A node's hub score is the sum of authority scores of nodes it points to
   ```
   hub(i) = Σ auth(j) for all j that i links to
   ```

4. **Normalization**: After each iteration, scores are normalized using L2 norm (sqrt of sum of squares)

5. **Iteration**: Steps 2-4 repeat until convergence (50 iterations)

#### Key Insight

- Good hubs point to good authorities
- Good authorities are pointed to by good hubs
- This mutual reinforcement leads to stable rankings

---

### EigenTrust

**File:** `src/eigen_trust.rs`

EigenTrust is a reputation management algorithm designed for peer-to-peer networks. It extends PageRank concepts to handle both **trust** and **distrust**, making it robust against malicious actors.

#### How It Works

1. **Local Trust Matrix**: Each peer maintains trust/distrust opinions about other peers
   - Trust and distrust are kept separate (a peer cannot be both trusted and distrusted by the same peer)

2. **Positive Trust Propagation**:
   - Similar to PageRank, trust scores propagate through the network
   - Pre-trusted peers (seed nodes) provide a baseline trust
   - Formula: `score = α × pre_trust + (1-α) × global_trust`
   - Where `α` is the pre-trust weight (0.5)

3. **Negative Trust (Distrust) Computation**:
   - Distrust is propagated in a single pass using the converged trust scores
   - Distrust from highly-trusted peers carries more weight

4. **Final Score Adjustment**:
   - Final score = positive score - negative score
   - This allows identifying truly malicious actors

#### Snap Score Calculation

For evaluating specific items (snaps), the algorithm computes:
- **Score**: Ratio of weighted positive votes to total weighted votes
- **Confidence**: Total weight of votes (from trusted peers only)
- **State**: Based on thresholds - `Unverified`, `Reported`, `Contested`, or `Endorsed`

#### Attack Resistance

The implementation includes test cases for:
- **Sybil attacks**: Multiple fake identities trying to boost reputation
- **Sleeping agent attacks**: Malicious peers that initially behave well

---

### Transitive Trust

**File:** `src/transitive_trust.rs`

Transitive Trust computes trust scores based on the principle that trust propagates through chains: if A trusts B, and B trusts C, then A has some derived trust in C.

#### How It Works

1. **Graph Structure**: The network is modeled as a directed graph with:
   - Positive edges (trust relationships with weights)
   - Negative edges (distrust relationships with weights)

2. **Priority Queue Traversal**:
   - Starting from a source node (with trust score 1.0)
   - Nodes are processed in order of their current trust score (highest first)
   - This ensures trust flows from more trusted to less trusted nodes

3. **Score Propagation**:
   - For each unvisited neighbor with lower score:
   - Positive score: `p_score += (node_score - p_score) × positive_weight`
   - Negative score: `n_score += (node_score - n_score) × negative_weight`

4. **Net Score**: Final trust = positive score - negative score

#### Key Properties

- Trust attenuates as it propagates (further nodes get lower scores)
- Negative edges can reduce or negate positive trust
- Algorithm is greedy - processes highest-trust nodes first
- Prevents circular trust inflation

---

### Graph Neural Network (GNN)

**File:** `src/gnn.rs`

This implements a simple Graph Neural Network that learns edge weights and node biases through backpropagation to match desired trust labels.

#### How It Works

1. **Architecture**:
   - **Weights**: Learnable edge weights between connected nodes
   - **Biases**: Learnable self-loop weights for each node
   - **Activation**: `tanh` function for non-linearity

2. **Message Passing** (Forward Pass):
   ```
   For each iteration:
     aggregation[i] = Σ weight[i][j] × state[j]
     new_state[i] = tanh(bias[i] × state[i] + aggregation[i])
   ```

3. **Training**:
   - **Labels**: Some nodes have target scores (supervised learning)
   - **Loss**: Sum of absolute errors for labeled nodes
   - **Optimization**: Gradient descent with learning rate decay
   - Uses automatic differentiation (via `rustydiff` crate)

4. **Learning Process**:
   - Multiple training runs with random initialization
   - 100 epochs per run, 100 runs total
   - Learning rate starts at 0.01 and decays by 0.999 each epoch

#### Use Case

This approach is useful when:
- You have some known trust relationships (labels)
- You want to learn the underlying trust propagation weights
- The network structure is fixed but edge strengths are unknown

---

## Getting Started

### Prerequisites

- Rust (latest stable version)
- Cargo

### Running

```bash
# Clone the repository
git clone <repository-url>
cd algos

# Run the default algorithm (transitive trust)
cargo run

# To run a different algorithm, modify main.rs to call the desired run_job() function
```

### Modifying Which Algorithm Runs

Edit `src/main.rs` to call the desired algorithm:

```rust
fn main() {
    // Uncomment the algorithm you want to run:
    // page_rank::run_job();
    // hubs_and_auth::run_job();
    // eigen_trust::functional_case();
    // eigen_trust::sybil_case();
    // eigen_trust::sleeping_agent_case();
    transitive_trust::run_job();
    // gnn::run_job();
}
```

## Dependencies

- **rustydiff**: Automatic differentiation library for gradient computation in the GNN
- **rand**: Random number generation for GNN weight initialization
- **priority-queue**: Priority queue implementation for transitive trust algorithm

## Utility Functions

**File:** `src/utils.rs`

Common helper functions used across algorithms:

- `transpose`: Matrix transposition
- `normalise`: Row normalization (sum to 1)
- `normalise_sqrt`: L2 normalization (used in HITS)
- `vec_scalar_mul`: Scalar-vector multiplication
- `vec_add`: Vector addition

## License

This project is open source. Please check the repository for license details.
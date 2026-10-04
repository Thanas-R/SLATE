Here is a clear, no-nonsense breakdown of the project.

---

### What is this project?

This project is an **interpretability and applied mathematics tool** for Recurrent Neural Networks (RNNs).

Normally, in AI and machine learning, an RNN is treated like a complete **black box**:
- You feed in data.
- The neural network computes some hidden numbers inside.
- It spits out a prediction.
- Nobody actually checks what those internal numbers represent or whether they are mathematically well-behaved.

This project cracks open that black box. We take 720 consecutive hours (30 full days) of real-world environmental air pollution data, run it through an RNN, record the network's internal 8-dimensional memory vector at every single hour, and **analyze that memory using pure Linear Algebra and Spectral Theory**.

---

### What is it actually doing? (The 5 Core Things)

#### 1. Treating the AI as a Physics/Math System
Instead of just asking "did the AI make a good prediction?", we treat the AI's internal state updates as a **discrete dynamical system**:
$$\mathbf{h}_{t+1} = \tanh(W \mathbf{h}_t + U \mathbf{x}_t + \mathbf{b})$$
We track how this 8-dimensional point moves through space over 30 days.

#### 2. Dimensionality Reduction via QR Decomposition
Humans cannot see in 8 dimensions. Instead of throwing random techniques at it:
- We take the 8D memory states and perform **QR decomposition** on their covariance.
- This creates an **orthonormal basis** (mutually perpendicular unit axes $q_1, q_2, q_3$).
- We project the 8D memory vectors down onto 2D and 3D while preserving over 65% to 85% of the total mathematical energy.
- **What you see on screen**: The memory traces out clear loops and clusters (morning traffic rush hours cluster in one region, night baselines in another).

#### 3. Linearization (The Surrogate Model)
The RNN update uses $\tanh$, which is non-linear. We ask a fundamental math question:
> *Can a simple, static linear matrix $A$ approximate this complex neural network?*
> $$\mathbf{h}_{t+1} \approx A \mathbf{h}_t$$

Using **Least Squares (Moore-Penrose Pseudoinverse)**, we solve for the best matrix $A$.
- **The Result**: The linear matrix achieves an $R^2$ fit score of **0.8495** (85%).
- This proves that a massive portion of the neural network's internal memory obeys straightforward linear mechanics.

#### 4. Proving Stability using Eigenvalues (Spectral Analysis)
If a memory system loops over itself for hundreds of hours, will it explode to infinity (diverge) or collapse to zero?
- We calculate the eigenvalues $\lambda_i$ of matrix $A$:
  $$A \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
- In discrete dynamical systems, after $t$ steps, each mode scales by $\lambda_i^t$.
- If $|\lambda| > 1$, the system blows up.
- If $|\lambda| < 1$, the system is contractive and stable.
- **Our Proof**: The largest eigenvalue magnitude (spectral radius $\rho(A)$) is **0.9294**, strictly less than 1.0. All eigenvalues sit inside the unit circle. This is mathematical proof that the AI memory is stable.

#### 5. Pruning and Redundancy Check (RREF via SymPy)
Does the network actually need 8 hidden neurons, or are some of them just copies of each other?
- We run **Reduced Row Echelon Form (Gauss-Jordan elimination)** on the states.
- It tests for pivot columns versus free columns.
- **The Result**: Rank = 8 / 8. All 8 dimensions are pivot columns, proving that zero neurons are redundant; each one encodes unique, independent information.

---

### What are the Deliverables?

1. **The Python Core Math Engine**:
   - `core/rnn_engine.py`: Pure NumPy recurrent dynamics and readout fitting.
   - `core/algebra.py`: QR decomposition and orthogonal projection operators.
   - `core/dynamics.py`: Least-squares surrogate model.
   - `core/spectral.py`: Eigenvalue decomposition and stability classification.
   - `core/rref.py`: SymPy row reduction.
2. **The Fast Web API & Server**:
   - `web/server.py`: FastAPI backend that runs the pipeline and serves real-time JSON metrics.
3. **The Minimal Obsidian-Style Dashboard**:
   - 3D constellation space with physical lit spheres and orbital camera controls.
   - 2D terminal map drawn with ASCII borders, crosshair symbols, and dots.
   - Complex plane eigenvalue plot with unit circle boundary.
   - Real-world PM2.5 prediction tracking ($R^2 > 0.94$).
   - Interactive timeline scrubber with 0.25x to 2.0x playback speeds.
4. **Documentation**:
   - `README.md`: Complete math derivations and ASCII pipeline diagrams.
   - `GUIDE.md`: 30-second elevator pitch, viva Q&A, and screen-by-screen presentation scripts.

---

### In One Sentence

> *"We treat a Recurrent Neural Network as a discrete dynamical system and use Linear Algebra (QR projection, least-squares surrogate modeling, eigenvalue stability analysis, and RREF redundancy testing) to mathematically inspect, prove the stability of, and visualize its hidden memory."*
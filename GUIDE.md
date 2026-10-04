# SLATE: Spectral Latent Analysis & Trajectory Exploration -- Team Guide & Viva Manual

A complete pedagogical and presentation manual for teammates, students, and examiners.
No emojis. No em dashes.

---

## 1. HOW TO EXPLAIN THIS PROJECT IN 30 SECONDS

If someone asks what this project is:
> "SLATE is a mathematical research laboratory that studies what happens inside a Recurrent Neural Network. Normally, people treat RNNs as black boxes. In our project, we capture the internal memory vector at every single hour of air pollution data. We treat this sequence of vectors as a discrete dynamical system. We then use Linear Algebra: QR decomposition for orthogonal projection, least squares to estimate a linear transition matrix, and eigenvalue analysis to prove whether the network's internal memory is stable or unstable."

---

## 2. THE 6 MATHEMATICAL MODULES EXPLAINED FOR TEAMMATES

### Module 1: Data Matrix Representation
- What it is: Taking raw pollution sensors (CO, NO2, Temperature, Relative Humidity, PM10) across 720 hours and organizing them into a table with 720 rows and 5 columns.
- Mathematical object: X in R^(720 x 5).
- Why we do it: Machine learning models take vectors, not raw text. Stacking vectors into a matrix lets us use matrix multiplications.
- Code location: data/dataset_loader.py

### Module 2: The Recurrent Transition (The Dynamical Step)
- What it is: At each hour t, the network receives input x_t (new weather) and combines it with its previous memory h_t to produce the new memory h_(t+1):
  h_(t+1) = tanh(W @ h_t + U @ x_t + b)
- Why we use tanh: The hyperbolic tangent squashes all values into (-1, +1). This prevents numbers from growing uncontrollably during recursive multiplication.
- Code location: core/rnn_engine.py

### Module 3: Orthonormal Basis via QR Decomposition
- What it is: The hidden memory has 8 dimensions. The raw 8 dimensions are correlated with each other. QR decomposition breaks the memory covariance into two matrices:
  H_centered^T = Q @ R
  where Q contains perpendicular unit vectors: Q^T @ Q = I.
- Why we do it: Q gives us independent, mutually perpendicular coordinate axes that capture the directions of highest variation.
- Code location: core/algebra.py

### Module 4: Orthogonal Projection (2D and 3D)
- What it is: Human eyes cannot visualize 8 dimensions. We take the top 2 or 3 vectors from Q (called basis matrix B) and project the 8D state onto them:
  z_t = B^T @ (h_t - mu)
- Why we do it: This maps every 8D memory vector to a 2D coordinate (x, y) or 3D coordinate (x, y, z) while preserving over 65% of the total geometric energy.
- Code location: core/algebra.py

### Module 5: Least-Squares Linear Surrogate Dynamics
- What it is: The real neural network is non-linear because of tanh. We ask: can we find a single linear matrix A such that:
  h_(t+1) approx A @ h_t
  We solve this overdetermined system using the Moore-Penrose pseudoinverse:
  A = (H_prev^+ @ H_next)^T
- Why it matters: Our linear model achieves an R^2 score > 0.84. This proves that over 84% of the complex neural network's behavior can be modeled by simple linear matrix multiplication.
- Code location: core/dynamics.py

### Module 6: Spectral Stability and the Complex Unit Circle
- What it is: We compute the eigenvalues of transition matrix A:
  A @ v_i = lambda_i * v_i
  In a discrete dynamical system, iterating t steps multiplies the mode by (lambda_i)^t.
- The Stability Rule:
  * If |lambda| < 1: The mode shrinks exponentially to 0. Memory is contractive and stable.
  * If |lambda| = 1: The mode neither shrinks nor grows. It produces sustained periodic oscillation.
  * If |lambda| > 1: The mode multiplies infinitely. Memory diverges and blows up.
- Result in SLATE: Our dominant eigenvalue has magnitude 0.929 < 1.0. All eigenvalues are strictly inside the unit circle, proving the RNN memory is asymptotically stable.
- Code location: core/spectral.py

---

## 3. HOW TO DEMO AND EXPLAIN EACH SCREEN IN THE UI

When presenting the application live, start with:
```bash
python run.py
```
Dashboard opens at http://127.0.0.1:8000

UI Controls:
- [ i / info ]: Toggle explanation card in header
- [ x ]: Dismiss explanation card
- Speed options: 0.25x, 0.5x, 1.0x, 2.0x in timeline bar
- Space: Play/pause temporal trajectory
- Keys 1 to 6: Switch between the 6 mathematical views

### Screen 1: 3D State Space Constellation
- How to demonstrate: Click and drag to orbit, scroll to zoom, press Space to animate.
- What to say:
  "Here you see the 3D state space. Each tiny dot is one hour of internal memory in the neural network. The connecting line (#424242) shows how memory moves over 30 days. Dots form clusters corresponding to daily rush hour and photochemical cycles. The floating badge tracks the active state."

### Screen 2: 2D Memory Map (Orthonormal Phase Portrait)
- How to demonstrate: Press 2 on your keyboard or click [ 02: 2D MAP ].
- What to say:
  "This is the 2D phase portrait. The solid path is the actual neural network trajectory. The dashed path is our linear surrogate model A @ h_t. The close alignment proves that linear algebra captures over 84% of the recurrent memory dynamics."

### Screen 3: Stability Circle (Complex Eigenvalue Plane)
- How to demonstrate: Press 3 or click [ 03: STABILITY ].
- What to say:
  "This is our spectral stability analysis. The dashed circle is the unit circle (|lambda| = 1). Every eigenvalue of matrix A is plotted as a dot. All dots reside strictly inside the circle with spectral radius 0.929, mathematically proving the RNN memory is stable."

### Screen 4: Surrogate Model Error
- How to demonstrate: Press 4 or click [ 04: SURROGATE ].
- What to say:
  "This screen shows the hourly prediction error ||h_(t+1) - A @ h_t||. The low average residual confirms the linear surrogate model remains accurate over all 720 hours."

### Screen 5: RREF Latent Redundancy
- How to demonstrate: Press 5 or click [ 05: RREF ].
- What to say:
  "We computed the Reduced Row Echelon Form using SymPy. The rank is 8 out of 8, and all dimensions are pivot columns, proving zero duplicate neurons and full memory utilization."

### Screen 6: Real-World PM2.5 Prediction
- How to demonstrate: Press 6 or click [ 06: PREDICTION ].
- What to say:
  "We verify that this memory solves an engineering task: predicting PM2.5 air pollution. The R^2 score exceeds 0.94, confirming the geometric state space encodes genuine environmental physics."

---

## 4. VIVA QUESTIONS AND READY-TO-SPEAK ANSWERS

Question 1: What is the problem statement of your project?
Answer:
"Standard RNNs treat hidden states as opaque black boxes, making it impossible to know whether internal representations are stable, redundant, or explainable. Our project investigates whether the latent state trajectory of an RNN can be modeled as a discrete dynamical system and analyzed using classical Linear Algebra, specifically QR orthogonal projection, least-squares linear surrogate estimation, and spectral stability analysis."

Question 2: Why do you call the RNN a discrete dynamical system?
Answer:
"Because the network maintains an internal state vector h_t that updates iteratively at discrete integer time steps according to the rule h_(t+1) = f(h_t, x_t). The sequence h_0 -> h_1 -> h_2 -> ... forms a discrete trajectory through state space."

Question 3: Why did you use QR decomposition instead of raw coordinates?
Answer:
"The raw 8 dimensions of the neural network are correlated with each other. QR decomposition constructs an orthonormal basis B where all vectors are mutually perpendicular unit vectors (B^T @ B = I). This eliminates correlation and guarantees that our 2D and 3D projections retain maximum geometric variance without distortion."

Question 4: What is the projector operator and what are its properties?
Answer:
"The projector operator is P = B @ B^T. It is symmetric (P^T = P) and idempotent (P^2 = P). Idempotency means projecting a second time changes nothing: once a vector is in the subspace, it stays in the subspace."

Question 5: Why did you use Least Squares to find matrix A?
Answer:
"Because we have 720 time steps and only 8 dimensions. The system H_next approx H_prev @ A^T is overdetermined with no exact solution. Least squares uses the Moore-Penrose pseudoinverse to find the unique matrix A that minimizes the sum of squared Euclidean errors."

Question 6: What does the spectral radius rho(A) tell you?
Answer:
"The spectral radius is the maximum absolute eigenvalue of matrix A. In a discrete system h_t = A^t @ h_0, if rho(A) < 1, all modes decay exponentially and the system is asymptotically stable. If rho(A) > 1, the system would explode exponentially."

Question 7: How does RREF fit into this machine learning project?
Answer:
"RREF separates pivot columns from free columns. In neural network analysis, pivot columns represent linearly independent dimensions, while free columns represent redundant neurons that duplicate existing information. Our RREF analysis showed a full rank of 8, proving all 8 dimensions are utilized."

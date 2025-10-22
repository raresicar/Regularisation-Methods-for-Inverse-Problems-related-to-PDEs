# Regularisation Methods for Inverse Problems related to PDEs
 
🧮 Regularisation Methods for Inverse Problems Related to PDEs

This repository contains the source code developed for my Bachelor’s Thesis at Babeș-Bolyai University, Faculty of Mathematics and Computer Science (2025).
The project explores numerical methods for solving ill-posed inverse problems associated with elliptic partial differential equations (PDEs) — with a focus on the Cauchy problem and the Unique Continuation problem.

📖 Overview

Inverse problems arise in many applications such as medical imaging, non-destructive testing, and geophysics, where one seeks to reconstruct unknown quantities from partial or indirect measurements.
Because these problems are typically ill-posed in the sense of Hadamard, they require regularisation to ensure stable, meaningful solutions.

This project implements and analyses several regularisation techniques—particularly Tikhonov-type regularisation—formulated as PDE-constrained optimisation problems and solved using finite element methods (FEM).

🧩 Main Chapters

Abstract Framework:

Linear inverse problems in Banach/Hilbert spaces.

Generalised solution and inverse.

Compact operators and ill-posedness.

Regularisation methods (spectral and variational).

PDE-based Inverse Problems:

Weak formulations of elliptic PDEs and well-posedness.

From weak form to inverse problems. 

Regularised formulations for:

The Cauchy problem for elliptic operators (Laplace, Helmholtz).

The Unique Continuation problem.

Numerical Implementation:

Discretisation using finite elements.

Numerical experiments demonstrating stability and convergence.

⚙️ Technologies & Libraries

Python for numerical implementation

FEniCS / dolfin for finite element discretisation

NumPy, Matplotlib for numerical computation and visualization

📊 Results

The implemented methods illustrate how Tikhonov-type regularisation stabilises the reconstruction of solutions from incomplete boundary or interior data.
Numerical examples include simulations for Laplace’s and Helmholtz’s equations, comparing different regularisation parameters and mesh refinements.

🧠 Academic Context

This repository accompanies my Bachelor’s Thesis:

“Regularisation Methods for Inverse Problems Related to Partial Differential Equations”
Supervisor: Lect. Dr. Mihai Nechita
Author: Răhăian Rareș, Babeș-Bolyai University, 2025

The work received the 2nd prize at the “Sesiunea de Comunicări Științifice ale Studenților – Matematică, 2025”.

# Image Filtering and Denoising

**Team:** Ettore Cirillo, Angelo Notarnicola, Stefano Pedretti

## 📌 Overview
This challenge was developed for the Numerical Linear Algebra course (A.A. 2024/2025). The main goal is to apply image filters and find the approximate solution of linear system to process a greyscale image.

## 🛠️ Technologies
* **Language:** C++.
* **Libraries:** Eigen library and LIS library.
* **Core Concepts:** Convolution filters , sparse matrix operations, iterative solvers, and preconditioners.

## 🚀 Key Features
* **Image Convolution:** Implemented blurring/smoothing, sharpening, and edge detection by translating convolution kernels into matrix-vector multiplications.
* **Image Denoising:** Applied anisotropic diffusion to effectively process and reduce image noise.
* **Iterative Solvers:** Computed approximate solutions for large-scale linear systems prescribing strict tolerances (e.g., $10^{-9}$ and $10^{-10}$) using the LIS and Eigen libraries.
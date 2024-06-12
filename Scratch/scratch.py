import scipy as sp
import numpy as np
import cupy
from cupyx.scipy.sparse.linalg import spilu, gmres, LinearOperator
from cupyx.scipy.sparse import csc_matrix
import json
import inspect
import time
import matplotlib.pyplot as plt


def callback_cpu(xk):
    # callback function for cg method
    frame = inspect.currentframe().f_back
    print("Iteration :", frame.f_locals["inner_iter"])
    print()


def callback_gpu(xk):
    # callback function for cg method
    frame = inspect.currentframe().f_back
    print("Iteration :", frame.f_locals["iters"])
    print()


# ==============================================================================
# Load K and b
# ==============================================================================
K = open(f"K.json")
b = open(f"b.json")

K = np.array(json.load(K))
b = np.array(json.load(b))

# Set solver settings
toler = 1e-10
maxiter = 100

# ==============================================================================
# Solve system Kx=b on CPU
# ==============================================================================
# Solve system with numpy
t0_np = time.perf_counter()
x_np = np.linalg.solve(K, b)
tf_np = time.perf_counter()

# Convert to sparse matrix format
K_csc = sp.sparse.csc_array(K)

# Create a linear operator for the preconditinoer
ilu = sp.sparse.linalg.spilu(K_csc)
Mx = lambda x: ilu.solve(x)
M = sp.sparse.linalg.LinearOperator((len(b), len(b)), Mx)

# Sove using scipy sparse algorithms
# print("GMRES CPU Callback:")
t0_sp = time.perf_counter()
x_sp, info = sp.sparse.linalg.gmres(
    A=K_csc,
    b=b,
    M=M,
    # callback=callback_cpu,
    rtol=toler,
    maxiter=maxiter,
    callback_type="legacy",
)
tf_sp = time.perf_counter()

# ==============================================================================
# Solve system Kx=b on GPU
# ==============================================================================
# Transfer sytem to GPU
K_gpu = cupy.asarray(K)
K_gpu_csc = csc_matrix(K_gpu)
b_gpu = cupy.asarray(b)

# Create a linear operator for the preconditioner
ilu_gpu = spilu(K_gpu_csc)
Mx_gpu = lambda x: ilu_gpu.solve(x)
M_gpu = LinearOperator((len(b_gpu), len(b_gpu)), Mx_gpu)

# Solve using cupyx sparse algorithms
# print("GMRES GPU Callback:")
t0_cpx = time.perf_counter()
x_cpx, info_cpx = gmres(
    A=K_gpu,
    b=b_gpu,
    M=M_gpu,
    # callback=callback_gpu,
    tol=toler,
    maxiter=maxiter,
)
tf_cpx = time.perf_counter()
# Send back solution from gpu to cpu
x_cpx_sent_back = x_cpx.get()

# ==============================================================================
# Print out info
# ==============================================================================
print("CPU allclose : ", np.allclose(x_np, x_sp, rtol=toler))
print("GPU allclose : ", np.allclose(x_np, x_cpx_sent_back, rtol=toler))
print(f"Numpy solve = {tf_np-t0_np:.4f} s")
print(f"Scipy solve = {tf_sp-t0_sp:.4f} s")
print(f"Cupyx solve = {tf_cpx-t0_cpx:.4f} s")

# ==============================================================================
# Plot results
# ==============================================================================
fig, ax = plt.subplots()
rel_err_sp = np.divide(x_sp - x_np, x_np)
rel_err_cpyx = np.divide(x_cpx_sent_back - x_np, x_np)
ax.plot(np.linspace(0, len(b), len(b)), rel_err_cpyx, label="cupyx: gmres")
ax.plot(np.linspace(0, len(b), len(b)), rel_err_sp, label="scipy: gmres")
ax.set_yscale("log")
ax.set_title(f"# of Triangular Elements = {len(b)}")
ax.legend(loc="upper right")
ax.grid()
plt.show()

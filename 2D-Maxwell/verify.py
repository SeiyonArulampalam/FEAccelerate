import json
import numpy as np
import matplotlib.pyplot as plt

x_gpu = open("gpu.json")
x_cpu = open("cpu.json")

x_gpu_set = json.load(x_gpu)
x_cpu_set = json.load(x_cpu)

print("\nCheck CPU and GPU solutions are close:\n")
for i in range(len(x_cpu_set)):
    x_cpu_i = np.array(x_cpu_set[i])
    x_gpu_i = np.array(x_gpu_set[i])
    print(f"Iteration {i}:", np.allclose(x_gpu_i, x_cpu_i))

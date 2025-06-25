import sys

sys.path.append("src")
sys.path.append("tesseracts")

import os

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api
import sheaths.tanh_sheath.tesseract_api as tanh_sheath_tesseract_api
import sheaths.vlasov.tesseract_api as vlasov_sheath_tesseract_api
import jpu
import equinox as eqx
import mlflow
from spscml.whole_device_model.local_wrapper import apply
import optax
import numpy as np
jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()
from run_wdm_without_tesseract import change_wdm
from spscml.fusion import fusion_power, bremsstrahlung_power

def extract_value(x):
    # Check if it's a NumPy or JAX array
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        # If it's a scalar array like np.array([1.0]), return the scalar
            return x.item()
    else:
        return x  # Already a scalar or something else

def objective(Vp):
    results, params = change_wdm(Vp)
    n = results["n"]
    L = params["Lz"] # setting this as the electrode gap (not quite true)
    a = results["a"]
    T = results["T"]
    dt = results["ts"][1] - results["ts"][0]

    power_out = jnp.sum(fusion_power(n, L, a, T))
    power_lost = jnp.sum(bremsstrahlung_power(n, L, a, T))
    print(power_out*dt)
    print(power_lost*dt)
    net = power_out - power_lost
    return net * dt

Vp_input = 10000
print(jax.grad(objective)(Vp_input))

def grad_obj(Vp):
    Vp = jnp.array(extract_value(Vp))
    return jax.grad(objective)(Vp)

def scipy_vg(_Vp):
    # """Value and gradient function for scipy optimization."""
    # __Vp = jnp.array(_Vp)
    __Vp = jnp.array(extract_value(_Vp))  

    fval, gradVal = fgrad_fn(__Vp)
    fval = np.array(fval)
    gradVal = np.array(gradVal)
    return fval, gradVal
    
print(f"Vp: {Vp_input}, Objective: {objective(Vp_input)}")
print(f"grad: {grad_obj(Vp_input)}")

f = lambda Vp: objective(Vp)
fgrad_fn = jax.value_and_grad(f)
    
fval, gradVal = fgrad_fn(Vp_input)
print(f"Value and Gradient of objective function at Vp={Vp_input} f: {fval:.5f} W df/dx: {gradVal:.5f} W/V")

res = opt.minimize(scipy_vg, Vp_input, method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': 100}, bounds=[(400, 10e3)])

# # Use scipy minimize with L-BFGS-B
# res = minimize(fun=objective, x0=Vp_input, jac=grad_obj, method='L-BFGS-B')

# Print result
print("Optimal x:", res.x)
print("Final loss:", res.fun)
print("Converged:", res.success)

# solver = optax.lbfgs()
# params = jnp.array(Vp_input)
# opt_state = solver.init(params)
# for i in range(2):
#     print(i)
#     grad = grad_obj(params)
#     updates, opt_state = solver.update(grad, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     print('Objective function: {:.2E}'.format(objective(params)))








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
from spscml.fusion import fusion_power, bremsstrahlung_power
import optax
import scipy.optimize as opt 
# from scipy.optimize import minimize
import numpy as np

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()

# Parse command line arguments
def parse_args():
    args = sys.argv[1:]  # Skip the script name
    

    # Use the initial plasma from Fig 7 of Shumlak et al. (2012) as an example
    n0 = 6e22 * ureg.m**-3

    # Default values
    Vc0 = 40*1e3  # 40kV
    T = 20.0      # 20 eV
    Vp = 2000     # V
    tesseract = "tanh_sheath"  # Default tesseract image

    R = 1.5e-3
    L = 2.0e-7
    C = 222*1e-6
    
    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == '--Vc0' and i + 1 < len(args):
            Vc0 = float(args[i + 1])
            i += 2
        elif args[i] == '--T' and i + 1 < len(args):
            T = float(args[i + 1])
            i += 2
        elif args[i] == '--Vp' and i + 1 < len(args):
            Vp = float(args[i + 1])
            i += 2
        elif args[i] == '--tesseract' and i + 1 < len(args):
            tesseract = args[i + 1]
            i += 2
        elif args[i] == '--help' or args[i] == '-h':
            print("Usage: python run_wdm.py [--Vc0 VALUE] [--T VALUE] [--Vp VALUE] [--image NAME]")
            print(f"  --Vc0 VALUE   Total capacitor voltage in volts (default: {Vc0})")
            print(f"  --T VALUE     Initial temperature in eV (default: {T})")
            print(f"  --Vp VALUE    Initial plasma voltage in volts (default: {Vp})")
            print("  --tesseract NAME  Tesseract name (default: tanh_sheath)")
            print("  --help, -h    Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}")
            print("Use --help for usage information")
            sys.exit(1)
        
    return Vc0, T, Vp, tesseract, n0, R, L, C

Vc0, T_input, Vp_input, tesseract_name, n0, R, L, C = parse_args()

print(f"Running WDM simulation with:")
print(f"  Vc0 = {Vc0} V")
print(f"  T = {T_input} eV") 
print(f"  Vp = {Vp_input} V")
print(f"  n0 = {n0} m^-3")
print(f"  Tesseract = {tesseract_name}")
print(f"  R = {R} ohms") 
print(f"  L = {L} henries") 
print(f"  C = {C} farads") 

Lz = 0.5
Lp = -.4e-7
Lp_prime = Lp / Lz
L_tot = L - Lp

Z = 1.0

if tesseract_name == "tanh_sheath":
    tesseract_api = tanh_sheath_tesseract_api
elif tesseract_name == "vlasov_sheath":
    tesseract_api = vlasov_sheath_tesseract_api

sheath_tx = Tesseract.from_tesseract_api(tesseract_api)

def call_tess(Vc0, T_input, Vp_input, n0, tesseract_api, R, L, C) -> dict:
    Vp0 = Vp_input * ureg.volts

    T0 = T_input * ureg.eV
    if ~isinstance(Vp0, jnp.ndarray):
        Vparray = jnp.array(Vp0.magnitude)
    else:
        Vparray = Vp0.magnitude
        
    j = apply_tesseract(sheath_tx,dict(
        n=jnp.array(n0.magnitude), T=jnp.array(T0.magnitude), Vp=Vparray, Lz=jnp.array(0.5)
        ))["j"] * (ureg.A / ureg.m**2)
    N = ((8*jnp.pi * (1 + Z) * T0 * n0**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
    jax.debug.print("N = {}", N)

    Ip = (j * N / n0).to(ureg.A)
    jax.debug.print("Ip = {}", Ip)

    a0 = ((N / n0 / jnp.pi)**0.5).to(ureg.m)

    result = apply(dict(
        Vc0=Vc0,
        Ip0=Ip.magnitude,
        a0=a0.magnitude,
        N=N.magnitude,
        Lp_prime=Lp_prime,
        Lz=Lz,
        R=R, L=L, C=C,
        dt=5e-8,
        t_end=1e-5,
        mlflow_parent_run_id=None
    ), sheath_tx)

    result["a"] = jnp.sqrt(N.magnitude / result["n"] / jnp.pi)

    return result

def extract_value(x):
    # Check if it's a NumPy or JAX array
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        # If it's a scalar array like np.array([1.0]), return the scalar
            return x.item()
    else:
        return x  # Already a scalar or something else

def objective(Vp, R, L, C):
    # Vp = jnp.array(extract_value(Vp))
    results = call_tess(Vc0, T_input, Vp, n0, tesseract_api, R, L, C) 
    dt = results["ts"][1] - results["ts"][0]
    diff = fusion_power(results["n"], Lz, results["a"], results["T"]) - bremsstrahlung_power(results["n"], Lz, results["a"], results["T"])
    return -diff.sum() * dt

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


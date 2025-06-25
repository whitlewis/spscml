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

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()
from run_wdm_without_tesseract import change_wdm
from spscml.fusion import fusion_power, bremsstrahlung_power


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

print(objective(10000))
# Vp =              # Plasma voltage
# Vp = Vp.astype(jnp.float64)
# obj_fn_value = objective(Vp)
# print(f"Objective function value for Vp={Vp} V: {obj_fn_value:.2f} W")

# # try out grad of objective function with respect to Vp

# f = lambda Vp: objective(Vp)
# grad_f = jax.grad(f)(jnp.array(Vp))
# print(f"Gradient of objective function at Vp={Vp} V: {grad_f:.2f} W/V")












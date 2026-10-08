#!/usr/bin/env python3
"""
verify_r51_mbar_factor.py

Verify whether the Table-1 / Table-A.1 mass discrepancy (R5.1) is explained
*solely* by the cold-gas correction (Boselli 2014), or whether a log-mean vs
arithmetic-mean difference is also required.

Boselli+2014 cold-gas fraction model (same as scripts_replicable/forward_lensing_delta_sigma.py):
    f_cold(M_star) = 10^(-0.69 * log10(M_star) + 6.63)
    M_bar = M_star * (1 + f_cold)

Table-1 M_star values (pre-fix, before gas correction):
    Late:  9.5e9  Msun
    Early: 2.0e10 Msun
    Blue:  7.8e9  Msun   (from ESTAT.md session 2026-10-07 pendent 3)
    Red:   1.2e10 Msun

Table-A.1 M_bar targets (already include gas correction):
    Late:  1.5e10 Msun
    Early: 2.7e10 Msun
    Blue:  1.3e10 Msun
    Red:   1.8e10 Msun
"""
import numpy as np

def f_cold(mstar_msun):
    """Boselli+2014 cold-gas fraction."""
    return 10.0**(-0.69 * np.log10(mstar_msun) + 6.63)

def mbar(mstar_msun):
    return mstar_msun * (1.0 + f_cold(mstar_msun))

data = [
    ("Late",  9.5e9,  1.5e10),
    ("Early", 2.0e10, 2.7e10),
    ("Blue",  7.8e9,  1.3e10),
    ("Red",   1.2e10, 1.8e10),
]

print("=" * 72)
print("R5.1 verification: does gas correction alone explain Table1/A1 discrepancy?")
print("=" * 72)
print(f"{'Type':>6}  {'M_star [1e10]':>14}  {'M_bar_gas [1e10]':>17}  "
      f"{'M_bar_target [1e10]':>20}  {'ratio_gas/target':>17}  {'fully_expl':>10}")
print("-" * 90)

all_ok = True
for name, mstar, target in data:
    mb_gas  = mbar(mstar)
    ratio   = mb_gas / target
    ok      = abs(ratio - 1.0) < 0.06   # within 6%
    flag    = "YES" if ok else "NO -- additional factor needed"
    if not ok:
        all_ok = False
    print(f"{name:>6}  {mstar/1e10:>14.3f}  {mb_gas/1e10:>17.3f}  "
          f"{target/1e10:>20.3f}  {ratio:>17.4f}  {flag:>10}")

print("-" * 90)
if all_ok:
    print("\nCONCLUSION: gas correction alone (Boselli+2014) fully explains the")
    print("Table-1 / Table-A.1 discrepancy for all four morphological subsets.")
    print("The 'log-mean vs arithmetic-mean' explanation in R5.1 is NOT required")
    print("and should be removed from the response.")
else:
    print("\nCONCLUSION: gas correction does NOT fully explain the discrepancy.")
    print("The additional factor (log-mean / arithmetic-mean) is needed.")

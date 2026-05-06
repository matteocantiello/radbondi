"""Generate the Kn_eff vs ell/ell_crit figure with fate regions."""

import numpy as np
import matplotlib.pyplot as plt
import radbondi as rb
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': False,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.frameon': False,
})

# Load
M_BH = 1e-16 * rb.M_sun
GM = G * M_BH
rS = 2 * GM / c_light**2
ell_crit = 4 * GM / c_light
sol = rb.load("examples/paper_sweep_output/mbh_logM-16.00.npz")
gamma, mu = sol.ambient_gamma, sol.ambient_mu
r, rho_p, T_p, v_p = sol.r, sol.rho, sol.T, np.abs(sol.v)
cs = np.sqrt(gamma * kB * T_p / (mu * m_p))

E4 = (4.803e-10)**4
n_e = (sol.ambient_X + 0.5 * sol.ambient_Y) * rho_p / m_p
lam = (kB * T_p)**2 / (np.pi * n_e * E4 * 5.0)
Kn = lam / r

i_coll = np.where(np.diff(np.sign(Kn - 1.0)))[0][-1]
r_coll = r[i_coll]
sigma_perp = cs[i_coll] / np.sqrt(gamma)
v_coll = v_p[i_coll]
E_coll = 0.5 * v_coll**2 - GM / r_coll
ell_max = np.sqrt(GM**2 / (2 * abs(E_coll)))

amb = rb.AmbientMedium(T=sol.ambient_T, rho=sol.ambient_rho,
                        mu=mu, gamma=gamma, X=sol.ambient_X, Y=sol.ambient_Y)
cool = Cooling.default()
eps_ff = np.array([float(cool.total_emissivity(rho_p[i], T_p[i], amb)[0])
                    for i in range(len(r))])


def Kn_at(rt):
    if rt < r[0] or rt > r[-1]:
        return 1e10
    return 10**float(np.interp(np.log10(rt), np.log10(r),
                                np.log10(np.maximum(Kn, 1e-30))))


def dEE(ell):
    if ell <= 0 or ell >= ell_max:
        return 0.0
    ac, bc, cc = E_coll, GM, -(ell**2)/2
    disc = bc**2 - 4*ac*cc
    if disc < 0:
        return 0.0
    r1 = (-bc + np.sqrt(disc))/(2*ac)
    r2 = (-bc - np.sqrt(disc))/(2*ac)
    cands = [x for x in [r1, r2] if x > 0]
    if not cands:
        return 0.0
    rp = min(cands)
    dr = r_coll - rp
    if dr < 1e-15:
        return 0.0
    ri, ro = rp + 0.05*dr, r_coll - 0.05*dr
    m = (r >= ri) & (r <= ro)
    if np.sum(m) < 5:
        return 0.0
    ro2 = r[m]
    Eo = ell**2/(2*r_coll**2) - GM/r_coll
    vr2 = 2*(Eo + GM/ro2) - ell**2/ro2**2
    vr2 = np.maximum(vr2, 0.0)
    vr = np.sqrt(vr2)
    vt = np.sqrt(vr2 + ell**2/ro2**2)
    vr = np.maximum(vr, 0.01*vt)
    KE = 0.5*np.mean(vt**2)
    cr = eps_ff[m]/np.maximum(rho_p[m], 1e-30)
    return 2*1.1*np.trapz(cr/vr, ro2)/max(KE, 1e-30)


# Compute
ell_ratios = np.linspace(0.5, 30, 400)
Kn_eff_arr = np.full_like(ell_ratios, np.nan)
r_circ_arr = np.full_like(ell_ratios, np.nan)

for i, lr in enumerate(ell_ratios):
    ell = lr * ell_crit
    if ell < ell_crit:
        continue
    if ell > ell_max:
        continue
    rc = ell**2 / GM
    r_circ_arr[i] = rc / rS
    d = dEE(ell)
    xi = np.sqrt(2/max(d, 1e-30)) if d > 0 else 1e10
    xi = min(xi, 1e6)
    Kn0 = Kn_at(rc)
    Kn_eff_arr[i] = Kn0 / max(xi, 1)

# Plot
xlo, xhi = 0.0, 27.0
ylo, yhi = 0.08, 250.0

fig, ax = plt.subplots(figsize=(7.0, 4.6))

# Horizontal fate bands (Kn_eff regimes)
ax.axhspan(ylo, 1.0, color='#009E73', alpha=0.13, lw=0)
ax.axhspan(1.0, 3.0, color='#E69F00', alpha=0.13, lw=0)
ax.axhspan(3.0, yhi, color='#D55E00', alpha=0.10, lw=0)

# Vertical exclusion bands (no recycling: direct capture or reflected)
ax.axvspan(xlo, 1.0, color='0.55', alpha=0.30, lw=0, hatch='///')
ax.axvspan(ell_max/ell_crit, xhi, color='0.55', alpha=0.30, lw=0, hatch='///')

# Threshold lines
ax.axhline(1.0, color='0.35', ls=(0, (4, 2)), lw=0.7)
ax.axhline(3.0, color='0.55', ls=(0, (1, 2)), lw=0.6)

# Kn_eff curve
valid = ~np.isnan(Kn_eff_arr)
ax.semilogy(ell_ratios[valid], Kn_eff_arr[valid],
            color='k', lw=1.8, zorder=5,
            label=r'${\rm Kn}_{\rm eff}(\ell)={\rm Kn}_0/\xi$')

# Mark the Kn_eff = 1 crossing
ev = ell_ratios[valid]
kv = Kn_eff_arr[valid]
i_x = np.where(np.diff(np.sign(kv - 1.0)))[0]
if len(i_x):
    j = i_x[0]
    ell_x = np.interp(0.0, np.log10(kv[j:j+2])[::-1] if kv[j] < kv[j+1]
                      else np.log10(kv[j:j+2]),
                      ev[j:j+2][::-1] if kv[j] < kv[j+1] else ev[j:j+2])
    ax.plot([ell_x], [1.0], marker='o', ms=5.5, mfc='white',
            mec='k', mew=1.2, zorder=6)
    ax.annotate(rf'$\ell_\star\!\simeq\!{ell_x:.0f}\,\ell_{{\rm crit}}$',
                xy=(ell_x, 1.0), xytext=(ell_x+1.0, 0.22),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'))

# Inline labels for the horizontal bands. Placed in the open region
# between the curve and the hatched "Reflected" zone.
band_x = 19.0
band_kw = dict(ha='center', va='center', fontsize=10,
               bbox=dict(facecolor='white', edgecolor='none',
                         alpha=0.65, pad=1.5))
ax.text(band_x, 0.18, 'Collisionalized', color='#005f47', **band_kw)
ax.text(band_x, 1.73, 'Marginal',         color='#8a5a00', **band_kw)
ax.text(band_x, 30.0, 'Reservoir',        color='#a23800', **band_kw)

# Inline labels for vertical exclusion bands (rotated, mid-height)
ax.text(0.55, 8.0, 'Direct capture', color='0.25',
        ha='center', va='center', fontsize=9, rotation=90)
ax.text((ell_max/ell_crit + xhi)/2.0, 8.0, 'Reflected',
        color='0.25', ha='center', va='center',
        fontsize=9, rotation=90)

# Axes
ax.set_xlabel(r'$\ell\,/\,\ell_{\rm crit}$')
ax.set_ylabel(r'${\rm Kn}_{\rm eff}$ at $r_{\rm circ}(\ell)$')
ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
ax.legend(loc='lower left', fontsize=10)

# Top axis: r_circ / r_S = 8 (ell/ell_crit)^2
def ell_to_rcirc(x):
    return 8.0 * np.asarray(x)**2
def rcirc_to_ell(y):
    y = np.asarray(y, dtype=float)
    return np.sqrt(np.maximum(y, 0.0) / 8.0)
secax = ax.secondary_xaxis('top', functions=(ell_to_rcirc, rcirc_to_ell))
secax.set_xlabel(r'$r_{\rm circ}\,/\,r_S$', labelpad=6)
secax.set_xticks([10, 100, 1000, 5000])
secax.set_xticklabels(['10', '100', '1000', '5000'])
secax.minorticks_off()

plt.tight_layout()
plt.savefig('docs/kn_map.pdf', bbox_inches='tight')
plt.savefig('docs/kn_map.png', dpi=200, bbox_inches='tight')
print("Saved docs/kn_map.pdf/.png")

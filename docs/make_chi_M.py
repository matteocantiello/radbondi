"""Generate the chi(M) vs log M figure with r_coll/r_sonic ratio."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'mathtext.fontset': 'cm',
})

# Stage 4 results
logM = np.array([-16.0, -15.8, -15.6, -15.3])
chi = np.array([0.688, 0.949, 0.941, 1.000])
r_coll_rS = np.array([6961, 4525, 2524, 1044])
r_sonic_rS = np.array([777, 395, 2693, 17423])
ratio = r_coll_rS / r_sonic_rS

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1],
                                             'hspace': 0.08})

# Top: chi(M)
ax1.plot(logM, chi, 'ko-', ms=8, lw=2.5, zorder=5)
ax1.axhline(1.0, color='gray', ls=':', lw=1)
ax1.axhline(0.7, color='#CC0000', ls='--', lw=1, alpha=0.5)
ax1.fill_between([-16.2, -15.4], 0, 1, alpha=0.06, color='#CC0000',
                  label='correction regime')
ax1.set_ylabel(r'$\chi = \dot{M}/\dot{M}_B$', fontsize=16)
ax1.set_ylim(0.55, 1.08)
ax1.set_xlim(-16.25, -15.1)
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.2)
ax1.text(-15.95, 0.72, r'$\chi \sim 0.7$', fontsize=13, color='#CC0000',
         ha='center')
ax1.text(-15.5, 1.03, r'$\chi = 1$', fontsize=13, color='gray',
         ha='center')

# Bottom: r_coll / r_sonic
ax2.semilogy(logM, ratio, 's-', color='#0072B2', ms=7, lw=2)
ax2.axhline(1.0, color='gray', ls=':', lw=1)
ax2.fill_between([-16.2, -15.1], 0.01, 1, alpha=0.08, color='#009E73',
                  label=r'$r_{\rm coll} < r_{\rm sonic}$ (no correction)')
ax2.fill_between([-16.2, -15.1], 1, 100, alpha=0.08, color='#CC0000',
                  label=r'$r_{\rm coll} > r_{\rm sonic}$ (pile-up subsonic)')
ax2.set_ylabel(r'$r_{\rm coll} / r_{\rm sonic}$', fontsize=16)
ax2.set_xlabel(r'$\log_{10}(M/M_\odot)$', fontsize=16)
ax2.set_ylim(0.03, 30)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.2)

plt.savefig('docs/chi_M.pdf', dpi=300, bbox_inches='tight')
plt.savefig('docs/chi_M.png', dpi=150, bbox_inches='tight')
print("Saved docs/chi_M.pdf/.png")

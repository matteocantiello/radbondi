"""Generate the collisionless pile-up schematic in the paper's style.

Shows concentric radii (r_B, r_coll, r_sonic, r_S), a few representative
particle orbits (captured, misses, spiraling in), bremsstrahlung emission,
and the re-thermalization process at r_coll.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
import matplotlib.colors as mcolors

plt.rcParams.update(plt.rcParamsDefault)
try:
    import seaborn as sns
    sns.set(style="ticks", palette="muted",
            rc={"xtick.bottom": True, "ytick.left": True})
except ImportError:
    pass
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 22,
    'mathtext.fontset': 'cm',
})

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

cx, cy = 0.50, 0.50

# Radii (schematic, not to scale)
r_BH = 0.016
r_sonic = 0.13
r_coll = 0.28
r_B = 0.44

# Colors
col_bondi = '#0072B2'
col_coll = '#009E73'
col_sonic = '#CC6600'
col_bh = 'black'
col_infall = '#555555'
col_orbit_miss = '#CC0000'
col_orbit_cap = '#0066CC'
col_spiral = '#8844AA'
col_photon = '#D55E00'
col_therm = '#8B4513'
col_brems_zone = '#FFCC00'

# =====================================================================
# Background shading
# =====================================================================

# Collisional region: subtle blue tint (r_coll < r < r_B)
n_rings = 40
for i in range(n_rings):
    r_outer = r_coll + (r_B - r_coll) * (i + 1) / n_rings
    frac = i / n_rings
    alpha = 0.10 * (1 - frac)
    ring = plt.Circle((cx, cy), r_outer,
                       fc=mcolors.to_rgba('#0072B2', alpha), ec='none', zorder=0)
    ax.add_patch(ring)
ax.add_patch(Circle((cx, cy), r_coll, fc='white', ec='none', zorder=1))

# Bremsstrahlung emission zone: warm glow around pericenter region
# (between ~r_BH and r_sonic — where dE/E is 1-20%)
n_inner = 50
for i in range(n_inner):
    r_o = r_sonic * (1 - i / n_inner)
    frac = i / n_inner
    alpha = 0.12 * frac**0.6
    ring = plt.Circle((cx, cy), r_o,
                       fc=mcolors.to_rgba('#FF8800', alpha), ec='none', zorder=1)
    ax.add_patch(ring)

# =====================================================================
# Circles for key radii
# =====================================================================
ax.add_patch(Circle((cx, cy), r_B, fc='none', ec=col_bondi, ls='-', lw=2.5, zorder=5))
ax.add_patch(Circle((cx, cy), r_coll, fc='none', ec=col_coll, ls=(0, (8, 5)), lw=3.0, zorder=5))
ax.add_patch(Circle((cx, cy), r_sonic, fc='none', ec=col_sonic, ls=(0, (5, 4)), lw=2.8, zorder=5))
ax.add_patch(Circle((cx, cy), r_BH, fc=col_bh, ec='none', zorder=10))

# =====================================================================
# Radius labels — placed cleanly outside the busy zone
# =====================================================================
ang_rB = np.radians(52)
ax.text(cx + r_B * np.cos(ang_rB) + 0.015,
        cy + r_B * np.sin(ang_rB) + 0.015,
        r'$r_{\rm B}$', fontsize=21, color=col_bondi, ha='left', va='bottom',
        zorder=15)

# r_coll label — top, with white background
ang_rc = np.radians(88)
ax.text(cx + (r_coll) * np.cos(ang_rc),
        cy + (r_coll) * np.sin(ang_rc),
        r'$r_{\rm coll}$', fontsize=20, color=col_coll, ha='center', va='center',
        fontweight='bold', zorder=15,
        bbox=dict(fc='white', ec=col_coll, alpha=0.95, pad=3,
                  boxstyle='round,pad=0.2', lw=1.5))

# r_sonic label — left, with white background
ax.text(cx - r_sonic, cy + 0.005,
        r'$r_{\rm sonic}$', fontsize=18, color=col_sonic, ha='center', va='center',
        fontweight='bold', zorder=15,
        bbox=dict(fc='white', ec=col_sonic, alpha=0.95, pad=3,
                  boxstyle='round,pad=0.2', lw=1.5))

ax.text(cx, cy - r_BH - 0.020, r'BH', fontsize=12,
        color=col_bh, ha='center', va='top', zorder=15,
        fontweight='bold')

# =====================================================================
# Region labels
# =====================================================================

# Collisional region
ax.text(cx + 0.01, cy + r_B - 0.06,
        'Collisional\n(hydrodynamic)',
        fontsize=15, ha='center', va='center', color=col_bondi, style='italic',
        linespacing=0.85,
        bbox=dict(fc='white', ec='none', alpha=0.9, pad=3), zorder=15)

# Collisionless region — on the left side, clear of orbits
ax.text(cx - r_coll + 0.06, cy - 0.05,
        'Two-stream\ncollisionless',
        fontsize=13, ha='center', va='center', color=col_coll, style='italic',
        linespacing=0.85,
        bbox=dict(fc='white', ec='none', alpha=0.9, pad=2), zorder=15)

# Bremsstrahlung loss zone label — white region ABOVE the warm zone
ax.text(cx + 0.02, cy + r_sonic + 0.030,
        r'$\Delta E/E \sim 1$–$20\%$' + '\n' + r'per passage (depends on $r_{\rm peri}$)',
        fontsize=13, ha='center', va='bottom', color=col_photon,
        bbox=dict(fc='white', ec=col_photon, alpha=0.95, pad=3,
                  boxstyle='round,pad=0.2', lw=1.0), zorder=15)


# =====================================================================
# Particle orbits — 3 clean, well-separated orbits
# =====================================================================

def draw_elliptical_orbit(ax, cx, cy, r_apo, r_peri, angle, color,
                          lw=2.0, alpha=0.7, zorder=6, arrow_pos=0.75):
    """Draw an elliptical orbit centered on the BH."""
    a = (r_apo + r_peri) / 2
    b = np.sqrt(r_apo * r_peri)
    c_off = a - r_peri  # distance from BH (focus) to center of ellipse

    theta = np.linspace(0, 2 * np.pi, 400)
    x_ell = a * np.cos(theta)
    y_ell = b * np.sin(theta)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Shift so BH is at focus (offset center by c_off along major axis)
    x_rot = cx + (x_ell - c_off) * cos_a - y_ell * sin_a
    y_rot = cy + (x_ell - c_off) * sin_a + y_ell * cos_a

    ax.plot(x_rot, y_rot, color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle='round')

    # Arrowhead
    idx = int(len(theta) * arrow_pos)
    di = 3
    dx = x_rot[(idx + di) % len(theta)] - x_rot[(idx - di) % len(theta)]
    dy = y_rot[(idx + di) % len(theta)] - y_rot[(idx - di) % len(theta)]
    ln = np.sqrt(dx**2 + dy**2)
    if ln > 0:
        ux, uy = dx / ln, dy / ln
        px, py = -uy, ux
        hl = 0.016
        hw = hl * 0.5
        tri = Polygon([
            [x_rot[idx] + hl * 0.3 * ux, y_rot[idx] + hl * 0.3 * uy],
            [x_rot[idx] - hl * ux + hw * px, y_rot[idx] - hl * uy + hw * py],
            [x_rot[idx] - hl * ux - hw * px, y_rot[idx] - hl * uy - hw * py],
        ], closed=True, fc=color, ec='none', alpha=alpha, zorder=zorder)
        ax.add_patch(tri)


# Orbit A: CAPTURED — nearly radial, small pericenter (right side)
# Extends from collisional zone past r_coll into the BH
draw_elliptical_orbit(ax, cx, cy, r_apo=r_coll + 0.06, r_peri=r_BH * 0.7,
                      angle=np.radians(-15), color=col_orbit_cap, lw=2.2,
                      alpha=0.55, arrow_pos=0.70)

# Orbit B: MISSES — larger pericenter, comes from collisional zone,
# crosses r_coll, swings through, returns past r_coll
draw_elliptical_orbit(ax, cx, cy, r_apo=r_coll + 0.06, r_peri=0.065,
                      angle=np.radians(160), color=col_orbit_miss, lw=2.2,
                      alpha=0.65, arrow_pos=0.75)

# Orbit C: SPIRALING IN — after multiple passes with energy loss,
# orbit has shrunk. Two progressively tighter passes.
draw_elliptical_orbit(ax, cx, cy, r_apo=r_sonic + 0.02, r_peri=0.032,
                      angle=np.radians(245), color=col_spiral, lw=1.8,
                      alpha=0.55, arrow_pos=0.72)
draw_elliptical_orbit(ax, cx, cy, r_apo=r_sonic - 0.015, r_peri=0.022,
                      angle=np.radians(275), color=col_spiral, lw=1.5,
                      alpha=0.40, arrow_pos=0.72)


# =====================================================================
# Wavy bremsstrahlung photons emitted from the hot zone
# =====================================================================

def draw_wavy_photon(ax, x0, y0, x1, y1, color=col_photon, alpha=0.5,
                     lw=1.5, n_waves=3):
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_len = 0.014
    frac_wave = max(1.0 - head_len / length - 0.01, 0.65)
    t = np.linspace(0, frac_wave, 60)
    amp = 0.007
    xw = x0 + t * dx + amp * np.sin(t * n_waves * 2 * np.pi) * px
    yw = y0 + t * dy + amp * np.sin(t * n_waves * 2 * np.pi) * py
    ax.plot(xw, yw, color=color, lw=lw, alpha=alpha, zorder=8,
            solid_capstyle='round')
    hw = head_len * 0.45
    tri = Polygon([
        [x1, y1],
        [x1 - head_len * ux + hw * px, y1 - head_len * uy + hw * py],
        [x1 - head_len * ux - hw * px, y1 - head_len * uy - hw * py],
    ], closed=True, fc=color, ec='none', alpha=alpha, zorder=8)
    ax.add_patch(tri)


# Photons radiating outward from the hot zone
photon_data = [
    (np.radians(20),  0.050, 0.08),
    (np.radians(80),  0.055, 0.08),
    (np.radians(190), 0.045, 0.07),
    (np.radians(290), 0.060, 0.08),
    (np.radians(340), 0.050, 0.07),
]
for ang, r0, dr in photon_data:
    x0 = cx + r0 * np.cos(ang)
    y0 = cy + r0 * np.sin(ang)
    x1 = cx + (r0 + dr) * np.cos(ang + np.radians(10))
    y1 = cy + (r0 + dr) * np.sin(ang + np.radians(10))
    draw_wavy_photon(ax, x0, y0, x1, y1, alpha=0.45)


# =====================================================================
# Infall arrows at r_B
# =====================================================================
for i in range(8):
    angle = 2 * np.pi * i / 8 + np.radians(10)
    r_s = r_B + 0.055
    r_e = r_B + 0.012
    ax.annotate('',
                xy=(cx + r_e * np.cos(angle), cy + r_e * np.sin(angle)),
                xytext=(cx + r_s * np.cos(angle), cy + r_s * np.sin(angle)),
                arrowprops=dict(arrowstyle='->', color=col_infall, lw=1.6,
                               alpha=0.35))




# =====================================================================
# Orbit labels — placed outside the circles, connected by thin lines
# =====================================================================

# =====================================================================
# Legend box — lower left corner
# =====================================================================
leg_x, leg_y = 0.02, 0.93
leg_spacing = 0.032
leg_fontsize = 12
line_len = 0.035

entries = [
    (col_orbit_cap, '-',  r'Captured ($\ell < \ell_{\rm crit}$)'),
    (col_orbit_miss, '-', r'Bounces back ($\ell > \ell_{\rm crit}$)'),
    (col_spiral, '-',     r'Orbit shrinkage (energy loss)'),
]

for i, (color, ls, label) in enumerate(entries):
    y_i = leg_y - i * leg_spacing
    ax.plot([leg_x, leg_x + line_len], [y_i, y_i],
            color=color, lw=2.5, ls=ls, alpha=0.8, zorder=15)
    ax.text(leg_x + line_len + 0.010, y_i, label,
            fontsize=leg_fontsize, color=color, ha='left', va='center',
            zorder=15)


# Infall label
ax.text(cx - r_B - 0.02, cy - 0.06,
        r'$\dot{M}_B$', fontsize=20, ha='right', va='center', color=col_infall,
        zorder=15)

# Scale note — lower right corner
ax.text(0.97, 0.020,
        r'(not to scale: $r_{\rm coll}/r_S \approx 7000$,'
        r'$\;r_B/r_{\rm coll} \approx 25$)',
        fontsize=13, ha='right', va='bottom', color='#555555', style='italic',
        zorder=15)

ax.set_xlim(0.00, 1.00)
ax.set_ylim(0.00, 1.00)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('docs/schematic_pileup.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0.02)
plt.savefig('docs/schematic_pileup.png', dpi=150, bbox_inches='tight',
            pad_inches=0.02)
print("Saved docs/schematic_pileup.pdf/.png")

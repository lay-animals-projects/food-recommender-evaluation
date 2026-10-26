"""Render an area-proportional two-set Venn diagram using supplied counts."""
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


OUT = Path(__file__).resolve().parent
TOTAL = 536
PREFERENCE_ONLY, BOTH, HEALTH_ONLY, NEITHER = 19, 245, 264, 8
assert PREFERENCE_ONLY + BOTH + HEALTH_ONLY + NEITHER == TOTAL

# Circle areas encode set sizes; solve the separation for the exact overlap.
r = math.sqrt((PREFERENCE_ONLY + BOTH) / TOTAL / math.pi)
R = math.sqrt((HEALTH_ONLY + BOTH) / TOTAL / math.pi)
target = BOTH / TOTAL


def overlap(d):
    a = math.acos(max(-1, min(1, (d*d + r*r - R*R) / (2*d*r))))
    b = math.acos(max(-1, min(1, (d*d + R*R - r*r) / (2*d*R))))
    radicand = (-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)
    return r*r*a + R*R*b - 0.5*math.sqrt(max(0, radicand))


lo, hi = R-r, R+r
for _ in range(100):
    mid = (lo+hi)/2
    if overlap(mid) > target:
        lo = mid
    else:
        hi = mid
d = (lo+hi)/2
assert abs(overlap(d) - target) < 1e-10

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig = plt.figure(figsize=(12, 7.2), facecolor="white")
fig.text(.065, .935, "Preference and health coverage", fontsize=24,
         weight="bold", color="#172B3A")
fig.text(.065, .888, "Overlap across 536 recommendation explanations",
         fontsize=13, color="#5E6B75")

ax = fig.add_axes([.045, .18, .91, .66])
ax.set_aspect("equal")
ax.set_xlim(-.94, 1.12)
ax.set_ylim(-.68, .75)
ax.axis("off")
ax.add_patch(Rectangle((-.93, -.665), 2.04, 1.40,
                       facecolor="#FCFDFE", edgecolor="#D8E1E7", lw=1.2))

x1, x2 = -.19, -.19+d
pref = Circle((x1, 0), r, facecolor="#E9C46A", alpha=.72,
              edgecolor="#B08421", lw=2, zorder=2)
health = Circle((x2, 0), R, facecolor="#2A9D8F", alpha=.38,
                edgecolor="#16776E", lw=2, zorder=3)
ax.add_patch(pref)
ax.add_patch(health)

ax.text(-.68, .64, "USER PREFERENCES", fontsize=12, weight="bold",
        color="#896414", ha="left")
ax.text(-.68, .565, "264 explanations · 49.3%", fontsize=11,
        color="#5E6B75", ha="left")
ax.text(.51, .64, "HEALTH JUSTIFICATION", fontsize=12, weight="bold",
        color="#126F66", ha="center")
ax.text(.51, .565, "509 explanations · 95.0%", fontsize=11,
        color="#5E6B75", ha="center")

# The small preference-only crescent gets an external label for legibility.
crescent_x = ((x1-r) + (x2-R))/2
ax.annotate("Preference only\n19 (3.5%)", xy=(crescent_x, 0),
            xytext=(-.89, .27), ha="left", va="center", fontsize=13,
            color="#172B3A", linespacing=1.7,
            arrowprops={"arrowstyle": "-", "color": "#896414", "lw": 1.5,
                        "connectionstyle": "angle,angleA=0,angleB=90,rad=10"},
            zorder=6)

ax.text(-.12, .065, "Both", ha="center", fontsize=15,
        color="#172B3A", zorder=6)
ax.text(-.12, -.025, "245", ha="center", va="center", fontsize=28,
        weight="bold", color="#172B3A", zorder=6)
ax.text(-.12, -.125, "45.7%", ha="center", fontsize=14,
        color="#172B3A", zorder=6)

health_label_x = (x1+r + x2+R)/2
ax.text(health_label_x, .065, "Health only", ha="center", fontsize=15,
        color="#172B3A", zorder=6)
ax.text(health_label_x, -.025, "264", ha="center", va="center", fontsize=28,
        weight="bold", color="#172B3A", zorder=6)
ax.text(health_label_x, -.125, "49.3%", ha="center", fontsize=14,
        color="#172B3A", zorder=6)

ax.text(.99, -.50, "Neither", ha="right", fontsize=13, color="#5E6B75")
ax.text(.99, -.59, "8 (1.5%)", ha="right", fontsize=17,
        weight="bold", color="#172B3A")

fig.text(.065, .105, "Percentages use all 536 explanations as the denominator.",
         fontsize=11, color="#5E6B75")
fig.text(.065, .067, "Circle areas and overlap are proportional to counts; values are rounded to one decimal place.",
         fontsize=10.5, color="#5E6B75")

fig.savefig(OUT / "coverage_venn.png", dpi=220, facecolor="white")
fig.savefig(OUT / "coverage_venn.svg", facecolor="white")
print(OUT / "coverage_venn.png")

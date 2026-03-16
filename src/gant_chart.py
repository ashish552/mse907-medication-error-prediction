import matplotlib.pyplot as plt

tasks = [
    ("Topic selection & initial scope", 1, 1),
    ("Literature review + gap identification", 2, 3),
    ("Proposal writing & submission", 4, 6),
    ("Data extraction & dataset construction", 6, 8),
    ("Preprocessing & feature engineering", 8, 10),
    ("Proxy label development & validation", 9, 10),
    ("Model training & tuning (group split)", 10, 12),
    ("Evaluation & comparison reporting", 11, 12),
    ("Explainability analysis", 11, 12),
    ("Prototype development (Streamlit)", 12, 12),
    ("WIP report writing & submission", 12, 13),
    ("UI polishing (CSS + layout refinements)", 14, 14),
    ("Final report & presentation preparation", 15, 15),
]

labels = [t[0] for t in tasks]
starts = [t[1] for t in tasks]
ends = [t[2] for t in tasks]
durations = [(e - s + 1) for s, e in zip(starts, ends)]
y_pos = list(range(len(tasks)))

fig, ax = plt.subplots(figsize=(12, 6.5))

for i, (name, s, e) in enumerate(tasks):
    ax.barh(i, e - s + 1, left=s, height=0.55)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()

weeks = list(range(1, 16))
ax.set_xlim(1, 15)
ax.set_xticks(weeks)
ax.set_xticklabels([f"W{w}" for w in weeks], fontsize=8)

ax.grid(True, axis="x")
ax.grid(False, axis="y")

ax.set_xlabel("Project Weeks", fontsize=8)
ax.set_title("Project Timeline (Week 1–Week 15)", fontsize=10)

# Dark border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig("gantt_chart_weeks_clean.png", dpi=300, bbox_inches="tight")
plt.show()
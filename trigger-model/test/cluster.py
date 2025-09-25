import os
import matplotlib.pyplot as plt
import numpy as np

timesteps = [90, 300, 900, 1500, 2400, 3000]
intra_distance_means = [0.19665, 0.1813, 0.174083, 0.168814285, 0.169885714, 0.18365]
intra_distance_stds = [0.087893, 0.057117, 0.045861, 0.044909, 0.044556, 0.055935]

inter_distance_means = [0.2253, 0.1875, 0.1601, 0.1689, 0.1647, 0.1829]
inter_distance_stds = [0, 0.078154846, 0.098516, 0.094037035, 0.094819, 0.079118]

cluster_counts = [2, 4, 6, 7, 7, 4]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(timesteps, intra_distance_means, marker='o', color='#1f77b4', label='Intra-cluster Distance')
ax1.fill_between(timesteps,
                 np.array(intra_distance_means) - np.array(intra_distance_stds),
                 np.array(intra_distance_means) + np.array(intra_distance_stds),
                 color='#1f77b4', alpha=0.2)

ax1.plot(timesteps, inter_distance_means, marker='s', color='#7f7f7f', label='Inter-cluster Distance')
ax1.fill_between(timesteps,
                 np.array(inter_distance_means) - np.array(inter_distance_stds),
                 np.array(inter_distance_means) + np.array(inter_distance_stds),
                 color='#7f7f7f', alpha=0.2)
ax1.set_xlabel("Streaming Data Volume (N)", fontsize=24)
ax1.set_ylabel("Cluster Distance", fontsize=24)
ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(axis='x', labelsize=18)
ax1.grid(True, linestyle='--', alpha=0.5)
ax2 = ax1.twinx()
ax2.plot(timesteps, cluster_counts, marker='^', color='#d62728', label='Cluster Count', linestyle='--')
ax2.set_ylabel("Number of Clusters", fontsize=24)
ax2.tick_params(axis='y', labelsize=18)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=18)

plt.title("Cluster Distance and Count Over Data Volume", fontsize=32)
plt.tight_layout()
plt.xticks(timesteps,fontsize = 18) 
tsne_path = os.path.join("/path/to/your/trigger/results", "cluster_distance.png") 
plt.savefig(tsne_path, dpi=300, bbox_inches="tight") 
print(f"[INFO] t-SNE图已保存到 {tsne_path}") 
plt.show()

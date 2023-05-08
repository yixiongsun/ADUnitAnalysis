import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(8,3))
sns.set_style('ticks')
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
plt.plot(x * 180 / np.pi, np.cos(x),c='black')
plt.xticks([-360, -270,-180,-90,0,90,180,270,360])
plt.xlim(-180, 180)
plt.yticks([])
sns.despine(left=True,bottom=False,right=True,top=True)
plt.savefig("output/sine.png", dpi=300)
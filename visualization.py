import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

x, y, x1, x2, x3 = preprocess(output_param='G')
x, y = x.to_numpy(), y.to_numpy().reshape(-1, 1)

n_components = 2
tsne = TSNE(n_components=n_components)
z = tsne.fit_transform(x1)
print(z)
print(z.shape)

df = pd.DataFrame()
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

# Have to manually choose the number of classes
sns.scatterplot(x="comp-1", y="comp-2",
                palette=sns.color_palette("hls", 3),
                data=df).set(title="Iris data T-SNE projection")
plt.show()
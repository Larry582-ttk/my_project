"""PCA 降维探索（可选运行，用来观察降维效果）"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('data/raw/all_accidents.csv', encoding='utf-8-sig')
titles = df['Accident Title'].tolist()

# 清洗
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

titles_clean = [clean_text(t) for t in titles]

# 向量化
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(titles_clean)

# PCA 降维（保留95%方差）
pca = PCA(n_components=0.95)
embeddings_pca = pca.fit_transform(embeddings)  #84个特征
print(f"Dimension reduced: 398 -> {embeddings_pca.shape[1]}")

# 降维后聚类
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.85,
    metric='cosine',
    linkage='average'
)
labels = clustering.fit_predict(embeddings_pca)

# 画树状图
linkage_matrix = linkage(embeddings_pca, method='average', metric='cosine')
plt.figure(figsize=(16, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
)
plt.tight_layout()
plt.show()

# 输出聚类结果
for cluster_id in sorted(set(labels)):
    indices = [i for i, label in enumerate(labels) if label == cluster_id]
    print(f"Cluster {cluster_id} (total {len(indices)}):")
    for idx in indices[:5]:
        print(f"  - {titles[idx]}")

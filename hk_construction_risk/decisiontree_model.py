"""使用决策树进行分类"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import re
import csv
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
import joblib

# ==================== 配置 ====================
RAW_DATA = 'data/raw/all_accidents.csv'
TEST_DATA = 'data/raw/test_set.csv'
LABELED_FILE = 'data/intermediate/labeled.csv'
TO_LABEL_FILE = 'data/intermediate/to_label.csv'
OUTPUT_LABELS = 'data/output/accidents_with_labels.csv'
OUTPUT_PREDICTIONS = 'data/output/prediction_results_decision_tree.csv'
MODEL_SAVE_PATH = 'models/decision_tree_model.pkl'

# ==================== 创建输出目录 ====================
os.makedirs('data/output', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/intermediate', exist_ok=True)

# 9 大类名称
MAJOR_NAMES = {
    0: "Others",
    1: "Fall from Height",
    2: "Struck by Falling Object",
    3: "Struck by vehicle / moving object",
    4: "Trapped / crushed / caught / collapse",
    5: "Electrocution",
    6: "Exposure to harmful substance / asphyxiation / poisoning",
    7: "Fire / explosion / thermal burn",
    8: "Drowning",
}

# 聚类ID -> 大类ID 映射
OTHER_MAP = {
    0: 7, 1: 6, 2: 4, 6: 6, 8: 5, 9: 8,
    10: 1, 11: 8, 12: 4, 16: 4, 17: 4, 18: 7, 22: 3, 23: 7
}

# 需要人工标注的聚类ID
TYPES_TO_MANUAL = [3, 4, 5, 7, 13, 14, 20, 21]


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==================== 主流程 ====================

# 1. 读取数据
print("Loading data...")
df = pd.read_csv(RAW_DATA, encoding='utf-8-sig')
titles = df['Accident Title'].tolist()

# 2. 清洗
titles_clean = [clean_text(t) for t in titles]

# 3. 向量化
print("Encoding with Sentence-BERT...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(titles_clean)

# 4. 层次聚类
print("Clustering...")
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.6,
    metric='cosine',
    linkage='average'
)
labels = clustering.fit_predict(embeddings)

# 5. 映射到9大类
final_labels = [0] * len(titles)
for i, label in enumerate(labels):
    if label in OTHER_MAP:
        final_labels[i] = OTHER_MAP[label]

# 6. 读取人工标注
if os.path.exists(LABELED_FILE):
    print(f"Loading manual labels from {LABELED_FILE}...")
    with open(LABELED_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['index'])
            label_val = int(row['label'])
            final_labels[idx] = label_val
else:
    print("Exporting unlabeled data for manual labeling...")
    to_label = []
    for i, label in enumerate(labels):
        if label in TYPES_TO_MANUAL:
            to_label.append({'index': i, 'type': label, 'sub_type': '', 'title': titles[i], 'label': ''})
    with open(TO_LABEL_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'type', 'sub_type', 'title', 'label'])
        writer.writeheader()
        writer.writerows(to_label)
    print(f"Saved {len(to_label)} items to {TO_LABEL_FILE}")
    print(f"Please label them in {LABELED_FILE} and re-run.")
    exit()

# 7. 统计类别分布
print("\nCategory distribution:")
for label, count in sorted(Counter(final_labels).items()):
    print(f"  {MAJOR_NAMES[label]}: {count}")

# 8. 训练决策树
print("\nTraining Decision Tree...")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(embeddings, final_labels)
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# 9. 保存带标签的数据
label_data = pd.DataFrame({
    'time': df['Date'],
    'titles': titles,
    'labels': final_labels,
    'types': [MAJOR_NAMES[l] for l in final_labels]
})
label_data.to_csv(OUTPUT_LABELS, index=False, encoding='utf-8-sig')
print(f"Labeled data saved to {OUTPUT_LABELS}")

# 10. 预测新数据
print(f"\nPredicting new data from {TEST_DATA}...")
df_new = pd.read_csv(TEST_DATA, encoding='utf-8-sig')
titles2 = df_new['Accident'].tolist()
new_titles = [clean_text(t) for t in titles2]
new_embeddings = model.encode(new_titles)
predictions = clf.predict(new_embeddings)

print("\nPrediction results:")
for title, pred in zip(titles2, predictions):
    print(f"  {title} -> {MAJOR_NAMES[pred]}")

print("\nPrediction distribution:")
for label, count in sorted(Counter(predictions).items()):
    print(f"  {MAJOR_NAMES[label]}: {count}")

results_df = pd.DataFrame({
    'titles': titles2,
    'prediction_labels': predictions,
    'prediction_types': [MAJOR_NAMES[p] for p in predictions]
})
results_df.to_csv(OUTPUT_PREDICTIONS, index=False, encoding='utf-8-sig')
print(f"Results saved to {OUTPUT_PREDICTIONS}")

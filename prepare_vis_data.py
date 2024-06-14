import networkx as nx
import pickle
import config
import pandas as pd
import ast
import json
from collections import Counter

# 加载图数据
graph_path = config.graph_path
with open(r"D:\social_network\data\graph_data\demo.pkl", 'rb') as f:
    G = pickle.load(f)

# 加载labels数据
labels_path = r'D:\social_network\data\labels.csv'
labels_df = pd.read_csv(labels_path)

# 将labels数据转换为字典
labels_dict = {}
for index, row in labels_df.iterrows():
    labels_dict[row['user_id']] = ast.literal_eval(row['labels'])

# 统计标签出现次数
all_labels = [label for labels in labels_dict.values() for label in labels]
label_counts = Counter(all_labels)

# 获取出现次数最多的前10个标签
top_labels = [label for label, _ in label_counts.most_common(10)]

# 保留100个节点及其相应的边
subgraph = G.subgraph(list(G.nodes)[:100])

# 创建节点和边的数据
nodes = []
edges = []
for node in subgraph.nodes:
    labels = labels_dict.get(node, [])
    nodes.append({"id": int(node), "labels": labels})  # 转换为int类型

for edge in subgraph.edges:
    edges.append({"source": int(edge[0]), "target": int(edge[1])})  # 转换为int类型

# 保存为JSON文件，确保中文字符正确显示
graph_data = {"nodes": nodes, "edges": edges, "top_labels": top_labels}
with open("graph_data.json", "w", encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=4)

import pandas as pd
import networkx as nx
import pickle
import os
from SLPA import SLPA as SLPA_altered
from SLPA_base import SLPA as SLPA_base
import config
from sklearn.metrics import precision_recall_fscore_support
import random
random.seed(2024)

graph_path = config.graph_path
weibo_data_path = config.data_path
neo4j_data_path = config.neo4j_path
result_path = config.result_path

experiment_setting = 'altered_demo' # base or altered; demo or full, 部分数据 or 全量数据

if experiment_setting.split('_')[1] == 'demo':
    users_df = pd.read_csv(f"{weibo_data_path}/users_demo.csv")
else:
    users_df = pd.read_csv(f"{weibo_data_path}/users_full.csv")

users_df["TAGS"] = users_df["TAGS"].fillna(' ')

filtered_users_set = users_df["USER_ID"].values

tags_dict = {row["USER_ID"]: row["TAGS"].split('_') if row["TAGS"] != ' ' else []
             for index, row in users_df.iterrows()}
print("Tags dictionary created.")

# 划分传播集和测试集
propagation_ratio = 0.9
users_list = list(tags_dict.keys())
random.shuffle(users_list)
propagation_count = int(len(users_list) * propagation_ratio)
propagation_users = set(users_list[:propagation_count])
test_users = set(users_list[propagation_count:])

propagation_tags = {user: tags for user, tags in tags_dict.items() if user in propagation_users}
test_tags = {user: [] for user in test_users}  # 测试集用户初始化标签为空集

# 合并标签字典
all_tags = {**propagation_tags, **test_tags}

# 保存测试集标签到 CSV 文件
test_tags_df = pd.DataFrame(
    [{'user_id': user, 'tags': tags_dict[user]} for user in test_users])
test_tags_df = test_tags_df.sort_values(by='user_id')
os.makedirs(f"{result_path}/SLPA_{experiment_setting}", exist_ok=True)
test_tags_df.to_csv(f"{result_path}/SLPA_{experiment_setting}/test_labels.csv", index=False)

if os.path.exists(f"{graph_path}/{experiment_setting.split('_')[1]}.pkl"):
    print("Loading graph from existing pkl...")
    with open(f"{graph_path}/{experiment_setting.split('_')[1]}.pkl", 'rb') as f:
        G = pickle.load(f)
    print(f"Graph loaded. nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
else:
    print("Creating graph...")
    edges_df = pd.read_csv(f'{weibo_data_path}/第0层+第1层+第2层关系边.txt',
                           sep='\t', header=None, names=["blogger", "follower"])
    forward_edges_df = pd.read_csv(f'{weibo_data_path}/第0层+第1层+第2层关系且转发边.txt',
                                   sep='\t', header=None, names=["blogger", "follower", "count"])
    print("edges loaded.")

    filtered_edges_df = edges_df[(edges_df['blogger'].isin(filtered_users_set)) & (
        edges_df['follower'].isin(filtered_users_set))]
    filtered_forward_edges_df = forward_edges_df[(forward_edges_df['blogger'].isin(
        filtered_users_set)) & (forward_edges_df['follower'].isin(filtered_users_set))]
    print("edges filtered.")
    # 创建有向图
    G = nx.DiGraph()
    G.add_nodes_from(filtered_users_set)
    edges_with_weights = [(row["follower"], row["blogger"], 1)
                          for _, row in filtered_edges_df.iterrows()]
    forward_edges_with_weights = [(row["follower"], row["blogger"], row["count"] + 1)
                                  for _, row in filtered_forward_edges_df.iterrows()]
    G.add_weighted_edges_from(edges_with_weights)
    G.add_weighted_edges_from(forward_edges_with_weights)
    print("Graph structure created.")
    with open(f"{graph_path}/{experiment_setting.split('_')[1]}.pkl", 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph structure saved. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# 计算 PageRank
if experiment_setting.split('_')[0] == 'altered':
    page_rank = nx.pagerank(G, weight='weight')

# 运行 SLPA 算法
T = 10  # 迭代次数
r = 0.3  # 阈值

if experiment_setting.split('_')[0] == 'base':
    slpa = SLPA_base(G, T, r, propagation_tags)
else:
    slpa = SLPA_altered(G, T, r, propagation_tags, page_rank)

communities = slpa.execute()
print("Label propagation completed.")

# 将 SLPA 社区结果转化为标签
final_labels = {}
for label, nodes in communities.items():
    for node in nodes:
        if node not in final_labels:
            final_labels[node] = []
        final_labels[node].append(label)

# 将传播后的标签与原始标签合并
for node in all_tags:
    if node in final_labels:
        final_labels[node] = list(set(final_labels[node]) | set(all_tags[node]))
    else:
        final_labels[node] = all_tags[node]

# 将最终标签转换为 DataFrame 以便查看
final_labels_df = pd.DataFrame(
    [(node, labels) for node, labels in final_labels.items()], columns=["user_id", "labels"])

# 保存最初的和最后的标签
result_dir = f"{result_path}/SLPA_{experiment_setting}"
os.makedirs(result_dir, exist_ok=True)

# 将原始的 tags_dict 保存为 initial_labels.csv
initial_labels_df = pd.DataFrame(
    [(node, labels) for node, labels in tags_dict.items()], columns=["user_id", "labels"])

initial_labels_df = initial_labels_df.sort_values(by="user_id")
initial_labels_df.to_csv(f'{result_dir}/initial_labels.csv', index=False)

# 确保 final_labels_df 包含所有用户
final_labels_df = final_labels_df.sort_values(by="user_id")
for user_id in initial_labels_df["user_id"]:
    if user_id not in final_labels_df["user_id"].values:
        final_labels_df = final_labels_df.append(
            {"user_id": user_id, "labels": []}, ignore_index=True
        )

final_labels_df = final_labels_df.sort_values(by="user_id")
final_labels_df.to_csv(f'{result_dir}/final_labels.csv', index=False)

print("Final labels saved.")

# 计算测试集的Micro-F1
true_labels = []
pred_labels = []

for user_id in test_users:
    true_set = set(tags_dict[user_id])
    pred_set = set(final_labels[user_id])
    true_labels.extend([1] * len(true_set))
    pred_labels.extend([1 if tag in pred_set else 0 for tag in true_set])

precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='micro')

# Macro-F1 计算方式
true_labels_macro = []
pred_labels_macro = []

for user_id in test_users:
    true_set = set(tags_dict[user_id])
    pred_set = set(final_labels[user_id])
    if len(true_set) > 0:  # 确保用户有真实标签
        true_labels_macro.append(1)
        pred_labels_macro.append(1 if len(true_set & pred_set) > 0 else 0)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    true_labels_macro, pred_labels_macro, average='binary')

print(f"Micro Precision: {precision_micro}")
print(f"Micro Recall: {recall_micro}")
print(f"Micro F1-Score: {f1_micro}")

print(f"Macro Precision: {precision_macro}")
print(f"Macro Recall: {recall_macro}")
print(f"Macro F1-Score: {f1_macro}")

metrics_result_df = pd.DataFrame([{
    'Micro Precision': precision_micro,
    'Micro Recall': recall_micro,
    'Micro F1-Score': f1_micro,
    'Macro Precision': precision_macro,
    'Macro Recall': recall_macro,
    'Macro F1-Score': f1_macro
}])
metrics_result_df.to_csv(f"{result_dir}/metrics_result.csv", index=False)
print("Metrics result saved。")
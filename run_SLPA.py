import pandas as pd
import networkx as nx
import pickle
import os
from SLPA import SLPA
import config
from utils import split_tags, compare_labels
from sklearn.metrics import precision_recall_fscore_support

graph_path = config.graph_path
weibo_data_path = config.data_path
neo4j_data_path = config.neo4j_path
result_path = config.result_path

experiment_setting = 'demo' # demo or full, 全量跑可能会很慢

if experiment_setting == 'demo':
    users_df = pd.read_csv(
        f"{weibo_data_path}/users_demo.csv")
else:
    users_df = pd.read_csv(
        f"{weibo_data_path}/users_full.csv")

# duplicate_user_ids = users_df[users_df.duplicated(subset="USER_ID", keep=False)]  # 某些节点既是1层，又是2层
# if not duplicate_user_ids.empty:
#     print(f"Found duplicate USER_IDs: {duplicate_user_ids}")

# TAG列填充空值
users_df["TAGS"] = users_df["TAGS"].fillna(' ')

filtered_users_set = users_df["USER_ID"].values

# 读取这些用户的tag
tags_dict = {row["USER_ID"]: row["TAGS"].split('_') if row["TAGS"]!=' ' else []
             for index, row in users_df.iterrows()}
print("Tags dictionary created.")

# 标签切割，分为传播标签和验证标签
propagation_tags, validation_tags = split_tags(tags_dict,val_ratio=0.5)

# 保存validation_tags到CSV文件
validation_tags_df = pd.DataFrame(
    [{'user_id': user, 'tags': tags} for user, tags in validation_tags.items()])
# 排序
validation_tags_df = validation_tags_df.sort_values(by='user_id')
validation_tags_df.to_csv(f"{result_path}/SLPA/validation_labels.csv", index=False)

if os.path.exists(f"{graph_path}/{experiment_setting}.pkl"):
    print("Loading graph from existing pkl...")
    with open(f"{graph_path}/{experiment_setting}.pkl", 'rb') as f:
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
    # 添加所有保留的用户到图中
    G.add_nodes_from(filtered_users_set)
    # 为 edges_df 中的边设置默认权重 1
    edges_with_weights = [(row["follower"], row["blogger"], 1)
                          for _, row in filtered_edges_df.iterrows()]
    # 为 forward_edges_df 中的边设置指定权重
    forward_edges_with_weights = [(row["follower"], row["blogger"], row["count"])
                                  for _, row in filtered_forward_edges_df.iterrows()]
    # 批量添加所有边到图中
    G.add_weighted_edges_from(edges_with_weights)
    G.add_weighted_edges_from(forward_edges_with_weights)
    print("Graph structure created.")
    # 保存图结构
    with open(f"{graph_path}/{experiment_setting}.pkl", 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph structure saved. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
# 运行SLPA算法
T = 10  # 迭代次数
r = 0.3  # 阈值

slpa = SLPA(G, T, r, tags_dict)
communities = slpa.execute()
print("Label propagation completed.")

# 将SLPA社区结果转化为标签
final_labels = {}
for label, nodes in communities.items():
    for node in nodes:
        if node not in final_labels:
            final_labels[node] = []
        final_labels[node].append(label)

# 将传播后的标签与原始标签合并
for node in tags_dict:
    if node in final_labels:
        final_labels[node] = list(set(final_labels[node]) | set(tags_dict[node]))
    else:
        final_labels[node] = tags_dict[node]

# 将最终标签转换为 DataFrame 以便查看
final_labels_df = pd.DataFrame(
    [(node, labels) for node, labels in final_labels.items()], columns=["user_id", "labels"])

# 保存最初的和最后的标签
result_dir = f"{result_path}/SLPA"
os.makedirs(result_dir, exist_ok=True)

# 将原始的tags_dict保存为initial_labels.csv
initial_labels_df = pd.DataFrame(
    [(node, labels) for node, labels in tags_dict.items()], columns=["user_id", "labels"])

# 按照user_id排序
initial_labels_df = initial_labels_df.sort_values(by="user_id")
initial_labels_df.to_csv(f'{result_dir}/initial_labels.csv', index=False)



# 确保final_labels_df包含所有用户
final_labels_df = final_labels_df.sort_values(by="user_id")
for user_id in initial_labels_df["user_id"]:
    if user_id not in final_labels_df["user_id"].values:
        final_labels_df = final_labels_df.append(
            {"user_id": user_id, "labels": []}, ignore_index=True
        )

# 再次排序以确保顺序一致
final_labels_df = final_labels_df.sort_values(by="user_id")
final_labels_df.to_csv(f'{result_dir}/final_labels.csv', index=False)

print("Final labels saved.")

# 检查传播前后的差异
labels_added_df = pd.merge(
    initial_labels_df, final_labels_df, on='user_id', suffixes=('_initial', '_final'))

# 比较标签集合前后的变化
labels_added_result_df = labels_added_df.apply(compare_labels, axis=1)

# 保存比较结果到CSV文件，包含user_id、added_tags和removed_tags列
labels_added_result_df.to_csv(f"{result_dir}/labels_added.csv", index=False)

# 计算指标
# 将validation_tags_df与labels_added_result合并
labels_comparison_df = pd.merge(
    validation_tags_df, labels_added_result_df, on='user_id', suffixes=('_validation', '_added'))

# 初始化精度、召回率和F1分数计算所需的标签列表
true_labels = []
pred_labels = []

# 计算每个用户的精度、召回率和F1分数
for index, row in labels_comparison_df.iterrows():
    validation_set = set(row['tags_validation'])
    added_set = set(row['tags_added'])
    true_labels.extend([1] * len(validation_set))
    pred_labels.extend([1 if tag in added_set else 0 for tag in validation_set])

# 使用micro指标计算精度、召回率和F1分数
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='micro')

# 输出结果
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# 保存计算结果到CSV文件
metrics_result_df = pd.DataFrame([{
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}])
metrics_result_df.to_csv(f"{result_dir}/metrics_result.csv", index=False)
print("Metrics result saved。")
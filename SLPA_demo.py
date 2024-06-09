import pandas as pd
import networkx as nx
import pickle
import os
from SLPA import SLPA
import config

# 加载图结构和标签数据
graph_path = config.graph_path
graph_name = "social_network_graph_demo"
weibo_data_path = config.data_path
neo4j_data_path = config.neo4j_path
result_path = config.result_path

tags_df = pd.read_csv(f"{weibo_data_path}/tag.txt", sep='\t',
                      header=None, names=["user_id", "tags"], dtype={"tags": str})
tags_df["tags"] = tags_df["tags"].fillna('')
tags_dict = {row["user_id"]: row["tags"].split(
    '_') if row["tags"] else [] for index, row in tags_df.iterrows()}
print("Tags dictionary created.")

if os.path.exists(f"{graph_path}/{graph_name}.pkl"):
    print("Loading graph from existing pkl...")
    with open(f"{graph_path}/{graph_name}.pkl", 'rb') as f:
        G = pickle.load(f)
    print("Graph loaded.")
else:
    print("Loading and filtering data...")
    # 1. 加载过滤用户列表
    filtered_users_df = pd.read_csv(f"{neo4j_data_path}/users.csv")
    filtered_users_set = set(filtered_users_df.iloc[:, 0])
    print("Filtered users loaded.")
    # 2. 加载原始数据
    user_df = pd.read_csv(
        f"{weibo_data_path}/第0层+第1层+第2层关系点.txt", header=None, names=["user_id"])
    forward_user_df = pd.read_csv(
        f"{weibo_data_path}/第0层+第1层+第2层关系且转发边.txt", header=None, names=["forward_user_id"])
    edges_df = pd.read_csv(f'{weibo_data_path}/第0层+第1层+第2层关系边.txt',
                           sep='\t', header=None, names=["blogger", "follower"])
    forward_edges_df = pd.read_csv(f'{weibo_data_path}/第0层+第1层+第2层关系且转发边.txt',
                                   sep='\t', header=None, names=["blogger", "follower", "count"])
    print("Original data loaded.")

    # 3. 过滤节点和边
    filtered_edges_df = edges_df[(edges_df['blogger'].isin(filtered_users_set)) & (
        edges_df['follower'].isin(filtered_users_set))]
    filtered_forward_edges_df = forward_edges_df[(forward_edges_df['blogger'].isin(
        filtered_users_set)) & (forward_edges_df['follower'].isin(filtered_users_set))]
    print("Data filtered.")

    # 获取所有保留的用户ID
    all_filtered_user_ids = set(filtered_edges_df['blogger']) | set(filtered_edges_df['follower']) | set(
        filtered_forward_edges_df['blogger']) | set(filtered_forward_edges_df['follower'])

    # 4. 创建有向图
    G = nx.DiGraph()

    # 添加所有保留的用户到图中
    G.add_nodes_from(all_filtered_user_ids)

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
    with open(f"{graph_path}/{graph_name}.pkl", 'wb') as f:
        pickle.dump(G, f)
    print("Graph structure saved.")


def initialize_labels(graph, initial_tags):
    labels = {}
    for node in graph.nodes():
        if node in initial_tags and initial_tags[node]:
            labels[node] = initial_tags[node]
        else:
            labels[node] = ["unknown"]  # 使用 "unknown" 作为默认标签
    return labels


print("Initializing labels...")
initial_labels = initialize_labels(G, tags_dict)
print("Labels initialized.")

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

# 将initial_labels中的 "unknown" 删掉
for node in initial_labels:
    if "unknown" in initial_labels[node]:
        initial_labels[node].remove("unknown")

# 将传播后的标签与原始标签合并，并删除 "unknown" 标签
for node in final_labels:
    if node in tags_dict:
        final_labels[node] = list(
            set(final_labels[node]) | set(tags_dict[node]))
    else:
        final_labels[node] = list(set(final_labels[node]))
    if "unknown" in final_labels[node]:
        final_labels[node].remove("unknown")

for node in tags_dict:
    if node not in final_labels:
        final_labels[node] = tags_dict[node]

# 将最终标签转换为 DataFrame 以便查看
final_labels_df = pd.DataFrame(
    [(node, labels) for node, labels in final_labels.items()], columns=["user_id", "labels"])

# 保存最初的和最后的标签
result_dir = f"{result_path}/SLPA"
os.makedirs(result_dir, exist_ok=True)
initial_labels_df = pd.DataFrame(
    [(node, labels) for node, labels in initial_labels.items()], columns=["user_id", "labels"])
initial_labels_df.to_csv(
    f'{result_dir}/initial_labels.csv', index=False)
final_labels_df.to_csv(f'{result_dir}/final_labels.csv', index=False)
print("Final labels saved.")

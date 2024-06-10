import pandas as pd
import random
from tqdm import tqdm


# 读取有标签的用户
with open(r'D:\social_network\data\weibo_user_data\tag.txt', 'r', encoding='utf-8') as f:
    tagged_users = {}
    for line in f:
        parts = line.strip().split('\t')
        user_id = parts[0]
        tags = parts[1] if len(parts) > 1 else ""
        tagged_users[user_id] = tags


# 读取关系边文件，包含关注边和关注且转发边
edges = set()
with open(r'D:\social_network\data\weibo_user_data\第0层+第1层+第2层关系边.txt', 'r') as f:
    for line in f:
        node1, node2 = line.strip().split()
        edges.add((node1, node2))

with open(r'D:\social_network\data\weibo_user_data\第0层+第1层+第2层关系且转发边.txt', 'r') as f:
    for line in f:
        node1, node2, count = line.strip().split()
        edges.add((node1, node2))

# 统计有标签用户的被关注次数
node_followers_count = {}
for node1, node2 in edges:
    if node1 in tagged_users:
        if node1 not in node_followers_count:
            node_followers_count[node1] = 0
        node_followers_count[node1] += 1

# 找出被关注最多的13个大V节点
top_13_nodes = sorted(node_followers_count, key=node_followers_count.get, reverse=True)[:13]

# 找出第1层和第2层节点
level_1_nodes = set()
level_2_nodes = set()

# 遍历每个大V节点，选择最多100个粉丝作为第1层节点
for top_node in tqdm(top_13_nodes):
    node_1st_level = [node2 for node1, node2 in edges if node1 == top_node]
    sampled_level_1_nodes = set(random.sample(node_1st_level, min(100, len(node_1st_level))))
    level_1_nodes.update(sampled_level_1_nodes)
    # 遍历每个第1层节点，选择最多5个粉丝作为第2层节点
    for level_1_node in tqdm(sampled_level_1_nodes):
        node_2nd_level = [node2 for node1, node2 in edges if node1 == level_1_node]
        sampled_level_2_nodes = set(random.sample(node_2nd_level, min(5, len(node_2nd_level))))
        level_2_nodes.update(sampled_level_2_nodes)

# 创建结果列表并添加LEVEL信息
results = []

for node in top_13_nodes:
    results.append((node, 0))

for node in level_1_nodes:
    results.append((node, 1))

for node in level_2_nodes:
    results.append((node, 2))

# 将结果转换为DataFrame
df = pd.DataFrame(results, columns=["USER_ID", "LEVEL"])
df['TAGS'] = df['USER_ID'].apply(lambda x: tagged_users.get(x, ""))

# 输出结果为CSV文件
df.to_csv("D:/social_network/data/weibo_user_data/users_demo.csv", index=False)

print("CSV文件已保存: users_demo.csv")
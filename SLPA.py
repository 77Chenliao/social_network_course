import collections
import os
import numpy as np
import random
random.seed(2024)
np.random.seed(2024)


class SLPA:
    def __init__(self, G, T, r, initial_labels, page_rank):
        """
        :param G: 图本身
        :param T: 迭代次数T
        :param r: 满足社区次数要求的阈值r
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._r = r
        self._initial_labels = initial_labels
        self._pagerank = page_rank

    def _calculate_similarity(self, labels1, labels2):
        """
        计算两个标签集合的 Jaccard 相似系数
        :param labels1: 第一个标签集合
        :param labels2: 第二个标签集合
        :return: 相似性（0 到 1 之间的值）
        """
        set1 = set(labels1)
        set2 = set(labels2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 0
        return intersection / union

    def execute(self):
        # 节点存储器初始化
        node_memory = [{tag: 1 for tag in self._initial_labels[node]}
                       if node in self._initial_labels else {} for node in self._G.nodes()]

        # 节点索引映射
        node_to_index = {node: index for index,
                         node in enumerate(self._G.nodes())}
        index_to_node = {index: node for node, index in node_to_index.items()}

        # 算法迭代过程
        for t in range(self._T):
            print(f"Iteration {t + 1}...")
            # 任意选择一个监听器
            order = list(np.random.permutation(self._n))
            for i in order:
                listener = index_to_node[i]
                label_list = {}
                listener_labels = node_memory[node_to_index[listener]].keys()
                # 从speaker中选择一个标签传播到listener
                for neighbor in self._G.neighbors(listener):
                    neighbor_index = node_to_index[neighbor]
                    neighbor_page_rank = self._pagerank[neighbor]
                    sum_label = sum(node_memory[neighbor_index].values())

                    # 从该neighbor中选择一个标签传播到listener
                    if sum_label > 0:
                        values = list(node_memory[neighbor_index].values())
                        exp_values = np.exp(values) # 指数化，使差距更大
                        label_probs_exp = exp_values / exp_values.sum()  # 归一化
                        selected_label = list(node_memory[neighbor_index].keys())[
                            np.random.multinomial(1, label_probs_exp).argmax()]
                        # 计算标签相似性
                        neighbor_labels = node_memory[neighbor_index].keys()
                        similarity = self._calculate_similarity(listener_labels, neighbor_labels)

                        # 标签传播，乘以 PageRank 和标签相似性
                        label_list[selected_label] = label_list.get(
                            selected_label, 0) + 1 * neighbor_page_rank * similarity + 0.1 # 0.1 是为了防止除零

                # listener选择一个最流行的标签添加到内存中
                if label_list:
                    max_v = max(label_list.values())
                    selected_label = random.choice(
                        [label for label, count in label_list.items() if count == max_v])
                    listener_index = node_to_index[listener]
                    node_memory[listener_index][selected_label] = node_memory[listener_index].get(
                        selected_label, 0) + label_list[selected_label]

        # 根据阈值threshold删除不符合条件的标签
        for memory in node_memory:
            sum_label = sum(memory.values())
            threshold_num = sum_label * self._r
            for k, v in list(memory.items()):
                if v < threshold_num:
                    del memory[k]

        communities = collections.defaultdict(list)
        # 扫描memory中的记录标签，相同标签的节点加入同一个社区中
        for index, memory in enumerate(node_memory):
            node = index_to_node[index]
            for label in memory.keys():
                communities[label].append(node)

        # 返回值是个数据字典，value以集合的形式存在
        return communities

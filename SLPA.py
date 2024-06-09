import collections
import os
import numpy as np
import random
random.seed(2024)
np.random.seed(2024)


class SLPA:
    def __init__(self, G, T, r, initial_labels=None):
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

    def execute(self):
        # 节点存储器初始化
        node_memory = [{tag: 1 for tag in self._initial_labels[node]} if node in self._initial_labels else {} for node in self._G.nodes()]

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
                # 从speaker中选择一个标签传播到listener
                for neighbor in self._G.neighbors(listener):
                    neighbor_index = node_to_index[neighbor]
                    sum_label = sum(node_memory[neighbor_index].values())
                    if sum_label>0:
                        label_probs = [
                            float(c) / sum_label for c in node_memory[neighbor_index].values()]
                        selected_label = list(node_memory[neighbor_index].keys())[
                            np.random.multinomial(1, label_probs).argmax()]
                        label_list[selected_label] = label_list.get(
                            selected_label, 0) + 1

                # listener选择一个最流行的标签添加到内存中
                if label_list:
                    max_v = max(label_list.values())
                    selected_label = random.choice(
                        [label for label, count in label_list.items() if count == max_v])
                    listener_index = node_to_index[listener]
                    node_memory[listener_index][selected_label] = node_memory[listener_index].get(
                        selected_label, 0) + 1

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

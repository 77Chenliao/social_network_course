import pandas as pd
from config import result_path
from utils import compare_labels

algo = 'SLPA'
result_dir = f"{result_path}/{algo}"


# 读取初始标签和最终标签的CSV文件
initial_labels_df = pd.read_csv(f'{result_dir}/initial_labels.csv')
final_labels_df = pd.read_csv(f'{result_dir}/final_labels.csv')

# 将标签列从字符串转换为列表
initial_labels_df['labels'] = initial_labels_df['labels'].apply(eval)
final_labels_df['labels'] = final_labels_df['labels'].apply(eval)

# 创建一个包含初始和最终标签的DataFrame
labels_comparison_df = pd.merge(
    initial_labels_df, final_labels_df, on='user_id', suffixes=('_initial', '_final'))

# 比较标签集合前后的变化
labels_comparison_result = labels_comparison_df.apply(compare_labels, axis=1)

# 保存比较结果到CSV文件，包含user_id、added_tags和removed_tags列
labels_comparison_result.to_csv(f"{result_dir}/labels_added.csv", index=False)
# 打印部分比较结果
print(labels_comparison_result.head())

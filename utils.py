import random
random.seed(2024)
import pandas as pd

def split_tags(tags_dict, val_ratio=0.5):
    propagation_tags = {}
    validation_tags = {}
    for user, tags in tags_dict.items():
        if len(tags) > 1:
            random.shuffle(tags)
            split_point = int(len(tags) * val_ratio)
            propagation_tags[user] = tags[:split_point]
            validation_tags[user] = tags[split_point:]
        else: # 如果只有一个标签，直接放到验证集
            propagation_tags[user] = []
            validation_tags[user] = tags
    return propagation_tags, validation_tags


def compare_labels(row):
    initial_set = set(row['labels_initial'])
    final_set = set(row['labels_final'])
    added_tags = final_set - initial_set
    return pd.Series({'user_id': row['user_id'], 'tags': list(added_tags)})



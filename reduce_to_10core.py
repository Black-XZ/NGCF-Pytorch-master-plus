"""
将 Books 数据集从 5-core 筛选到 10-core
保留至少有 10 条交互记录的用户和物品
"""

import os
from collections import defaultdict

DATA_DIR = r"e:\NGCF-PyTorch-master\Data\Books"
OUTPUT_DIR = r"e:\NGCF-PyTorch-master\Data\Books_10core"

# 10-core 阈值
MIN_INTERACTIONS = 10

def load_data():
    """加载 train.txt, test.txt, valid.txt"""
    train_interactions = []  # [(user_id, item_id), ...]
    test_interactions = []
    valid_interactions = []

    print("加载 train.txt...")
    with open(os.path.join(DATA_DIR, "train.txt"), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            for item_id in parts[1:]:
                train_interactions.append((user_id, int(item_id)))

    print(f"  train 交互数: {len(train_interactions)}")

    print("加载 test.txt...")
    with open(os.path.join(DATA_DIR, "test.txt"), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            for item_id in parts[1:]:
                test_interactions.append((user_id, int(item_id)))

    print(f"  test 交互数: {len(test_interactions)}")

    print("加载 valid.txt...")
    with open(os.path.join(DATA_DIR, "valid.txt"), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            for item_id in parts[1:]:
                valid_interactions.append((user_id, int(item_id)))

    print(f"  valid 交互数: {len(valid_interactions)}")

    return train_interactions, test_interactions, valid_interactions


def filter_to_kcore(train_interactions, test_interactions, valid_interactions, min_interactions=10):
    """迭代筛选直到达到 k-core"""
    print(f"\n开始 {min_interactions}-core 筛选...")

    # 合并所有交互
    all_interactions = train_interactions + test_interactions + valid_interactions

    while True:
        # 统计每个用户和物品的交互数
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)

        for u, i in all_interactions:
            user_counts[u] += 1
            item_counts[i] += 1

        # 找出不满足条件的用户和物品
        invalid_users = {u for u, c in user_counts.items() if c < min_interactions}
        invalid_items = {i for i, c in item_counts.items() if c < min_interactions}

        if not invalid_users and not invalid_items:
            break

        # 过滤掉无效交互
        all_interactions = [
            (u, i) for u, i in all_interactions
            if u not in invalid_users and i not in invalid_items
        ]

        print(f"  移除 {len(invalid_users)} 用户, {len(invalid_items)} 物品, 剩余 {len(all_interactions)} 交互")

    # 重新统计
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    for u, i in all_interactions:
        user_counts[u] += 1
        item_counts[i] += 1

    print(f"\n筛选完成:")
    print(f"  用户数: {len(user_counts)}")
    print(f"  物品数: {len(item_counts)}")
    print(f"  总交互数: {len(all_interactions)}")

    return all_interactions


def remap_and_save(all_interactions, test_interactions, valid_interactions):
    """重新映射ID并保存文件"""
    print("\n重新映射用户和物品ID...")

    # 获取保留的用户和物品集合
    users = set()
    items = set()
    for u, i in all_interactions:
        users.add(u)
        items.add(i)

    # 创建新ID映射
    user_to_new_id = {old: new for new, old in enumerate(sorted(users))}
    item_to_new_id = {old: new for new, old in enumerate(sorted(items))}

    print(f"  新用户ID范围: 0 - {len(user_to_new_id) - 1}")
    print(f"  新物品ID范围: 0 - {len(item_to_new_id) - 1}")

    # 读取原始映射文件，创建新的映射
    print("\n生成新的 user_list.txt 和 item_list.txt...")

    # 读取原始user_list
    old_user_mapping = {}
    with open(os.path.join(DATA_DIR, "user_list.txt"), 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                old_user_mapping[int(parts[1])] = parts[0]

    # 读取原始item_list
    old_item_mapping = {}
    with open(os.path.join(DATA_DIR, "item_list.txt"), 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                old_item_mapping[int(parts[1])] = parts[0]

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存新的 user_list.txt
    with open(os.path.join(OUTPUT_DIR, "user_list.txt"), 'w') as f:
        f.write("org_id remap_id\n")
        for old_id in sorted(user_to_new_id.keys()):
            if old_id in old_user_mapping:
                f.write(f"{old_user_mapping[old_id]} {user_to_new_id[old_id]}\n")

    # 保存新的 item_list.txt
    with open(os.path.join(OUTPUT_DIR, "item_list.txt"), 'w') as f:
        f.write("org_id remap_id\n")
        for old_id in sorted(item_to_new_id.keys()):
            if old_id in old_item_mapping:
                f.write(f"{old_item_mapping[old_id]} {item_to_new_id[old_id]}\n")

    # 重新划分数据并保存
    print("\n重新划分 train/test/valid...")

    # 分离出 train, test, valid 交互
    train_set = set(train_interactions)
    test_set = set(test_interactions)
    valid_set = set(valid_interactions)

    # 按用户组织数据
    user_train_items = defaultdict(list)
    user_test_items = defaultdict(list)
    user_valid_items = defaultdict(list)

    for u, i in all_interactions:
        new_u = user_to_new_id[u]
        new_i = item_to_new_id[i]

        if (u, i) in train_set:
            user_train_items[new_u].append(new_i)
        if (u, i) in test_set:
            user_test_items[new_u].append(new_i)
        if (u, i) in valid_set:
            user_valid_items[new_u].append(new_i)

    # 保存 train.txt
    with open(os.path.join(OUTPUT_DIR, "train.txt"), 'w') as f:
        for user_id in sorted(user_train_items.keys()):
            items = user_train_items[user_id]
            line = str(user_id) + " " + " ".join(map(str, items)) + "\n"
            f.write(line)

    # 保存 test.txt
    with open(os.path.join(OUTPUT_DIR, "test.txt"), 'w') as f:
        for user_id in sorted(user_test_items.keys()):
            items = user_test_items[user_id]
            line = str(user_id) + " " + " ".join(map(str, items)) + "\n"
            f.write(line)

    # 保存 valid.txt
    with open(os.path.join(OUTPUT_DIR, "valid.txt"), 'w') as f:
        for user_id in sorted(user_valid_items.keys()):
            items = user_valid_items[user_id]
            line = str(user_id) + " " + " ".join(map(str, items)) + "\n"
            f.write(line)

    print(f"\n完成! 输出目录: {OUTPUT_DIR}")
    print(f"  train.txt: {len(user_train_items)} 用户")
    print(f"  test.txt: {len(user_test_items)} 用户")
    print(f"  valid.txt: {len(user_valid_items)} 用户")


if __name__ == "__main__":
    # 加载数据
    train_interactions, test_interactions, valid_interactions = load_data()

    # 筛选到 10-core
    all_interactions = filter_to_kcore(
        train_interactions,
        test_interactions,
        valid_interactions,
        MIN_INTERACTIONS
    )

    # 重新映射ID并保存
    remap_and_save(all_interactions, test_interactions, valid_interactions)

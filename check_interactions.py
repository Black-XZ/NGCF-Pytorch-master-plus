"""
检查 Books_10core.jsonl 中的交互是否都在 train/valid/test 中出现
"""

import json
import os
from collections import defaultdict

# 配置路径
JSONL_FILE = r"E:\NGCF-PyTorch-master\Books_10core.jsonl"
DATA_DIR = r"E:\NGCF-PyTorch-master\Data\Books_10core"

print("加载 train, valid, test 交互数据...")

# 加载 train 中的交互
train_interactions = set()
with open(os.path.join(DATA_DIR, "train.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            uid = int(parts[0])
            items = [int(i) for i in parts[1:] if i.isdigit()]
            for item in items:
                train_interactions.add((uid, item))

print(f"  train 交互数: {len(train_interactions):,}")

# 加载 valid 中的交互
valid_interactions = set()
with open(os.path.join(DATA_DIR, "valid.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            uid = int(parts[0])
            items = [int(i) for i in parts[1:] if i.isdigit()]
            for item in items:
                valid_interactions.add((uid, item))

print(f"  valid 交互数: {len(valid_interactions):,}")

# 加载 test 中的交互
test_interactions = set()
with open(os.path.join(DATA_DIR, "test.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            uid = int(parts[0])
            items = [int(i) for i in parts[1:] if i.isdigit()]
            for item in items:
                test_interactions.add((uid, item))

print(f"  test 交互数: {len(test_interactions):,}")

# 合并所有交互
all_interactions = train_interactions | valid_interactions | test_interactions
print(f"  总交互数: {len(all_interactions):,}")

# 检查 jsonl 文件中的每条交互
print("\n检查 Books_10core.jsonl 中的交互...")

missing_interactions = []
total_count = 0

with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        total_count += 1

        if total_count % 500000 == 0:
            print(f"  已检查 {total_count:,} 条...")

        try:
            data = json.loads(line.strip())
            uid = data.get('user_id')
            item = data.get('item_id')

            if (uid, item) not in all_interactions:
                missing_interactions.append((uid, item))

        except:
            continue

print(f"\n检查完成!")
print(f"  jsonl 总记录数: {total_count:,}")
print(f"  不在 train/valid/test 中的记录数: {len(missing_interactions):,}")

if missing_interactions:
    print(f"\n前10条缺失的交互:")
    for uid, item in missing_interactions[:10]:
        print(f"    user_id={uid}, item_id={item}")
else:
    print("\n✓ 所有交互都存在于 train/valid/test 中!")

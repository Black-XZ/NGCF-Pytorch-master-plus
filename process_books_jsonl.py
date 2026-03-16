"""
处理 Books.jsonl 数据集：
1. 加载 10core 的 user 和 item 映射
2. 过滤掉无法映射的记录
3. 删除 images, verified_purchase, asin 列
4. 输出映射后的 user_id, item_id, rating, title, text
"""

import json
import os

# 配置路径
JSONL_FILE = r"e:\NGCF-PyTorch-master\Books.jsonl"
USER_LIST_FILE = r"e:\NGCF-PyTorch-master\Data\Books_10core\user_list.txt"
ITEM_LIST_FILE = r"e:\NGCF-PyTorch-master\Data\Books_10core\item_list.txt"
OUTPUT_FILE = r"e:\NGCF-PyTorch-master\Books_10core.jsonl"

print("加载用户映射...")

# 加载 user 映射: org_id -> remap_id
user_to_remap = {}
with open(USER_LIST_FILE, 'r', encoding='utf-8') as f:
    next(f)  # 跳过表头
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            user_to_remap[parts[0]] = int(parts[1])

print(f"  用户映射数: {len(user_to_remap)}")

print("加载物品映射...")

# 加载 item 映射: org_id -> remap_id
item_to_remap = {}
with open(ITEM_LIST_FILE, 'r', encoding='utf-8') as f:
    next(f)  # 跳过表头
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            item_to_remap[parts[0]] = int(parts[1])

print(f"  物品映射数: {len(item_to_remap)}")

# 处理 jsonl 文件
print(f"\n开始处理 {JSONL_FILE}...")
print("（由于文件较大，处理过程可能需要较长时间）")

total_lines = 0
kept_lines = 0
skipped_user = 0
skipped_item = 0

# 使用迭代器逐行处理，避免内存溢出
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# 先删除已存在的输出文件
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# 按批次写入，每 100000 行刷新一次
batch_size = 100000
batch_data = []

with open(JSONL_FILE, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        total_lines += 1

        if total_lines % 500000 == 0:
            print(f"  已处理 {total_lines:,} 行...")

        try:
            data = json.loads(line.strip())

            # 获取原始 ID
            original_user_id = data.get('user_id', '')
            original_item_id = data.get('parent_asin', '')

            # 检查是否在映射中
            if original_user_id not in user_to_remap:
                skipped_user += 1
                continue

            if original_item_id not in item_to_remap:
                skipped_item += 1
                continue

            # 获取映射后的 ID
            mapped_user_id = user_to_remap[original_user_id]
            mapped_item_id = item_to_remap[original_item_id]

            # 构建新记录，只保留需要的字段
            new_record = {
                'user_id': mapped_user_id,
                'item_id': mapped_item_id,
                'rating': data.get('rating'),
                'title': data.get('title', ''),
                'text': data.get('text', '')
            }

            batch_data.append(json.dumps(new_record, ensure_ascii=False))
            kept_lines += 1

            # 批量写入
            if len(batch_data) >= batch_size:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                    f_out.write('\n'.join(batch_data) + '\n')
                batch_data = []

        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"  错误: {e}")
            continue

# 处理剩余的数据
if batch_data:
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        f_out.write('\n'.join(batch_data) + '\n')

print(f"\n处理完成!")
print(f"  总行数: {total_lines:,}")
print(f"  保留行数: {kept_lines:,}")
print(f"  跳过（用户不在10core）: {skipped_user:,}")
print(f"  跳过（物品不在10core）: {skipped_item:,}")
print(f"\n输出文件: {OUTPUT_FILE}")

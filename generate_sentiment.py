"""
使用 VADER 对评论进行情感分析，生成情感分数文件
情感分数范围: [-1, 1]
- 1: 极端正面
- 0: 中性
- -1: 极端负面
"""
import json
import os
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def generate_sentiment(input_file, output_file, text_field='text'):
    """
    使用 VADER 对每条评论进行情感分析
    
    Args:
        input_file: 输入的 jsonl 文件路径
        output_file: 输出的情感分数文件路径
        text_field: 评论文本字段名
    """
    # 初始化 VADER 分析器
    analyzer = SentimentIntensityAnalyzer()
    
    # 禁用 HTML 标签解析（VADER 默认会尝试解析 HTML）
    # 我们已经在预处理时移除了 HTML 标签
    
    count = 0
    error_count = 0
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Analyzing sentiment"):
            try:
                data = json.loads(line.strip())
                
                user_id = data.get('user_id')
                item_id = data.get('item_id')
                text = data.get(text_field, '')
                
                if user_id is None or item_id is None:
                    continue
                
                # 如果文本为空，使用默认情感分数 0
                if not text or not text.strip():
                    sentiment = 0.0
                else:
                    # VADER 返回的 compound 分数范围是 [-1, 1]
                    scores = analyzer.polarity_scores(text)
                    sentiment = scores['compound']
                
                # 写入输出文件
                output_data = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'sentiment': round(sentiment, 4)
                }
                fout.write(json.dumps(output_data) + '\n')
                
                count += 1
                
            except Exception as e:
                error_count += 1
                continue
    
    print(f"\n完成！")
    print(f"处理成功: {count:,} 条")
    print(f"处理失败: {error_count:,} 条")
    print(f"输出文件: {output_file}")
    
    # 打印情感分数分布统计
    print("\n正在统计情感分数分布...")
    sentiments = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                sentiments.append(data['sentiment'])
            except:
                continue
    
    if sentiments:
        import numpy as np
        sentiments = np.array(sentiments)
        print(f"情感分数统计:")
        print(f"  最小值: {sentiments.min():.4f}")
        print(f"  最大值: {sentiments.max():.4f}")
        print(f"  平均值: {sentiments.mean():.4f}")
        print(f"  中位数: {np.median(sentiments):.4f}")
        print(f"  标准差: {sentiments.std():.4f}")
        
        # 分类统计
        positive = (sentiments > 0.1).sum()
        negative = (sentiments < -0.1).sum()
        neutral = len(sentiments) - positive - negative
        print(f"\n情感分类统计:")
        print(f"  正面 (>0.1):   {positive:,} ({positive/len(sentiments)*100:.1f}%)")
        print(f"  中性 (-0.1~0.1): {neutral:,} ({neutral/len(sentiments)*100:.1f}%)")
        print(f"  负面 (<-0.1):  {negative:,} ({negative/len(sentiments)*100:.1f}%)")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用 VADER 对评论进行情感分析')
    parser.add_argument('--input', type=str, required=True, help='输入的 jsonl 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出的情感分数文件路径')
    parser.add_argument('--text_field', type=str, default='text', help='评论文本字段名')
    
    args = parser.parse_args()
    
    generate_sentiment(args.input, args.output, args.text_field)

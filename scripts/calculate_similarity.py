#!/usr/bin/env python3
"""
Calculate similarity between inference_4_parsed.json and inference_most_parsed.json
"""

import json

def calculate_similarity(list1, list2):
    """
    Calculate similarity between two lists by comparing element by element.
    Returns: (matches, total, similarity_percentage)
    """
    if len(list1) != len(list2):
        return None, None, None
    
    matches = 0
    total = len(list1)
    
    for i in range(total):
        if list1[i] == list2[i]:
            matches += 1
    
    similarity = (matches / total * 100) if total > 0 else 0
    return matches, total, similarity

def main():
    # Load data
    with open('scripts/inference_4_parsed.json', 'r') as f:
        data = json.load(f)
    
    with open('scripts/inference_most_parsed.json', 'r') as f:
        most_frequent = json.load(f)
    
    print("=" * 70)
    print("相似度计算结果")
    print("=" * 70)
    print(f"\nMost frequent 列表长度: {len(most_frequent)}")
    print(f"None 值数量: {sum(1 for x in most_frequent if x is None)}")
    
    results = {}
    
    # Calculate similarity for each operation type and percentage
    for op_type in ['ins', 'del']:
        results[op_type] = {}
        print(f"\n{op_type.upper()}:")
        print("-" * 70)
        
        for pct in ['0.25', '0.5', '0.75']:
            pred_list = data[op_type][pct]
            matches, total, similarity = calculate_similarity(pred_list, most_frequent)
            
            if matches is not None:
                results[op_type][pct] = {
                    'matches': matches,
                    'total': total,
                    'similarity': similarity
                }
                print(f"  {pct}: {matches}/{total} = {similarity:.2f}%")
            else:
                print(f"  {pct}: 长度不匹配 (pred: {len(pred_list)}, most: {len(most_frequent)})")
    
    # Summary
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    
    all_similarities = []
    for op_type in ['ins', 'del']:
        for pct in ['0.25', '0.5', '0.75']:
            if pct in results[op_type]:
                all_similarities.append(results[op_type][pct]['similarity'])
    
    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        print(f"平均相似度: {avg_similarity:.2f}%")
        print(f"最高相似度: {max(all_similarities):.2f}%")
        print(f"最低相似度: {min(all_similarities):.2f}%")
    
    # Save results to JSON
    output_path = 'scripts/similarity_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()


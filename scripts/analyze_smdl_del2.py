#!/usr/bin/env python3
"""
Parse smdl_heatmap_inference_del2.log and calculate similarity with inference_most_parsed.json
"""

import json
import sys
sys.path.insert(0, '/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts')
from parse_inference_log import parse_log_file
from calculate_similarity import calculate_similarity

def main():
    # Parse the SMDL log file
    log_path = "/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts/smdl_heatmap_inference_del2.log"
    output_path = "/Users/ian/Project/VLN/Recurrent-VLN-BERT/scripts/smdl_del2_parsed.json"
    
    print("=" * 70)
    print("解析 SMDL Deletion 0.5 日志文件")
    print("=" * 70)
    print(f"\n解析文件: {log_path}")
    result = parse_log_file(log_path)
    
    print("\n提取的预测值:")
    for op_type in ["ins", "del"]:
        if op_type in result:
            print(f"  {op_type}:")
            for pct in ["0.25", "0.5", "0.75"]:
                if pct in result[op_type]:
                    print(f"    {pct}: {len(result[op_type][pct])} predictions")
    
    print(f"\n保存结果到: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Load most frequent values
    with open('scripts/inference_most_parsed.json', 'r') as f:
        most_frequent = json.load(f)
    
    print("\n" + "=" * 70)
    print("计算相似度")
    print("=" * 70)
    print(f"\nMost frequent 列表长度: {len(most_frequent)}")
    print(f"None 值数量: {sum(1 for x in most_frequent if x is None)}")
    
    # Calculate similarity for each operation type and percentage
    results = {}
    
    for op_type in ['ins', 'del']:
        if op_type not in result:
            continue
        results[op_type] = {}
        print(f"\n{op_type.upper()}:")
        print("-" * 70)
        
        for pct in ['0.25', '0.5', '0.75']:
            if pct not in result[op_type]:
                continue
            pred_list = result[op_type][pct]
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
        if op_type in results:
            for pct in ['0.25', '0.5', '0.75']:
                if pct in results[op_type]:
                    all_similarities.append(results[op_type][pct]['similarity'])
    
    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        print(f"平均相似度: {avg_similarity:.2f}%")
        print(f"最高相似度: {max(all_similarities):.2f}%")
        print(f"最低相似度: {min(all_similarities):.2f}%")
    
    # Save results to JSON
    similarity_output_path = 'scripts/smdl_del2_similarity_results.json'
    with open(similarity_output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n相似度结果已保存到: {similarity_output_path}")
    print("Done!")

if __name__ == "__main__":
    main()


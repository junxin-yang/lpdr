import json
from typing import Dict, Any, Tuple

# 您提供的JSON数据
with open("accuracy_by_class_mutil.json", "r") as f:
    data = json.load(f)

def calculate_metrics(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """计算每个子集的Precision、Recall和F1分数"""
    results = {}
    
    for subset_name, subset_data in data.items():
        # 提取数据
        correct = subset_data["correct"]  # TP (检测正确且识别正确)
        fp_iou = subset_data["FP"]       # 定位IoU<0.5的数量
        tp_detection = subset_data["TP"]  # 定位正确的数量
        fn = subset_data["FN"]           # 漏检数量
        total_gt = subset_data["total"]  # 真实标签总数
        
        # 计算定位正确但识别错误的数量
        detection_correct_recognition_wrong = tp_detection - correct
        
        # 总FP = 定位错误 + 定位正确但识别错误
        total_fp = fp_iou + detection_correct_recognition_wrong
        
        # 总TP = 定位正确且识别正确
        total_tp = correct
        
        # 计算指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[subset_name] = {
            "Precision": round(precision * 100, 2),  # 转换为百分比
            "Recall": round(recall * 100, 2),
            "F1": round(f1 * 100, 2),
            "TP": total_tp,
            "FP": total_fp,
            "FN": fn,
            "GT_Total": total_gt
        }
    
    return results

def calculate_overall_metrics(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """计算整体指标（所有子集的加权平均）"""
    total_tp = sum(result["TP"] for result in results.values())
    total_fp = sum(result["FP"] for result in results.values())
    total_fn = sum(result["FN"] for result in results.values())
    total_gt = sum(result["GT_Total"] for result in results.values())
    
    # 计算整体指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        "Overall_Precision": round(overall_precision * 100, 2),
        "Overall_Recall": round(overall_recall * 100, 2),
        "Overall_F1": round(overall_f1 * 100, 2),
        "Total_TP": total_tp,
        "Total_FP": total_fp,
        "Total_FN": total_fn,
        "Total_GT": total_gt
    }

def calculate_metrics(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """计算每个子集的Precision、Recall、F1分数和AP"""
    results = {}
    
    for subset_name, subset_data in data.items():
        correct = subset_data["correct"]
        fp_iou = subset_data["FP"]
        tp_detection = subset_data["TP"]
        fn = subset_data["FN"]
        total_gt = subset_data["total"]
        
        detection_correct_recognition_wrong = tp_detection - correct
        total_fp = fp_iou + detection_correct_recognition_wrong
        total_tp = correct
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        ap = total_tp / (total_tp + total_fp + fn) if (total_tp + total_fp + fn) > 0 else 0
        
        results[subset_name] = {
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1": round(f1 * 100, 2),
            "AP": round(ap * 100, 2),
            "TP": total_tp,
            "FP": total_fp,
            "FN": fn,
            "GT_Total": total_gt
        }
    
    return results

def print_results(results: Dict[str, Dict[str, float]], overall: Dict[str, float]):
    print("=" * 100)
    print(f"{'Subset':<15} {'Precision(%)':<12} {'Recall(%)':<10} {'F1(%)':<10} {'AP(%)':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'GT Total':<10}")
    print("-" * 100)
    for subset_name, metrics in results.items():
        print(f"{subset_name:<15} {metrics['Precision']:<12} {metrics['Recall']:<10} {metrics['F1']:<10} {metrics['AP']:<10} "
              f"{metrics['TP']:<8} {metrics['FP']:<8} {metrics['FN']:<8} {metrics['GT_Total']:<10}")
    print("-" * 100)
    print("=" * 100)


results = calculate_metrics(data)
overall_metrics = calculate_overall_metrics(results)

# 打印结果
print_results(results, overall_metrics)
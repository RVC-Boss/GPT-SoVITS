#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用字符映射工具

用于在文本标准化后，建立原始文本到标准化文本的字符级映射
"""


def build_char_mapping(original_text, normalized_text):
    """
    通过字符对齐算法，构建原始文本到标准化文本的映射
    
    Args:
        original_text: 原始文本
        normalized_text: 标准化后的文本
    
    Returns:
        dict: {
            "orig_to_norm": list[int], 原始文本每个字符对应标准化文本的位置
            "norm_to_orig": list[int], 标准化文本每个字符对应原始文本的位置
        }
    """
    # 使用动态规划找到最优对齐
    m, n = len(original_text), len(normalized_text)
    
    # dp[i][j] 表示 original_text[:i] 和 normalized_text[:j] 的对齐代价
    # 0 = 匹配, 1 = 替换, 插入, 删除
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    
    # 记录路径
    path = [[None] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            path[i][0] = ('del', i-1, -1)
    
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            path[0][j] = ('ins', -1, j-1)
    
    # 动态规划
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            orig_char = original_text[i-1]
            norm_char = normalized_text[j-1]
            
            # 匹配代价（相同字符代价为0，不同字符代价为1）
            match_cost = 0 if orig_char == norm_char else 1
            
            # 三种操作的代价
            costs = [
                (dp[i-1][j-1] + match_cost, 'match' if match_cost == 0 else 'replace', i-1, j-1),
                (dp[i-1][j] + 1, 'del', i-1, j),
                (dp[i][j-1] + 1, 'ins', i, j-1),
            ]
            
            min_cost, op, pi, pj = min(costs, key=lambda x: x[0])
            dp[i][j] = min_cost
            path[i][j] = (op, pi, pj)
    
    # 回溯路径，构建映射
    orig_to_norm = [-1] * len(original_text)
    norm_to_orig = [-1] * len(normalized_text)
    
    i, j = m, n
    alignments = []
    
    while i > 0 or j > 0:
        if path[i][j] is None:
            break
        
        op, pi, pj = path[i][j]
        
        if op in ['match', 'replace']:
            # 原始字符i-1 对应 标准化字符j-1
            alignments.append((i-1, j-1))
            i, j = pi, pj
        elif op == 'del':
            # 原始字符i-1 被删除，映射到当前标准化位置（如果存在）
            if j > 0:
                alignments.append((i-1, j-1))
            i = pi
        elif op == 'ins':
            # 标准化字符j-1 是插入的，没有对应的原始字符
            j = pj
    
    # 根据对齐结果建立映射
    for orig_idx, norm_idx in alignments:
        if orig_idx >= 0 and orig_idx < len(original_text):
            if orig_to_norm[orig_idx] == -1:
                orig_to_norm[orig_idx] = norm_idx
        
        if norm_idx >= 0 and norm_idx < len(normalized_text):
            if norm_to_orig[norm_idx] == -1:
                norm_to_orig[norm_idx] = orig_idx
    
    return {
        "orig_to_norm": orig_to_norm,
        "norm_to_orig": norm_to_orig
    }


def test_char_mapping():
    """测试字符映射功能"""
    test_cases = [
        ("50元", "五十元"),
        ("3.5度", "三点五度"),
        ("GPT-4", "GPT minus four"),  # 也可以测试英文
    ]
    
    for orig, norm in test_cases:
        print(f"原始: '{orig}'")
        print(f"标准化: '{norm}'")
        
        mappings = build_char_mapping(orig, norm)
        orig_to_norm = mappings["orig_to_norm"]
        norm_to_orig = mappings["norm_to_orig"]
        
        print(f"原始→标准化映射:")
        for i, c in enumerate(orig):
            norm_idx = orig_to_norm[i]
            if norm_idx >= 0 and norm_idx < len(norm):
                print(f"  [{i}]'{c}' → [{norm_idx}]'{norm[norm_idx]}'")
            else:
                print(f"  [{i}]'{c}' → 无映射")
        
        print(f"标准化→原始映射:")
        for i, c in enumerate(norm):
            orig_idx = norm_to_orig[i]
            if orig_idx >= 0 and orig_idx < len(orig):
                print(f"  [{i}]'{c}' ← [{orig_idx}]'{orig[orig_idx]}'")
            else:
                print(f"  [{i}]'{c}' ← 无对应")
        
        print()


if __name__ == "__main__":
    test_char_mapping()


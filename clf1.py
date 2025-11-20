# 以整数表示辫子单词： +i 表示 σ_i， -i 表示 σ_i^{-1}
# 例如 [1,2,1] 表示 σ1 σ2 σ1
def reduce_braid(word, n):
    # word: list of ints
    # n: braid strands count
    changed = True
    w = word[:]
    while changed:
        changed = False
        # 1) 消去 a, -a 相邻
        i = 0
        while i < len(w)-1:
            if w[i] == -w[i+1]:
                del w[i:i+2]
                changed = True
                i = max(i-1, 0)
            else:
                i += 1
        if changed:
            continue
        # 2) 远距可交换关系：|a|-|b|>1 可以互换
        i = 0
        while i < len(w)-1:
            a, b = w[i], w[i+1]
            if abs(a) - abs(b) > 1 or abs(b) - abs(a) > 1:
                # 交换
                w[i], w[i+1] = b, a
                changed = True
                i = max(i-1, 0)
            else:
                i += 1
        if changed:
            continue
        # 3) 三元关系替换： σ_i σ_{i+1} σ_i <-> σ_{i+1} σ_i σ_{i+1}
        i = 0
        while i < len(w)-2:
            a, b, c = w[i], w[i+1], w[i+2]
            # 仅匹配同号的基本三元关系（正幂）；也应处理逆元和混合情况作为练习
            if a > 0 and b > 0 and c > 0 and abs(a) == abs(c) == abs(b)-1:
                # 替换为 b a b
                w[i:i+3] = [b, a, b]
                changed = True
                i = max(i-1, 0)
            elif a < 0 and b < 0 and c < 0 and abs(a) == abs(c) == abs(b)-1:
                # 逆的三元关系
                w[i:i+3] = [b, a, b]
                changed = True
                i = max(i-1, 0)
            else:
                i += 1
    return w

# 示例
print(reduce_braid([1,2,1], 3))  # 应被规约为 [2,1,2]（或等价形式）
print(reduce_braid([1,-1,2], 3))  # 消去 1,-1
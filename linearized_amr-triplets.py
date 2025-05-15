import re


def parse_amr_to_triplets(linearized_amr):
    """
    将线性化 AMR 图解析成语义三元组 (subject, relation, object)。
    """
    # 正则匹配节点和关系
    node_pattern = r"\(\s*(\w+)\s*/\s*([\w-]+)\s*\)"  # 匹配 (z0 / wear-01) 形式，支持空格
    relation_pattern = r":(\w+)\s*(?:\(\s*(\w+)\s*\)|([\w-]+))"  # 匹配 :ARG1 (z1) 或 :ARG1 value

    # 查找所有节点
    nodes = re.findall(node_pattern, linearized_amr)
    if not nodes:  # 如果没有匹配到任何节点，则跳过
        print(f"未能解析节点: {linearized_amr}")
        return []

    node_dict = {var: concept for var, concept in nodes}  # 构建变量到概念的映射

    # 查找所有关系
    relations = re.findall(relation_pattern, linearized_amr)
    if not relations:  # 如果没有关系
        print(f"未能解析关系: {linearized_amr}")

    # 提取三元组
    triplets = []
    root_node = node_dict.get(nodes[0][0], None)  # 根节点概念
    for relation, target_var, target_value in relations:
        if target_var and target_var in node_dict:  # 如果是变量
            triplets.append((root_node, relation, node_dict[target_var]))
        elif target_value:  # 如果是直接值
            triplets.append((root_node, relation, target_value))

    return triplets


# 读取 AMR 文件并解析每一行
input_file = "/home/shenxiang/sda/Mengfanrong/NAAF/data/f30k_precomp/test_linearized_AMR_caps.txt"
output_file = "/home/shenxiang/sda/Mengfanrong/NAAF/data/f30k_precomp/test_amr_triplets.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        linearized_amr = line.strip()
        if linearized_amr:  # 跳过空行
            triplets = parse_amr_to_triplets(linearized_amr)
            if triplets:  # 跳过没有解析到节点的行
                outfile.write(str(triplets) + "\n")

print("已将解析结果保存到 test_amr_triplets.txt 文件中。")
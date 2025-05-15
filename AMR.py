#Text-AMR
import hanlp
import  os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
graph_parser = hanlp.load(hanlp.pretrained.amr.AMR3_GRAPH_PRETRAIN_PARSER)
# 输入和输出文件路径
input_file_path = '/home/CEFG/coco_precomp/testall_caps.txt'
output_file_path = '/home/CEFG/AMR_graph_testall_coco.txt'
with open(input_file_path, 'r') as file, open(output_file_path, 'w') as output_file:
    for line in file:
        line = line.strip()
        if line:
            amr_graph = graph_parser(line)
            output_file.write(str(amr_graph) + '\n\n')
#AMR-Text
import hanlp

generation = hanlp.load(hanlp.pretrained.amr2text.AMR3_GRAPH_PRETRAIN_GENERATION)

input_file_path = '/home/CEFG/AMR_graph_testall_coco.txt'
output_file_path = '/home/CEFG/AMR_graph2text_testall_coco.txt'

with open(input_file_path, 'r') as file, open(output_file_path, 'w') as output_file:
    amr_graphs = file.read().strip().split('\n\n')
    for amr_graph in amr_graphs:
        amr_graph = amr_graph.strip()
        if amr_graph:
            generated_text = generation(amr_graph)
            output_file.write(generated_text + '\n')


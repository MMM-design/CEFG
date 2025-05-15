from amr.IO import read_raw_amr_data
import logging
import torch

logging.getLogger("penman").setLevel(logging.WARNING)

class AMRParser:
    def __init__(self, fname, sentences_fname=None,linearized_AMR_path=None, dereify=True, remove_wiki=False, use_pointer_tokens=False, drop_parentheses=False):
        if fname.endswith(".pt"):
            self.sent_to_amr = torch.load(fname)
            return

        data, sentences = read_raw_amr_data(
            amr_path=fname,
            sentences_path=sentences_fname,
            linearized_AMR_path=linearized_AMR_path,  # 不写入输出文件
            dereify=dereify,
            remove_wiki=remove_wiki,
            use_pointer_tokens=use_pointer_tokens,
            drop_parentheses=drop_parentheses
        )
        # 构建句子到AMR的映射
        sent_to_amr = {s: g for g, s in zip(data, sentences)}
        self.sent_to_amr = sent_to_amr

    def parse(self, sent):
        if sent not in self.sent_to_amr:
            raise ValueError(f"Sentence '{sent}' not found in AMR data.")
        return self.sent_to_amr[sent]
amr_path = '/home/shenxiang/sda/Mengfanrong/ESL/AMR_graph_test.txt'
sentences_path = '/home/shenxiang/sda/zhuliqi/NAAF/data/f30k_precomp/test_caps.txt'
linearized_AMR_path='/home/shenxiang/sda/Mengfanrong/ESL/test_output_path.txt'
amr_parser = AMRParser(amr_path, sentences_path,linearized_AMR_path)

# 测试输出
sample_sentence = "A man wears an orange hat and glasses ."  # 替换为实际句子
try:
    amr_graph = amr_parser.parse(sample_sentence)
    print("AMR Graph for the sentence:", amr_graph)
except ValueError as e:
    print(e)
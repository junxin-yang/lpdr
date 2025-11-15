import os

full_path = "ocr/crnn/data/ccpd_blur/0359-5_21-151&285_417&398-417&398_179&377_151&285_389&306-0_0_4_33_32_25_12-59-4.jpg"

basename = os.path.basename(full_path)  # 获取文件名
print(basename)
# 从完整目录路径中提取ccpd_blur之后的部分
parent_dir = os.path.basename(os.path.dirname(full_path))  # 获取ccpd_blur
print(parent_dir)
result = os.path.join(parent_dir, basename)
print(result)
# 输出：ccpd_blur/0359-5_21-151&285_417&398-417&398_179&377_151&285_389&306-0_0_4_33_32_25_12-59-4.jpg
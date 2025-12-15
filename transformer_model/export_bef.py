import zipfile
import os

# 定义符合官方学术引用格式的RIS内容
ris_content = """
TY  - RPRT
AU  - 国家卫生健康委员会
TI  - 2022年我国卫生健康事业发展统计公报
CY  - 北京
PB  - 国家卫生健康委员会
PY  - 2023
UR  - http://www.nhc.gov.cn/
KW  - Public Health
KW  - Statistics
ER  -

TY  - BOOK
AU  - 周志华
TI  - 机器学习
CY  - 北京
PB  - 清华大学出版社
PY  - 2016
SP  - 371
EP  - 380
KW  - Machine Learning
ER  -

TY  - RPRT
AU  - 中国保险行业协会
TI  - 中国商业健康保险发展报告
CY  - 北京
PB  - 中国财政经济出版社
PY  - 2020
KW  - Insurance
ER  -

TY  - CONF
AU  - Vaswani, Ashish
AU  - Shazeer, Noam
AU  - Parmar, Niki
AU  - Uszkoreit, Jakob
AU  - Jones, Llion
AU  - Gomez, Aidan N
AU  - Kaiser, Łukasz
AU  - Polosukhin, Illia
TI  - Attention Is All You Need
JO  - Advances in Neural Information Processing Systems (NeurIPS)
VL  - 30
SP  - 5998
EP  - 6008
PY  - 2017
KW  - Transformer
KW  - Deep Learning
ER  -

TY  - JOUR
AU  - Huang, Xin
AU  - Khetan, Ashish
AU  - Cvitkovic, Milan
AU  - Karnin, Zohar
TI  - TabTransformer: Tabular Data Modeling Using Contextual Embeddings
JO  - arXiv preprint arXiv:2012.06678
PY  - 2020
UR  - https://arxiv.org/abs/2012.06678
KW  - Tabular Data
KW  - Transformer
ER  -

TY  - JOUR
AU  - Pedregosa, Fabian
AU  - Varoquaux, Gaël
AU  - Gramfort, Alexandre
AU  - Michel, Vincent
AU  - Thirion, Bertrand
AU  - Grisel, Olivier
AU  - Blondel, Mathieu
AU  - Prettenhofer, Peter
AU  - Weiss, Ron
AU  - Dubourg, Vincent
TI  - Scikit-learn: Machine Learning in Python
JO  - Journal of Machine Learning Research
VL  - 12
SP  - 2825
EP  - 2830
PY  - 2011
UR  - https://scikit-learn.org/stable/about.html#citing-scikit-learn
KW  - Python
KW  - Machine Learning Library
ER  -

TY  - JOUR
AU  - Fawcett, Tom
TI  - An Introduction to ROC Analysis
JO  - Pattern Recognition Letters
VL  - 27
IS  - 8
SP  - 861
EP  - 874
PY  - 2006
DO  - 10.1016/j.patrec.2005.10.010
KW  - ROC Analysis
KW  - Evaluation Metrics
ER  -

TY  - RPRT
AU  - World Health Organization
TI  - Global Report on Diabetes
CY  - Geneva
PB  - WHO Press
PY  - 2016
UR  - https://www.who.int/publications/i/item/9789241565257
KW  - Diabetes
KW  - Global Health
ER  -
"""

# 文件名配置
ris_filename = "official_references.ris"
zip_filename = "official_references.zip"

# 写入RIS文件
with open(ris_filename, "w", encoding="utf-8") as f:
    f.write(ris_content)

# 压缩为ZIP
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipf.write(ris_filename)

# 打印提示
print(f"成功生成: {zip_filename}")
print("包含：Scikit-learn 官方引用, NeurIPS 原始论文, WHO 官方报告等。")
# CMRC2018 数据归属

来源：Yiming Cui, Ting Liu, Wanxiang Che, Li Xiao, Zhipeng Chen, Wentao Ma, Shijin Wang, Guoping Hu. **A Span-Extraction Dataset for Chinese Machine Reading Comprehension**, EMNLP-IJCNLP 2019, pp.5886–5891. [论文](https://aclanthology.org/D19-1600/) · [官方仓库](https://github.com/ymcui/cmrc2018)。

版本：c0eb1b6ba219847457e6af3180da722bbeb656af，squad-style-data/cmrc2018_dev.json。采用CC BY-SA4.0，完整上游许可证原文在SOURCE_LICENSE.txt。此子集也按CC BY-SA4.0提供，不适用本仓库代码MIT许可。

本项目改动：确定性抽取96条、每篇文章最多一题、原文与答案字符串保留、增加本地元数据，并在抽样前执行参考答案offset核验、输入token长度限制、近重复排除。它不是CMRC官方隐藏test或榜单成绩。许可不提供担保；不暗示原作者认可本项目。

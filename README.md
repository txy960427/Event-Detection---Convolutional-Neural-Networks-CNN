# Event-Detection---Convolutional-Neural-Networks
Event Detection and Domain Adaptation with Convolutional Neural Networks论文复现

------2019.6.5更新--------

加入test部分，引入ranking loss，增加最终处理好的dev整个样本。

跑的结果dev大概67%，测试62.5%，和论文仍有差距，目前不知道如何调试。

-----------------------

把预训练的放在这个路径下，应该可以直接运行。
数据train.txt的格式为第一行为句子分词，第二行为每个单词的trigger标签，第三行为每个单词的entity type标签（只给了一个句子的格式）

只做了在dev上，dev上原则选取效果最好的，然后保存模型，测试test（未做），与论文f1大概差距5%。
思考问题可能存在数据划分上（给的别的论文作者的划分，与原论文划分有出入！），其次两个以上的trigger词没有进行特别处理。


顺便吐槽一句 真的好坑

# CHIP 2019 TASK2

* A task with respect to Natural Language Inference
* Just to be familiar with the code style of keras

---
## Updated on 2019.11.4
代码目录：
* build_db.py: 把腾讯词向量转换成数据库形式方便查找
* select_new_train.py: 从验证集中选择新的训练集
* word_level.py: ESIM、ESIM-add-category
* wordpiece_level: BERT、BERT-add-category 
* wordpiece_enhanced.py: BERT-enhanced-ESIM
* siamese_wordpiece_enhanced.py: 和BERT-enhanced-ESIM区别在采用和naive-ESIM中一样的孪生输入
* domain_supervised.py: category info不采用在输入端添加的方式，而作为一个子任务去监督
* vote_ensemble.py： 投票集成
* stacking_ensemble.py： stacking集成

stacking结果就不放了，单模验证集结果如下：
* ESIM: 0.8006
* BERT: 0.8726
* BERT-add-category: 0.8734
* BERT-enhanced-ESIM: 0.8743
* siamese-BERT-enhanced-ESIM: 0.86xx
* BERT-domain-supervised: 0.8711
* BERT-enhanced-ESIM-domain-supervised: 0.8768

不足之处：
* 一直在想怎么改进网络结构而不是想怎么把每一个模型的结果给调好
* 开始选择的fold数目不合理也导致了后面模型融合阶段收益没有预期高

排名A2B4，很是尴尬，不过第三世界研究室还是要 up up！


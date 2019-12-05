#### 项目背景
这是十一贝科技智能语音团队训练调整了整整一年后落地的项目，效果能和科大讯飞媲美，不如讯飞的点是识别结果中没有标点符号，在特定领域准确率极高。

国内中文语音识别相关的有用资料很少，技术相对封闭，搜索引擎中能找到的方法基本上都是10年前的传统方法，早已过时。

在这个过程中踩了很多坑，请教了很多人，很多个夜晚睡不着觉，想尽一切办法积累数据，有很多次会想放弃，但是感谢CTO，源源不断地给我们团队提供做下去的信念，感谢十一贝公司的包容和支持，给了我们团队持续专注地做一年的时间，让我们最后能呈现一个效果还不错的结果。

很赞同季逸超的观点，互联网领域的idea不值钱，实现也不值钱，值钱的是“**经过沉淀的idea + 反复推敲地执行**”

相关博客地址：https://blog.csdn.net/qq_30262201/category_9398117.html

----

# 和百度deepspeech的不同点
## 1.	框架选择

背景：2019年3月12号十一贝科技公司智能语音组接受了公司新采购的GPU机器一台，由于新机器适配的驱动版本太高（2019年2月发布），当前语音转写模型使用的深度学习框架theano偏学术研究，theano的开发团队在17年就加入了google，已经停止维护，theano不支持分布式，相比之下tensorflow框架更偏工程，已经是主流框架，支持分布式，支持新硬件，我们有必要对转写工程做框架调整。

百度模型框架：theano_0.8.2、keras_1.1.0

十一贝模型框架：tensorflow_1.13.1、keras_2.2.4

分析：根据调研资料显示，tensorflow新版本相比theano可以带来性能上一倍的提升，同时需要更大的内存。
 
## 2.	声学模型结构
在模型结构上主要做了6项调整，分析了每个调整项带来的影响：

|调整项	| 百度模型	| 十一贝模型	| 准确率 | 	性能 | 	资源占用|
|----|----|----|----|----|-----|
|网络结构|	1D_CNN+3*GRU|	1_DCNN+3*BiGRU	|有提升|	降低近一倍|	更大的内存|
|损失函数	|warp-ctc（baidu出品）	|tensorflow-ctc（google出品）	|不确定|	降低一点	|不确定|
|输出节点数|	27|	4563|	有提升	|提升一点|	增加|
|语音帧长|	20ms	|25ms	|有一点提升|	提升一点|	更小的内存|

> 参考论文：http://proceedings.mlr.press/v48/amodei16.pdf

> 论文博客：https://blog.csdn.net/qq_30262201/article/details/102654708

## 3.其他调整项：

（1）卷积层输出处理：忽略卷积层的前两位输出，因为它们通常无意义，且会影响模型最后的输出；

（2）BN层处理：最后一次训练冻结BN层，传入加载模型（纯开源数据训练的）的移动均值和方差。
调整后准确率平均提升2个百分点

## 4.	增加beam search和n-gram组合解码模块

- 百度模型是贪婪搜索解码
- 十一贝模型的解码模块使用现在GitHub 上比较热门的mozilla基金会实现的beam search解码模型，n-gram的作用就是进一步纠错，在权威性、准确率和性能方面都比之前deepspeech好很多，调整后准确率平均提升6个百分点以上。

#### 关于解码
> 为了在解码过程中整合语言模型信息，Graves＆Jaitly（2014）使用其经过CTC训练的神经网络对由基于HMM的最新系统生成的晶格或n最佳假设列表进行评分。 这引入了潜在的混淆因素，因为n最佳列表会很大程度上限制可能的转录集。 另外，它导致整个系统仍然依靠HMM语音识别基础结构来获得最终结果。 相反，我们提出的首遍解码结果使用神经网络和语言模型从头开始解码，**而不是对现有假设进行重新排序**。

> 以上来自论文：https://arxiv.org/pdf/1408.2873.pdf

> 相关博客：https://blog.csdn.net/qq_30262201/article/details/102653937


# deepspeech 环境搭建

新建虚拟环境：conda create -n tensorflow python=3.6

激活虚拟环境：source activate tensorflow

1.安装tensorflow：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.13.1

2.安装keras：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.2.4

3.安装语音流处理模块：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple soundfile==0.10.2

训练环境安装前三个就可以，测试环境需要后面两个

4.安装beam search解码模块（解码模块使用mozilla项目里面的）：pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp36-cp36m-manylinux1_x86_64.whl

报错platform不支持的话在mozilla的DeepSpeech里面执行进行安装：pip install $(python util/taskcluster.py --decoder)

gpu版：pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp35-cp35m-manylinux1_x86_64.whl

5.读字节流：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pydub



# 模型部署


./speech_model里面放入训练好的pb模型，和训练集的std、mean数据

./LM_model里面放入训练好的n-gram语言模型

入口voice_to_text.py

# 测试结果
100条数据堂电话语音数据上平均字错率0.02，句错率0.06

详细见./test_result/recongnnize_result.txt

# 说明
这里只开源训练和部署代码，不开源数据和模型，如果有帮助到你，烦请点个star。

master分支是部署代码

train分支是训练代码

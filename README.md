本项目基于百度飞桨实现，感谢百度 AI Studio 的支持。
百度 AI 安全对抗赛初赛第一名，决赛第五名。决赛分数92.63601，avg_mse:7.25582。

### 安装
下载代码：
```
git clone https://github.com/txyugood/baidu_AI_adversary.git
cd baidu_AI_adversary
```
下载预训练模型，下载地址：https://aistudio.baidu.com/aistudio/datasetdetail/19746
解压模型：
```
tar xvf pretrained.tar.bz2
```
建立环境：
```
conda create -n adversary python=3.7
pip install paddlepaddle-gpu==1.5.1.post97
```
运行程序：
```
conda activate adversary
python dog_ensemble_final.py
```
运行时间比较长，在百度 AI Studio 上运行大概需要10几个小时。
可以考虑在后台运行：
```
nohup python -u dog_ensemble_final.py&
```
通过以下命令查看 log：
```
tail -f nohup.out
```
运行结束后，输出文件路径为 datasets/outputs。

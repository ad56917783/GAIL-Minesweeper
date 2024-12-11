# GAIL-Minesweeper
基于生成式对抗模仿学习的扫雷游戏
## 配置环境
```
pip install torch pynput pyautogui pillow
```
## 使用方法
### 1. 采样人类玩家玩扫雷的截图和鼠标键盘记录(End键控制录制)
```
python sampler.py
```
### 2. AI通过采样人类游戏数据进行模仿学习（注意保持扫雷程序置顶）
```
python run.py
```
## 注意事项
* 扫雷程序打开后不要改动位置，需要截图后在画图中获取像素坐标传给**game_region**变量
* 游戏操作目前仅定义了鼠标的左单击、右单击和中间键(滚轮)单击
* 采样扫雷游戏一定要赢，甚至操作越快越好，这样符合专家经验的前提假设

## 存在的问题
* 采样只会存到一个游戏记录文件中，所以人类玩扫雷要连续赢
* 模型太过简单粗暴，特别是鉴别器的模型

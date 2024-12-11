from expert_sampler import *
from trainner import *
import numpy as np
from play import *

def main():  
    try:  
        # 设置游戏窗口区域 (x, y, width, height)  
        game_region = (0, 0, 163, 253)  
        
        # 初始化数据收集器，设置0.5秒的采样间隔  
        collector = ExpertDataCollector(game_region, sampling_interval=0.5)  
        
        print("\n操作说明:")  
        print("1. 按 'End' 键开始录制")  
        print("2. 正常进行游戏操作")  
        print("3. 再次按 'End' 键停止当前回合录制")  
        print("4. 重复以上步骤记录多个回合")  
        print("5. 按 Ctrl+C 结束程序并保存数据")  
        
        try:  
            while True:  
                time.sleep(1)  
        except KeyboardInterrupt:  
            print("\n准备保存数据并退出...")  
            if collector.recording:  
                collector._toggle_recording()  
            
            # 使用.pkl扩展名保存pickle文件  
            collector.save_demonstrations('expert_demos.pkl')  
            
            collector.cleanup()  
            
    except Exception as e:  
        print(f"发生错误: {e}")  
        if 'collector' in locals():  
            collector.cleanup()  
    
    finally:
        print(collector.get_statistics())

if __name__ == "__main__":  
    main() 
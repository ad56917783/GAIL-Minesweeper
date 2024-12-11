import pyautogui  
import numpy as np  
import time  
from pynput import mouse, keyboard  
from dataclasses import dataclass  
from typing import List, Tuple, Dict  
import cv2  
from threading import Lock, Thread  
import queue  
import pickle  

@dataclass  
class GameState:  
    """游戏状态数据类"""  
    timestamp: float  
    screen_state: np.ndarray  
    mouse_position: Tuple[int, int]  
    mouse_buttons: Dict[str, bool]  
    keyboard_state: Dict[str, bool]  

class ExpertDataCollector:  
    def __init__(self, game_window_region, sampling_interval=0.5):  
        """  
        参数:  
            game_window_region: (x, y, width, height) 扫雷游戏窗口的位置和大小  
            sampling_interval: 采样间隔，单位秒  
        """  
        self.game_region = game_window_region  
        self.sampling_interval = sampling_interval  
        self.recording = False  
        self.lock = Lock()  
        
        # 存储所有回合的数据  
        self.demonstrations = []  
        self.current_episode = []  
        
        # 实时状态跟踪  
        self.mouse_position = (0, 0)  
        self.mouse_buttons = {'left': False, 'right': False, 'middle': False}  
        self.keyboard_state = {}  
        
        # 初始化输入设备监听器  
        self._setup_listeners()  
        
        print(f"数据收集器已初始化。按 'End' 键开始/停止录制。")  
        print(f"采样间隔: {sampling_interval} 秒")  
        print("当前状态：未录制")  

    def _setup_listeners(self):  
        """设置输入设备监听器"""  
        self.keyboard_listener = keyboard.Listener(  
            on_press=self._on_key_press,  
            on_release=self._on_key_release  
        )  
        self.keyboard_listener.start()  
        
        self.mouse_listener = mouse.Listener(  
            on_move=self._on_mouse_move,  
            on_click=self._on_mouse_click  
        )  
        self.mouse_listener.start()  
    
    def _on_mouse_move(self, x, y):  
        """更新鼠标位置"""  
        self.mouse_position = (x, y)  
    
    def _on_mouse_click(self, x, y, button, pressed):  
        """更新鼠标按键状态"""  
        button_name = str(button).split('.')[-1]  
        self.mouse_buttons[button_name] = pressed  
    
    def _on_key_press(self, key):  
        """处理键盘按下事件"""  
        try:  
            key_name = key.char if hasattr(key, 'char') else str(key)  
            self.keyboard_state[key_name] = True  
            
            if key == keyboard.Key.end:  
                self._toggle_recording()  
        except AttributeError:  
            pass  
    
    def _on_key_release(self, key):  
        """处理键盘释放事件"""  
        try:  
            key_name = key.char if hasattr(key, 'char') else str(key)  
            self.keyboard_state[key_name] = False  
        except AttributeError:  
            pass  
    
    def _capture_state(self):  
        """捕获当前游戏状态"""  
        screenshot = pyautogui.screenshot(region=self.game_region)  
        screen_array = np.array(screenshot)  
        
        # 转换鼠标坐标为游戏窗口相对坐标  
        rel_x = max(0, min(self.mouse_position[0] - self.game_region[0], self.game_region[2]))  
        rel_y = max(0, min(self.mouse_position[1] - self.game_region[1], self.game_region[3]))  
        
        state = GameState(  
            timestamp=time.time(),  
            screen_state=screen_array,  
            mouse_position=(rel_x, rel_y),  
            mouse_buttons=self.mouse_buttons.copy(),  
            keyboard_state=self.keyboard_state.copy()  
        )  
        
        return state  
    
    def _recording_loop(self):  
        """记录循环，定期采样游戏状态"""  
        while self.recording:  
            try:  
                state = self._capture_state()  
                self.current_episode.append(state)  
                
                print(f"\r记录中... 当前回合帧数: {len(self.current_episode)}", end="")  
                
                time.sleep(self.sampling_interval)  
            except Exception as e:  
                print(f"\n记录过程出错: {e}")  
                break  
    
    def _toggle_recording(self):  
        """切换录制状态"""  
        with self.lock:  
            if not self.recording:  
                self.recording = True  
                self.current_episode = []  
                print("\n开始录制新回合...")  
                
                self.recording_thread = Thread(target=self._recording_loop)  
                self.recording_thread.start()  
            else:  
                self.recording = False  
                if hasattr(self, 'recording_thread'):  
                    self.recording_thread.join()  
                
                if self.current_episode:  
                    self.demonstrations.append(self.current_episode)  
                    print(f"\n回合结束，记录了 {len(self.current_episode)} 帧")  
                    print(f"当前共有 {len(self.demonstrations)} 个回合数据")  
                self.current_episode = []  
                print("录制已停止。按 'End' 键开始新回合。")  
    
    def save_demonstrations(self, filename):  
        """保存所有记录的数据"""  
        if self.recording:  
            print("请先停止录制再保存数据")  
            return  
        
        if not self.demonstrations:  
            print("没有可保存的数据")  
            return  
        
        try:  
            # 使用pickle保存数据  
            with open(filename, 'wb') as f:  
                pickle.dump(self.demonstrations, f)  
            print(f"\n数据已保存到 {filename}")  
            print(self.get_statistics())  
        except Exception as e:  
            print(f"保存数据时出错: {e}")  
    
    def get_statistics(self):  
        """获取收集的数据统计信息"""  
        if not self.demonstrations:  
            return "还没有收集到数据"  
        
        total_episodes = len(self.demonstrations)  
        total_frames = sum(len(episode) for episode in self.demonstrations)  
        avg_frames = total_frames / total_episodes  
        total_time = sum(  
            episode[-1].timestamp - episode[0].timestamp   
            for episode in self.demonstrations  
        )  
        
        stats = f"\n数据统计:\n"  
        stats += f"总回合数: {total_episodes}\n"  
        stats += f"总帧数: {total_frames}\n"  
        stats += f"平均每回合帧数: {avg_frames:.2f}\n"  
        stats += f"总记录时间: {total_time:.2f} 秒\n"  
        stats += f"平均每回合时长: {(total_time/total_episodes):.2f} 秒"  
        
        return stats  
    
    def cleanup(self):  
        """清理资源"""  
        self.recording = False  
        if hasattr(self, 'recording_thread'):  
            self.recording_thread.join()  
        self.keyboard_listener.stop()  
        self.mouse_listener.stop()  
        print("\n数据收集器已关闭")
from model import *
from processer import *
import pyautogui
import time
import numpy as np
import os

class MinesweeperAgent:  
    def __init__(self, game_region, device='cuda', model_path=None):  
        self.game_region = game_region  
        self.device = device  
        
        # 初始化网络和处理器  
        self.policy = PolicyNetwork().to(self.device)  
        self.state_processor = StateProcessor(game_region)  
        self.action_processor = ActionProcessor()  
        
        # 加载模型  
        if model_path and os.path.exists(model_path):  
            try:  
                checkpoint = torch.load(model_path, map_location=self.device)  
                self.policy.load_state_dict(checkpoint['policy_state_dict'])  
                print(f"Loaded model from {model_path}")  
            except Exception as e:  
                print(f"Error loading model: {e}")  
                print("Using untrained model")  
        else:  
            print(f"No model found at {model_path}, using untrained model")  
        
        self.policy.eval()  # 设置为评估模式  

    def get_action(self, game_state):  
        """  
        根据游戏状态选择动作  
        :param game_state: GameState对象  
        :return: (position, mouse_buttons)  
        """  
        with torch.no_grad():  
            # 处理状态  
            state = self.state_processor.process_state(game_state.screen_state)  
            state = state.unsqueeze(0).to(self.device)  # 添加批次维度  
            
            # 获取策略输出  
            positions, action_probs = self.policy(state)  
            
            # 解码动作  
            position, mouse_buttons = self.action_processor.decode_action(  
                positions[0],  # 移除批次维度  
                action_probs[0]  # 移除批次维度  
            )  
            
            return position, mouse_buttons  

    def play_game(self, max_steps=200, delay=0.5):  
        """  
        开始游戏循环  
        :param max_steps: 最大步数  
        :param delay: 每步之间的延迟（秒）  
        """  
        step = 0  
        
        while step < max_steps:  
            try:  
                # 捕获当前游戏状态  
                screenshot = pyautogui.screenshot(region=self.game_region)  
                states=[]
                
                # 处理状态  
                state = self.state_processor.process_state(screenshot)  
                if len(state.shape) == 3:  # 如果是RGB图像  
                    state = state.mean(dim=0)  # 转换为灰度图  
                # 添加通道维度  
                state = state.unsqueeze(0)  # (H,W) -> (1,H,W)  
                state = state.float() / 255.0  # 归一化  
                states.append(state)  
                states_tensor = torch.stack(states).to(self.device) 
                
                # 获取动作  
                with torch.no_grad():  
                    positions, action_probs = self.policy(states_tensor)  
                    position = positions[0].cpu().numpy()  
                    action_prob = action_probs[0].cpu().numpy()  
                    print(positions,action_probs)
                # 将归一化的位置转换回实际坐标  
                actual_x = int(position[0] * self.game_region[2])  
                actual_y = int(position[1] * self.game_region[3])  
                
                # 确定动作类型  
                action_type = np.argmax(action_prob)  
                mouse_buttons = {'left': False, 'right': False, 'middle': False}  
                if action_type == 0:  # left click  
                    mouse_buttons['left'] = True  
                elif action_type == 1:  # right click  
                    mouse_buttons['right'] = True  
                elif action_type == 2:  # middle click  
                    mouse_buttons['middle'] = True 
                elif action_type == 3:  # double click  
                    mouse_buttons['no_op'] = True  
                
                # 执行动作  
                screen_x = actual_x + self.game_region[0]  
                screen_y = actual_y + self.game_region[1]  
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)  
                
                if mouse_buttons['middle']:  
                    pyautogui.click(button='left')
                elif mouse_buttons['left']:  
                    pyautogui.click(button='left')  
                elif mouse_buttons['right']:  
                    pyautogui.click(button='right')  
                
                print(f"Step {step}: Position ({actual_x}, {actual_y}), Action {action_type}")  
                
                time.sleep(delay)  
                step += 1  
                
            except KeyboardInterrupt:  
                print("\nGame interrupted by user")  
                break  
            except Exception as e:  
                print(f"Error occurred: {e}")  
                import traceback  
                traceback.print_exc()  
                break
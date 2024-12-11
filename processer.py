import numpy as np  
import torch  
import torchvision.transforms as transforms  
from PIL import Image  

class StateProcessor:  
    def __init__(self, game_region):  
        self.game_region = game_region  
        self.target_size = (84, 84)  # 固定目标大小为84x84  
        
        # 定义图像转换流程  
        self.transform = transforms.Compose([  
            transforms.ToPILImage(),  
            transforms.Resize(self.target_size),  # 使用元组指定大小  
            transforms.ToTensor()  
        ])  
    
    def process_state(self, screen_state):  
        """  
        处理游戏屏幕状态  
        :param screen_state: 原始屏幕状态（PIL Image或numpy数组）  
        :return: 处理后的状态张量  
        """  
        # 如果输入是PIL Image，转换为numpy数组  
        if isinstance(screen_state, Image.Image):  
            screen_state = np.array(screen_state)  
        
        # 确保输入是numpy数组  
        if not isinstance(screen_state, np.ndarray):  
            raise ValueError("screen_state must be PIL Image or numpy array")  
        
        # 应用转换  
        processed_state = self.transform(screen_state)  
        
        return processed_state  

class ActionProcessor:  
    def __init__(self):  
        self.action_types = {  
            'left_click': 0,  
            'right_click': 1,  
            'middle_click': 2,  
            'no_op': 3  
        }  
    
    def encode_action(self, action):  
        """  
        将动作编码为网络可用的格式  
        :param action: Action对象，包含位置和类型  
        :return: (position_tensor, action_type_tensor)  
        """  
        #print(action)
        # 编码位置（归一化到[0,1]范围）  
        position = torch.tensor([  
            action.mouse_position[0] / 163,  # 使用游戏区域的宽度  
            action.mouse_position[1] / 253   # 使用游戏区域的高度  
        ], dtype=torch.float32)  
        
        # 编码动作类型（one-hot编码）  
        action_type = torch.zeros(len(self.action_types))  
        # 确定动作类型  
        if action.mouse_buttons['middle']:  
            action_type[self.action_types['middle_click']] = 1  
        elif action.mouse_buttons['left']:  
            action_type[self.action_types['left_click']] = 1  
        elif action.mouse_buttons['right']:  
            action_type[self.action_types['right_click']] = 1  
        else:  
            action_type[self.action_types['no_op']] = 1   
        
        return position, action_type  
    
    def decode_action(self, position_tensor, action_type_tensor):  
        """  
        将网络输出解码为实际动作  
        :param position_tensor: 位置张量  
        :param action_type_tensor: 动作类型张量  
        :return: (position_tuple, action_type_str)  
        """  
        # 解码位置（转换回实际坐标）  
        position = (  
            int(position_tensor[0].item() * 163),  # 使用游戏区域的宽度  
            int(position_tensor[1].item() * 253)   # 使用游戏区域的高度  
        )  
        
        # 解码动作类型（找到最大概率的动作）  
        action_type_idx = torch.argmax(action_type_tensor).item() 
         
        action_type = list(self.action_types.keys())[list(self.action_types.values()).index(action_type_idx)]  
        
        return position, action_type  

# 用于测试的辅助函数  
def test_processors():  
    # 创建测试数据  
    test_region = (0, 0, 163, 253)  
    test_image = np.random.randint(0, 255, (253, 163, 3), dtype=np.uint8)  
    
    # 测试StateProcessor  
    state_processor = StateProcessor(test_region)  
    processed_state = state_processor.process_state(test_image)  
    print(f"Processed state shape: {processed_state.shape}")  
    
    # 创建测试动作  
    class TestAction:  
        def __init__(self, position, action_type):  
            self.position = position  
            self.type = action_type  
    
    test_action = TestAction((80, 120), 'left_click')  
    
    # 测试ActionProcessor  
    action_processor = ActionProcessor()  
    position, action_type = action_processor.encode_action(test_action)  
    print(f"Encoded position: {position}")  
    print(f"Encoded action type: {action_type}")  
    
    # 测试解码  
    decoded_position, decoded_action_type = action_processor.decode_action(position, action_type)  
    print(f"Decoded position: {decoded_position}")  
    print(f"Decoded action type: {decoded_action_type}")  

if __name__ == "__main__":  
    test_processors()
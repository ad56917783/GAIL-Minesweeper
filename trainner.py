import torch.optim as optim  
from torch.utils.data import DataLoader, TensorDataset  
from model import *
from processer import *
import os

class GAILTrainer:  
    def __init__(self, game_region, device='cuda',model_path=None):  
        self.device = device  
        print(f"Initialized GAILTrainer with device: {device}")  
        print(f"Game region: {game_region}")  
        
        # 初始化网络  
        self.policy = PolicyNetwork().to(device)  
        self.discriminator = Discriminator().to(device)  
        
        # 初始化优化器  
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)  
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)  
        
        # 初始化状态处理器  
        self.state_processor = StateProcessor(game_region)  
        
        # 初始化动作处理器  
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

    def prepare_expert_data(self, demonstrations):  
        states = []  
        positions = []  
        action_types = []  
        
        for episode in demonstrations:  
            for action in episode:  
                # 处理状态  
                #print('screen_state',action.screen_state)
                state = self.state_processor.process_state(action.screen_state)  
                
                # 确保状态是正确的形状  
                if len(state.shape) == 3:  # 如果是RGB图像  
                    state = state.mean(dim=0)  # 转换为灰度图  
                
                # 添加通道维度  
                state = state.unsqueeze(0)  # (H,W) -> (1,H,W)  
                state = state.float() / 255.0  # 归一化  
                states.append(state)  
                
                # 处理动作（位置和类型）  
                position, action_type = self.action_processor.encode_action(action)  
                positions.append(position)  
                action_types.append(action_type)  
        
        # 转换为张量并移动到正确的设备上  
        states_tensor = torch.stack(states).to(self.device)  # (B,1,H,W)  
        positions_tensor = torch.stack(positions).to(self.device)  
        action_types_tensor = torch.stack(action_types).to(self.device)  
        
        #print(f"Final states tensor shape: {states_tensor.shape}")  
        #print(f"Positions tensor shape: {positions_tensor.shape}")  
        #print(f"Action types tensor shape: {action_types_tensor.shape}")  
        
        # 将动作信息组合为字典  
        actions_dict = {  
            'positions': positions_tensor,  
            'action_types': action_types_tensor  
        }  
        
        return states_tensor, actions_dict  

    def train_discriminator(self, expert_states, expert_actions, generated_states, generated_actions):  
        self.discriminator_optimizer.zero_grad()  
        
        batch_size = expert_states.size(0)  
        
        # 处理专家数据  
        # 将状态展平  
        expert_states_flat = expert_states.view(batch_size, -1)  # (B, 1*84*84)  
        
        # 连接动作信息  
        expert_input = torch.cat([  
            expert_states_flat,  # (B, 1*84*84)  
            expert_actions['positions'],  # (B, 2)  
            expert_actions['action_types']  # (B, 4)  
        ], dim=1)  # 最终形状: (B, 1*84*84 + 2 + 4)  
        #print('expert_input',expert_input)
        expert_scores = self.discriminator(expert_input)  
        #print('expert_scores',expert_scores)
        expert_loss = F.mse_loss(  
            expert_scores,   
            torch.ones_like(expert_scores, device=self.device)  
        )  
        
        # 处理生成数据  
        # 将状态展平  
        generated_states_flat = generated_states.view(batch_size, -1)  # (B, 1*84*84)  
        
        # 连接动作信息  
        generated_input = torch.cat([  
            generated_states_flat,  # (B, 1*84*84)  
            generated_actions['positions'],  # (B, 2)  
            generated_actions['action_types']  # (B, 4)  
        ], dim=1)  # 最终形状: (B, 1*84*84 + 2 + 4)  
        
        generated_scores = self.discriminator(generated_input)  
        #print('generated_scores',generated_scores)
        generated_loss = F.mse_loss(  
            generated_scores,   
            torch.zeros_like(generated_scores, device=self.device)  
        )  
        
        # 总损失  
        discriminator_loss = expert_loss + generated_loss  
        discriminator_loss.backward()  
        self.discriminator_optimizer.step()  
        
        return discriminator_loss.item()  

    def train_policy(self, states, rewards):  
        self.policy_optimizer.zero_grad()  
        
        # 生成动作  
        positions, action_probs = self.policy(states)  
        
        # 计算策略损失  
        # 使用奖励作为权重来更新策略  
        position_loss = -torch.mean(rewards * torch.sum(positions.log(), dim=1))  
        action_type_loss = -torch.mean(rewards * torch.sum(action_probs.log(), dim=1))  
        
        policy_loss = position_loss + action_type_loss  
        
        # 反向传播和优化  
        policy_loss.backward()  
        self.policy_optimizer.step()  
        
        return policy_loss.item()  

    def train_step(self, expert_demonstrations, batch_size=32):  
        # 准备专家数据  
        expert_states, expert_actions = self.prepare_expert_data(expert_demonstrations)  
        #print(f"Expert states shape in train_step: {expert_states.shape}")  
        
        # 生成模仿数据  
        with torch.no_grad():  
            positions, action_probs = self.policy(expert_states)  
            generated_actions = {  
                'positions': positions,  
                'action_types': action_probs  
            }  
        
        # 训练判别器  
        d_loss = self.train_discriminator(  
            expert_states,  
            expert_actions,  
            expert_states,  # 使用相同的状态  
            generated_actions  
        )  
        
        # 计算生成动作的奖励  
        with torch.no_grad():  
            # 展平状态  
            states_flat = expert_states.view(expert_states.size(0), -1)  
            generated_input = torch.cat([  
                states_flat,  
                generated_actions['positions'],  
                generated_actions['action_types']  
            ], dim=1)  
            rewards = torch.sigmoid(self.discriminator(generated_input))  
        
        # 训练策略网络  
        p_loss = self.train_policy(expert_states, rewards)  
        
        return d_loss, p_loss 

    def save_model(self, path):  
        """保存模型到指定路径"""  
        torch.save({  
            'policy_state_dict': self.policy.state_dict(),  
            'discriminator_state_dict': self.discriminator.state_dict(),  
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),  
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),  
        }, path)  
    
    def load_model(self, path):  
        """从指定路径加载模型"""  
        checkpoint = torch.load(path)  
        self.policy.load_state_dict(checkpoint['policy_state_dict'])  
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])  
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])  
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])  
from expert_sampler import *
from trainner import *
import numpy as np
from play import *
import warnings
warnings.filterwarnings("ignore")

# 选择设备  
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
print(f"Using device: {device}")  

# 初始化训练器  
game_region = (0, 0, 163, 253)  
trainer = GAILTrainer(game_region, device=device,model_path='minesweeper_model.pth')  

# 加载专家数据  
with open('expert_demos.pkl', 'rb') as f:  
    expert_demonstrations = pickle.load(f)  

# 开始训练  
for epoch in range(1000):  
    d_loss, p_loss = trainer.train_step(expert_demonstrations)  
    if epoch % 10 == 0:  
        print(f"Epoch {epoch}")  
        print(f"Discriminator Loss: {d_loss:.4f}")  
        print(f"Policy Loss: {p_loss:.4f}")  
        print("------------------------")

trainer.save_model('minesweeper_model.pth')  

# 2. 运行智能体  
agent = MinesweeperAgent(game_region, device=device, model_path='minesweeper_model.pth')  
agent.play_game()
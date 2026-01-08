"""
示例：如何使用NavGPT-2从RGB图像直接推理

这个示例展示了如何修改NavGPT-2的初始化代码，使其能够从RGB图像直接推理，
而不使用预存的图像特征。
"""

import torch
import sys
import os
from pathlib import Path

# 添加NavGPT-2路径
navgpt2_path = Path(__file__).parent / "map_nav_src"
sys.path.insert(0, str(navgpt2_path))

from r2r.agent import GMapNavAgent
from r2r.env_rgb import R2RNavBatchRGB
from utils.data import ImageFeaturesDB


class VisualEncoderWrapper:
    """视觉编码器包装器，包含visual_encoder和ln_vision"""
    
    def __init__(self, visual_encoder, ln_vision):
        self.visual_encoder = visual_encoder
        self.ln_vision = ln_vision
        self.eval()
    
    def eval(self):
        """设置为评估模式"""
        self.visual_encoder.eval()
        if self.ln_vision is not None:
            self.ln_vision.eval()
    
    def __call__(self, x):
        """前向传播：先通过visual_encoder，再通过ln_vision"""
        with torch.no_grad():
            features = self.visual_encoder(x)
            if self.ln_vision is not None:
                features = self.ln_vision(features)
            return features
    
    def parameters(self):
        """返回参数（用于获取device）"""
        return self.visual_encoder.parameters()


def create_rgb_env(args, instr_data, rank=0):
    """
    创建支持RGB图像的环境
    
    Args:
        args: 配置参数
        instr_data: 指令数据
        rank: 进程rank（用于分布式训练）
    
    Returns:
        rgb_env: RGB环境实例
        visual_encoder_wrapper: 视觉编码器包装器
    """
    # 步骤1: 临时创建一个环境以初始化agent并获取visual_encoder
    # 注意：这里仍然需要特征数据库来初始化，但稍后我们会使用RGB环境
    temp_feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    
    from r2r.env import R2RNavBatch
    temp_env = R2RNavBatch(
        temp_feat_db,
        None,
        args.connectivity_dir,
        args.candidate_file_dir,
        batch_size=1,
        angle_feat_size=args.angle_feat_size,
        seed=args.seed + rank,
        sel_data_idxs=None,
        name="temp",
    )
    
    # 步骤2: 初始化agent并加载checkpoint
    agent = GMapNavAgent(args, temp_env, rank=rank)
    if hasattr(args, "resume_file") and args.resume_file is not None:
        agent.load(args.resume_file)
    
    # 步骤3: 获取视觉编码器
    # 确保load_patch_feature=False，否则visual_encoder可能已被删除
    if not hasattr(agent.NavGPT.llm.Blip2InstructNav, 'visual_encoder'):
        raise ValueError(
            "visual_encoder not found. Please set args.load_patch_feature=False "
            "to prevent visual_encoder from being deleted."
        )
    
    visual_encoder = agent.NavGPT.llm.Blip2InstructNav.visual_encoder
    ln_vision = agent.NavGPT.llm.Blip2InstructNav.ln_vision
    
    # 步骤4: 创建视觉编码器包装器
    visual_encoder_wrapper = VisualEncoderWrapper(visual_encoder, ln_vision)
    
    # 步骤5: 创建RGB环境
    rgb_env = R2RNavBatchRGB(
        visual_encoder_wrapper,  # view_db参数（传入视觉编码器）
        instr_data,
        args.connectivity_dir,
        args.candidate_file_dir,
        batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size,
        seed=args.seed + rank,
        sel_data_idxs=None,
        name="train",
        visual_encoder=visual_encoder_wrapper,
    )
    
    return rgb_env, visual_encoder_wrapper


def main():
    """
    主函数：演示如何使用RGB环境
    """
    # 假设你已经有了args配置
    # args = ...
    
    # 确保load_patch_feature=False
    # args.load_patch_feature = False
    
    # 加载指令数据
    # instr_data = ...
    
    # 创建RGB环境
    # rgb_env, visual_encoder_wrapper = create_rgb_env(args, instr_data, rank=0)
    
    # 使用RGB环境初始化agent
    # agent = GMapNavAgent(args, rgb_env, rank=0)
    # if hasattr(args, "resume_file") and args.resume_file is not None:
    #     agent.load(args.resume_file)
    
    # 现在可以使用agent进行推理，它会自动从RGB图像编码特征
    # obs = rgb_env.reset()
    # action = agent.rollout(obs, ...)
    
    print("RGB inference setup complete. See RGB_INFERENCE_GUIDE.md for details.")


if __name__ == "__main__":
    main()


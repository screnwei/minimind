import torch
import sys
import os
import subprocess

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def model_to_dot(model, model_name="MiniMindForCausalLM"):
    """将模型结构转换为dot格式"""
    dot_content = [
        "digraph G {",
        "    // 全局设置",
        "    rankdir=TB;",
        "    node [shape=box, style=filled, fillcolor=lightblue];",
        "    edge [arrowhead=vee];",
        "",
        f'    // 根节点\n    {model_name} [shape=ellipse, fillcolor=lightgreen];'
    ]
    
    def add_module(name, module, parent=None):
        node_name = name.replace(".", "_").replace("(", "_").replace(")", "_")
        if hasattr(module, "weight"):
            shape = "x".join(str(x) for x in module.weight.shape)
            label = f"{type(module).__name__}\\n{shape}"
        else:
            label = type(module).__name__
        
        dot_content.append(f'    {node_name} [label="{label}"];')
        if parent:
            dot_content.append(f"    {parent} -> {node_name};")
        
        for child_name, child_module in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            add_module(full_child_name, child_module, node_name)
    
    # 添加所有模块
    for name, module in model.named_children():
        add_module(name, module, model_name)
    
    dot_content.append("}")
    return "\n".join(dot_content)

def print_model_architecture():
    # 创建配置
    config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        n_routed_experts = 1,
        vocab_size=6400,
        use_moe=True  # 启用MoE以查看完整架构
    )
    
    # 创建模型
    model = MiniMindForCausalLM(config)
    
    print("="*50)
    print("模型结构:")
    print("="*50)
    print(model)
    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 生成dot格式
    print("="*25)
    print("生成模型结构图...")
    dot_content = model_to_dot(model)
    
    # 保存dot文件
    dot_file = os.path.join(os.path.dirname(__file__), "model_architecture.dot")
    with open(dot_file, "w") as f:
        f.write(dot_content)
    
    # 生成图片
    png_file = os.path.join(os.path.dirname(__file__), "model_architecture.png")
    try:
        subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file], check=True)
        print(f"模型结构图已保存到: {png_file}")
    except subprocess.CalledProcessError as e:
        print(f"生成图片时出错: {e}")
    except FileNotFoundError:
        print("未找到dot工具，请先安装Graphviz")

if __name__ == "__main__":
    print_model_architecture() 
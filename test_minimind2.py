import torch
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.LMConfig import LMConfig
import unittest
import warnings

warnings.filterwarnings('ignore')

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif platform.processor() == 'arm' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

class TestMiniMind2(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.device = get_device()
        print(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.model_path = './MiniMind2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()

    def test_model_config(self):
        """测试模型配置"""
        config = self.model.config
        self.assertEqual(config.model_type, "minimind")
        self.assertTrue(hasattr(config, 'dim'))
        self.assertTrue(hasattr(config, 'n_layers'))
        self.assertTrue(hasattr(config, 'n_heads'))
        self.assertEqual(config.dim, 768)
        self.assertEqual(config.n_layers, 16)
        self.assertEqual(config.n_heads, 8)

    def test_tokenizer(self):
        """测试分词器"""
        test_text = "你好，请介绍一下自己。"
        tokens = self.tokenizer.encode(test_text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, test_text)

    def generate_response(self, prompt, max_new_tokens=200):
        """通用的生成函数"""
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始提示，只保留生成的部分
        response = response[len(chat_prompt):].strip()
        return response

    def test_model_generation(self):
        """测试模型生成"""
        prompt = "你好，请介绍一下自己。"
        response = self.generate_response(prompt)
        
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0, f"响应为空")
        self.assertFalse(any(c * 5 in response for c in "。，！？"), "检测到重复标点符号")

    def test_batch_generation(self):
        """测试批量生成"""
        prompts = [
            "你好，请介绍一下自己。",
            "什么是大语言模型？",
            "请解释一下量子计算。"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            response = self.generate_response(prompt)
            print(f"\n测试 {i}:")
            print(f"输入: {prompt}")
            print(f"输出: {response}")
            
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0, f"响应为空")
            self.assertFalse(any(c * 5 in response for c in "。，！？"), "检测到重复标点符号")

    def test_model_parameters(self):
        """测试模型参数"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型总参数量: {total_params / 1e6:.2f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
        
        self.assertTrue(total_params > 0)
        self.assertTrue(trainable_params > 0)

if __name__ == '__main__':
    unittest.main() 
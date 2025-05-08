import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 预训练数据集类，用于加载和处理预训练任务的数据
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集。
        参数：
            data_path: 数据文件路径，文件为jsonl格式，每行一个json对象，需包含'text'字段。
            tokenizer: 分词器，用于将文本转为token id。
            max_length: 最大序列长度，超出部分会被截断，不足会填充。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)  # 加载所有样本

    def load_data(self, path):
        """
        从指定路径加载数据，每行为一个json对象。
        返回：样本列表，每个元素为一个dict。
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取指定索引的样本，返回模型训练所需的输入、标签和损失掩码。
        返回：
            X: 输入token id序列（去掉最后一个token）
            Y: 标签token id序列（去掉第一个token，和X错位一位）
            loss_mask: 损失掩码，pad部分为0，其余为1
        """
        sample = self.samples[index]

        # 对文本进行分词和编码，返回pytorch tensor
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()  # 1D tensor
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # pad部分为0，其余为1

        # 构造输入和标签，分别为input_ids的[:-1]和[1:]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐标签
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    SFTDataset（Supervised Fine-Tuning 数据集）用于加载和处理SFT阶段的对话数据。
    数据格式为jsonl，每行为一个样本，包含conversations字段（对话轮次列表）。
    主要功能：
    1. 读取对话数据，构建符合ChatML格式的对话prompt。
    2. 使用分词器编码prompt，生成input_ids。
    3. 动态生成损失掩码，只对assistant回复部分计算loss。
    4. 返回模型训练所需的输入、标签和损失掩码。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化SFTDataset。
        参数：
            jsonl_path: 数据文件路径，jsonl格式，每行一个样本，包含conversations字段。
            tokenizer: 分词器对象。
            max_length: 最大序列长度，超出部分截断，不足部分填充。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)  # 加载所有样本
        # 获取assistant回复的起止特殊token的id序列
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.samples)

    def load_data(self, path):
        """
        从jsonl文件加载所有样本，每行为一个json对象。
        返回：样本列表。
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML格式的对话prompt。
        conversations: 对话轮次列表，每个元素为{'content': ...}
        偶数轮为user，奇数轮为assistant。
        返回：拼接后的对话字符串。
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """
        动态生成损失掩码：只对assistant回复部分计算loss。
        规则：在<|im_start|>assistant和<|im_end|>之间的token位置为1，其余为0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 查找assistant回复的起始位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 查找assistant回复的结束位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # assistant回复内容（不含起止token）位置mask为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
        获取指定索引的样本，返回模型训练所需的输入、标签和损失掩码。
        返回：
            X: 输入token id序列（去掉最后一个token）
            Y: 标签token id序列（去掉第一个token，和X错位一位）
            loss_mask: 损失掩码，pad部分为0，assistant回复部分为1，其余为0
        """
        sample = self.samples[index]
        # 构建对话prompt
        prompt = self._create_chat_prompt(sample['conversations'])
        # 分词并编码，长度截断/填充到max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练输入（X）、标签（Y）、损失掩码（loss_mask）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    DPODataset用于DPO（Direct Preference Optimization）训练阶段的数据集加载。
    该数据集每个样本包含一组对话（chosen）和一组被拒绝的对话（rejected），
    用于训练模型区分更优和较差的回复。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化DPODataset。
        参数：
            file_path: 数据文件路径，jsonl格式，每行一个样本，包含'chosen'和'rejected'字段。
            tokenizer: 分词器对象。
            max_length: 最大序列长度，超出部分截断，不足部分填充。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取pad token id
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 获取assistant回复的起止特殊token的id序列
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        # 加载所有样本
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的样本，返回模型训练所需的输入、标签和损失掩码。
        返回：
            x_chosen: 优选对话的输入token id序列（去掉最后一个token）
            y_chosen: 优选对话的标签token id序列（去掉第一个token，和x_chosen错位一位）
            mask_chosen: 优选对话的损失掩码
            x_rejected: 被拒绝对话的输入token id序列
            y_rejected: 被拒绝对话的标签token id序列
            mask_rejected: 被拒绝对话的损失掩码
        """
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        # 构建符合ChatML格式的对话prompt
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        # 分词并编码，长度截断/填充到max_length
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        # 获取token id序列
        chosen_input_ids = chosen_encoding['input_ids']
        rejected_input_ids = rejected_encoding['input_ids']
        # 生成损失掩码
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        # 构建训练输入（X）、标签（Y）、损失掩码（loss_mask）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,           # 优选对话输入
            'y_chosen': y_chosen,           # 优选对话标签
            'mask_chosen': mask_chosen,     # 优选对话损失掩码
            'x_rejected': x_rejected,       # 被拒绝对话输入
            'y_rejected': y_rejected,       # 被拒绝对话标签
            'mask_rejected': mask_rejected  # 被拒绝对话损失掩码
        }

    def _generate_loss_mask(self, input_ids):
        """
        动态生成损失掩码：只对assistant回复部分计算loss。
        规则：在<|im_start|>assistant和<|im_end|>之间的token位置为1，其余为0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 查找assistant回复的起始位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 查找assistant回复的结束位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # assistant回复内容（不含起止token）位置mask为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    RLAIFDataset 用于 RLHF（基于人类反馈的强化学习）或 RLAIF（基于人工智能反馈的强化学习）阶段的数据集加载。
    每个样本包含一段多轮对话，主要用于奖励模型或策略模型的训练。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化RLAIFDataset。
        参数：
            jsonl_path: 数据文件路径，jsonl格式，每行一个样本，包含conversations字段。
            tokenizer: 分词器对象。
            max_length: 最大序列长度，超出部分截断，不足部分填充。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 获取assistant回复的起止特殊token的id序列
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.samples)

    def load_data(self, path):
        """
        从jsonl文件加载所有样本，每行为一个json对象。
        返回：样本列表。
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML格式的对话prompt。
        conversations: 对话轮次列表，每个元素为{'content': ...}
        偶数轮为user，奇数轮为assistant。
        返回：
            prompt: 除最后一轮外的对话拼接成的prompt字符串
            answer: 最后一轮的回复内容
        """
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        """
        获取指定索引的样本，返回prompt和answer。
        返回：
            prompt: 除最后一轮外的对话prompt字符串（用于生成）
            answer: 最后一轮回复内容（用于奖励模型或策略模型评估）
        """
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,   # 对话历史prompt
            'answer': answer    # 当前回复内容
        }


if __name__ == "__main__":
    pass

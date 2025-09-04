import os
import json
import pandas as pd
import openai
from typing import Dict, Any, Optional

# 设置OpenAI API密钥
# 请替换为您的实际API密钥
client = openai.OpenAI(
    api_key="sk-b09c374d8bd2478fa94697ae79dad1bd",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#TODO: construct + refine  to generate rules, a advocator and refiner multi-round until satisfatory, 
#TODO: templates generate uses generated rules (and challenge des) to one tage containing zero and first belief,first stage belief contain other's  
class BeliefBuilder:
    def __init__(self, csv_path: str):
        """初始化Belief构建器

        Args:
            csv_path: CSV文件路径，包含各阶段的提示词
        """
        self._load_prompts(csv_path)
        self.belief = None
        self.history = []

    def _load_prompts(self, csv_path: str) :

        prompts = {}
        try:
            # 使用pandas读取CSV文件
            df = pd.read_csv(csv_path, encoding="utf-8")
            # 将DataFrame转换为字典
            prompts["zero-order"] = df["prompt"][0]
            prompts["first-check"] = df["prompt"][1]
            prompts["first-order"] = df["prompt"][2]
            prompts["final-check"] = df["prompt"][3]
            return prompts
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return {}

    def _call_openai_api(self, prompt: str) -> str:
        """调用OpenAI API
        Args:
            prompt: 提示词
            user_input: 用户输入，如果有的话
        Returns:
            API返回的文本响应
        """
        self.history.append({"role": "user", "content": prompt})
        messages = self.history

        try:
            response = client.chat.completions.create(
                # model="qwen3-235b-a22b-instruct-2507",
                model="qwen3-30b-a3b-thinking-2507",
                messages=messages,
                temperature=0.2,
                # 较低的温度以获得更确定性的输出
                max_tokens=2000,
            )
            output = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": output})
            return output
        except Exception as e:
            print(f"调用OpenAI API时出错: {e}")
            return ""

    def build_zero_stage_belief(self, challenge_description: str):
        """构建零阶belief

        Args:
            challenge_description: 挑战描述
            goal: 目标字典

        Returns:
            零阶belief字典
        """
        prompt = self.prompts.get("zero-order", "")
        prompt = prompt.replace("$Challenge_des$", challenge_description)

        if not prompt:
            raise ValueError("未找到zero-stage提示词")
        # 调用API
        response = self._call_openai_api(prompt)
        print(response)

    def refine_belief(self,output_file):
        """检查并精炼belief

        Args:
            belief: 待精炼的belief字典

        Returns:
            精炼后的belief字典
        """
        prompt = self.prompts.get("first-check", "")
        if not prompt:
            raise ValueError("未找到check-refine提示词")
        # 调用API
        response = self._call_openai_api(prompt)
        self.save_belief(response,output_file)
        print(response)
        # 解析返回的JSON
        # try:
        #     json_start = response.find('{')
        #     json_end = response.rfind('}')
        #     if json_start != -1 and json_end != -1:
        #         json_str = response[json_start:json_end+1]
        #         refined_belief = json.loads(json_str)
        #         self.belief = refined_belief
        #         return refined_belief
        #     else:
        #         print("未能从响应中提取JSON")
        #         print("原始响应:", response)
        #         return belief
        # except json.JSONDecodeError as e:
        #     print(f"JSON解析错误: {e}")
        #     print("原始响应:", response)
        #     return belief

    def build_one_stage_belief(self):
        """构建一阶belief

        Args:
            belief: 零阶belief字典
            partner_count: 合作伙伴数量

        Returns:
            一阶belief字典
        """
        prompt = self.prompts.get("first-order", "")
        if not prompt:
            raise ValueError("未找到one-stage提示词")
        # 调用API
        response = self._call_openai_api(prompt)
        print(response)
        # 解析返回的JSON
        # try:
        #     json_start = response.find('{')
        #     json_end = response.rfind('}')
        #     if json_start != -1 and json_end != -1:
        #         json_str = response[json_start:json_end+1]
        #         one_stage_belief = json.loads(json_str)
        #         self.belief = one_stage_belief
        #         return one_stage_belief
        #     else:
        #         print("未能从响应中提取JSON")
        #         print("原始响应:", response)
        #         return belief
        # except json.JSONDecodeError as e:
        #     print(f"JSON解析错误: {e}")
        #     print("原始响应:", response)
        #     return belief

    def final_check(self,output_path):
        """最终检查belief

        Args:
            belief: 待检查的belief字典

        Returns:
            最终检查后的belief字典
        """
        prompt = self.prompts.get("final-check", "")
        if not prompt:
            raise ValueError("未找到last-check提示词")
        # 调用API
        response = self._call_openai_api(prompt)
        self.save_belief(response,output_path)
        print(response)
        # 解析返回的JSON
        # try:
        #     json_start = response.find('{')
        #     json_end = response.rfind('}')
        #     if json_start != -1 and json_end != -1:
        #         json_str = response[json_start:json_end+1]
        #         final_belief = json.loads(json_str)
        #         self.belief = final_belief
        #         return final_belief
        #     else:
        #         print("未能从响应中提取JSON")
        #         print("原始响应:", response)
        #         return belief
        # except json.JSONDecodeError as e:
        #     print(f"JSON解析错误: {e}")
        #     print("原始响应:", response)
        #     return belief

    def build_complete_belief(self, challenge_description: str,outputfile:str):
        """构建完整的belief结构，包括所有阶段

        Args:
            challenge_description: 挑战描述
            goal: 目标字典
            partner_count: 合作伙伴数量

        Returns:
            最终的belief字典
        """
        # 1. 构建零阶belief
        print("构建零阶belief...")
        self.build_zero_stage_belief(challenge_description)

        # 2. 检查并精炼
        print("检查并精炼belief...")
        self.refine_belief(outputfile)

        # 3. 构建一阶belief
        print("构建一阶belief...")
        self.build_one_stage_belief()

        # 4. 最终检查
        print("最终检查belief...")
        self.final_check(outputfile)

    def save_belief(self, belief, output_path: str) -> None:
        """保存belief到JSON文件

        Args:
            belief: belief字典
            output_path: 输出文件路径
        """
        with open(output_path,'a+') as f:
            f.write(belief)
            f.write("\n")


# 使用示例
def main():
    # CSV文件路径
    csv_path = r"./construct_rules.csv"

    # 创建BeliefBuilder实例
    builder = BeliefBuilder(csv_path)

    challenge_description = "In this domain, two agents must collaborate to transporting as many target objects as possible to a designated bed location(unknown at beginning), using available containers given a shared goal. Each target object corresponds to a task, which can be either complete or incomplete. Agents can autonomously plan subgoals based on the overall objective.Objects, containers, and agents all have a location attribute. Initially, objects and containers are scattered across various rooms. A container can hold up to three objects, and an agent can carry up to two items at a time—these may be objects or containers. The domain consists of multiple rooms, and agents are deployed within this multi-room space, where they can freely move and explore. Each room’s exploration state is categorized as None (unexplored), Part (partially explored), or All (fully explored)."
    os.makedirs("./belief_rules",exist_ok=True)
    outputfile = "./belief_rules/rules.txt"
    with open(outputfile,"w") as f:
        f.write('')
    # 构建完整的belief
    builder.build_complete_belief(challenge_description,outputfile)
    process = builder.history
    # 保存结果
    # output_path = "d:\\Desktop\\internship\\agent\\belief_structure\\belief_construct\\final_belief.json"
    # builder.save_belief(final_belief, output_path)


if __name__ == "__main__":
    main()

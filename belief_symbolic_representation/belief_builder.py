import os
import json
import pandas as pd
import openai
from typing import Dict, Any, Optional
import re

# 设置OpenAI API密钥
# 请替换为您的实际API密钥
client = openai.OpenAI(
    api_key="sk-b09c374d8bd2478fa94697ae79dad1bd",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#Done: construct + refine  to generate rules, a advocator and refiner multi-round until satisfatory, 
#TODO: templates generate generated rules and challenge des to one tage containing zero and first belief,first stage belief contain other b 
class BeliefBuilder:
    def __init__(self, csv_path: str):
        """初始化Belief构建器

        Args:
            csv_path: CSV文件路径，包含各阶段的提示词
        """
        self.prompts = self._load_prompts(csv_path)
        self.belief = None
        self.history = []
        self.advice = None
        self.previous_content = None

    def _load_prompts(self, csv_path: str) -> Dict[str, str]:

        prompts = {}
        try:
            # 使用pandas读取CSV文件
            df = pd.read_csv(csv_path, encoding="utf-8")
            # 将DataFrame转换为字典
            prompts["init"] = df['prompt'][0]
            prompts['debate'] = df["prompt"][1]
            prompts['refine'] = df['prompt'][2]
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
        messages = [{"role": "user", "content": prompt}]

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

    def init_construction(self, challenge_description: str):
        
        prompt = self.prompts.get("init", "")
        prompt = prompt.replace("$Challenge_des$", challenge_description)

        if not prompt:
            raise ValueError("未找到zero-stage提示词")
        # 调用API
        response = self._call_openai_api(prompt)
        print(response)
        return response

    def disscussion(self,challenge_description:str,content:str):
        
        prompt = self.prompts.get("debate", "")
        if not prompt:
            raise ValueError("未找到check-refine提示词")
        # 调用API
        prompt = prompt.replace("$Challenge_des$", challenge_description)
        prompt = prompt.replace('$Alice_content$',content)
        response = self._call_openai_api(prompt)
        print(response)
        return response
        

    def refine(self,challenge_des:str,previous_content:str,advice:str):
        
        prompt = self.prompts.get("refine", "")
        prompt = prompt.replace("$Challenge_des$", challenge_des)
        prompt = prompt.replace('$previous_content$',previous_content)
        prompt = prompt.replace('$advice$',advice)
        if not prompt:
            raise ValueError("未找到one-stage提示词")
        # 调用API
        response = self._call_openai_api(prompt)
        print(response)
        return response

    
    def build_complete_belief(self, challenge_description: str,outputfile:str):
        
        
        
        for i in range(3):
            if i == 0:
                construction = self.init_construction(challenge_description)
            else:
                construction = self.refine(challenge_description,self.previous_content,self.advice)

            self.previous_content = construction

            disscussion = self.disscussion(challenge_description,construction)
            self.advice = disscussion
            match = re.search(r'satisfied:\s*([a-zA-Z]+)', disscussion, re.IGNORECASE)
            satisfied = None
            if match:
                satisfied  =  match.group(1).strip()
            print(satisfied)
            if satisfied.lower() == 'yes':
                break
        final_construction = self.previous_content
        match = re.search(r'construction:\s*(.*)', final_construction, re.DOTALL)
        if match:
            final_construction = match.group(1).strip()

        self.save_belief(final_construction,outputfile)
    def save_belief(self, belief, output_path: str) -> None:
        """保存belief到JSON文件

        Args:
            belief: belief字典
            output_path: 输出文件路径
        """
        with open(output_path,'w') as f:
            f.write(belief)
            f.write("\n")


# 使用示例
def main():
    # CSV文件路径
    csv_path = r"./construct.csv"

    # 创建BeliefBuilder实例
    builder = BeliefBuilder(csv_path)

    challenge_description = "In this domain, two agents must collaborate to transporting as many target objects as possible to a designated bed location(unknown at beginning)(totice that bed is an important entity class as destination with only attribution: location) using available containers given a shared goal(e.g. like transport 2 apples and one pen to the bed). Each target object corresponds to a task, which can be either complete or incomplete. Agents can autonomously plan subgoals based on the overall objective.Objects, containers, and agents all have a location attribute. Initially, objects and containers are scattered across various rooms. A container can hold up to three objects, and an agent can carry up to two items at a time—these may be objects or containers. The domain consists of multiple rooms, and agents are deployed within this multi-room space, where they can freely move and explore. Each room’s exploration state is categorized as None (unexplored), Part (partially explored), or All (fully explored)."
    os.makedirs("./belief_rules",exist_ok=True)
    outputfile = "./belief_rules/rules.txt"
    builder.build_complete_belief(challenge_description,outputfile)
    process = builder.history
    

if __name__ == "__main__":
    main()

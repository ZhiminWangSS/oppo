##
import random
import re
from typing import List
import json
import pandas as pd
import backoff
import torch
from tqdm import tqdm
import logging
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from openai import AzureOpenAI
from openai import OpenAIError
from openai import OpenAI
from datetime import datetime
import ast




class LLM_cobel:
    """
    大语言模型接口类
    主要功能：
    1. 支持多种大语言模型（OpenAI、DeepSeek、HuggingFace）
    2. 处理提示词模板
    3. 生成和执行规划
    4. 管理对话历史
    """

    def __init__(
        self,
        source,  # 'huggingface' or 'openai'
        lm_id,
        prompt_template_path,
        communication,
        cot,
        sampling_parameters,
        agent_id,
    ):
        """
        初始化大语言模型接口

        参数:
            source: 模型来源 ('huggingface', 'openai', 'deepseek')
            lm_id: 模型ID
            prompt_template_path: 提示词模板路径
            communication: 是否启用通信
            cot: 是否使用思维链
            sampling_parameters: 采样参数
            agent_id: 智能体ID
        """
        # 智能体基本信息
        self.rooms_explored = None  # 已探索的房间
        self.goal_desc = None  # 目标描述
        self.agent_id = agent_id  # 智能体ID
        self.agent_name = "Alice" if agent_id == 0 else "Bob"  # 智能体名称
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"  # 对手名称
        self.oppo_pronoun = "she" if agent_id == 1 else "he"  # 对手代词
        self.characters = 0
        self.tokens = 0
        self.api = 0
        # 调试和配置
        self.debug = sampling_parameters.debug  # 调试模式
        self.belief_debug = True
        self.rooms = []  # 房间列表

        # 提示词模板相关
        self.prompt_template_path = prompt_template_path
        self.single = "single" in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)

        
        #COBEL - zhimin
        rules_df = pd.read_csv("./LLM/belief_rules.csv")
        self.cobel_prompts_df = pd.read_csv("./LLM/cobel_prompts.csv")

        self.first_order_rules = rules_df['prompt'][0]
        self.zero_order_rules = rules_df['prompt'][1]
        self.prompt_template = (
            df["prompt"][0]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
        )
        if communication:
            self.generator_prompt_template = (
                df["prompt"][1]
                .replace("$AGENT_NAME$", self.agent_name)
                .replace("$OPPO_NAME$", self.oppo_name)
            )
        else:
            self.generator_prompt_template = None

        # 模型配置
        self.communication = communication  # 是否启用通信
        self.cot = cot  # 是否使用思维链
        self.source = source  # 模型来源
        self.model = None  # 模型实例
        self.tokenizer = None  # 分词器
        self.lm_id = lm_id  # 模型ID
        self.chat = (
            "gpt-3.5-turbo" in lm_id or "gpt-4" in lm_id or "deepseek" in lm_id
        )  # 是否为聊天模型
        self.OPENAI_KEY = None  # OpenAI API密钥
        self.total_cost = 0  # 总花费
        self.communication_cost = 0  # 通信花费
        # 根据不同来源初始化模型
        if self.source == "openai":
            # OpenAI模型初始化
            client = OpenAI(
                api_key="sk-57d87ae693d94216971bc2905b0a2647",
                base_url="https://api.deepseek.com",
            )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == "deepseek":
            # DeepSeek模型初始化
            client = OpenAI(
                api_key="sk-57d87ae693d94216971bc2905b0a2647",
                base_url="https://api.deepseek.com",
            )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == "hf":
            # HuggingFace模型初始化
            self.tokenizer = LlamaTokenizer.from_pretrained(self.lm_id, use_fast=True)
            self.model = LlamaForCausalLM.from_pretrained(
                self.lm_id, device_map="auto", load_in_4bit=True
            )
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                "use_cache": True,
                # 'output_scores': True,
                "return_dict_in_generate": True,
                "do_sample": True,
                # 'early_stopping': True,
            }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):

            @backoff.on_exception(backoff.expo, OpenAIError)
            def openai_generate(prompt, sampling_params):
                usage = 0
                try:
                    if self.chat:
                        response = client.chat.completions.create(
                            model=self.lm_id, messages=prompt, **sampling_params
                        )
                        self.api += 1
                        usage = response.usage.completion_tokens ## input:prompt output completion total:total
                        self.tokens += response.usage.completion_tokens
                        if self.debug:
                            with open(f"LLM/chat_raw.json", "a") as f:
                                f.write(
                                    json.dumps(
                                        response.choices[0].message.content, indent=4
                                    )
                                )
                                f.write("\n")
                        generated_samples = [
                            response.choices[i].message.content
                            for i in range(sampling_params["n"])
                        ]   
                        # if "gpt-4" or "gpt4" in self.lm_id:
                        #     usage = (
                        #         response.usage.prompt_tokens * 0.03 / 1000
                        #         + response.usage.completion_tokens * 0.06 / 1000
                        #     )
                        # elif "gpt-3.5" in self.lm_id:
                        #     usage = response.usage.total_tokens * 0.002 / 1000
                    # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                    #                   range(sampling_params['n'])]
                    elif "text-" in lm_id:
                        response = client.completions.create(
                            model=lm_id, prompt=prompt, **sampling_params
                        )
                        self.tokens += response.usage.completion_tokens
                        # print(json.dumps(response, indent=4))
                        if self.debug:
                            with open(f"LLM/raw.json", "a") as f:
                                f.write(json.dumps(response, indent=4))
                                f.write("\n")
                        generated_samples = [
                            response.choices[i].text
                            for i in range(sampling_params["n"])
                        ]
                    # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                    #               range(sampling_params['n'])]
                    else:
                        raise ValueError(f"{lm_id} not available!")
                except OpenAIError as e:
                    print(e)
                    raise e
                return generated_samples, usage

            def tokenize_dialog(dialog):
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                prompt_tokens = []
                # print(dialog)
                if dialog[0]["role"] == "system":
                    dialog = [
                        {
                            "role": dialog[1]["role"],
                            "content": B_SYS
                            + dialog[0]["content"]
                            + E_SYS
                            + dialog[1]["content"],
                        }
                    ] + dialog[2:]
                assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                    [msg["role"] == "assistant" for msg in dialog[1::2]]
                ), (
                    "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )
                dialog_tokens: List[int] = sum(
                    [
                        [self.tokenizer.bos_token_id]
                        + self.tokenizer.encode(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                            add_special_tokens=False,
                        )
                        + [self.tokenizer.eos_token_id]
                        for prompt, answer in zip(
                            dialog[::2],
                            dialog[1::2],
                        )
                    ],
                    [],
                )
                assert (
                    dialog[-1]["role"] == "user"
                ), f"Last message must be from user, got {dialog[-1]['role']}"
                dialog_tokens += [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                    add_special_tokens=False,
                )
                prompt_tokens.append(dialog_tokens)
                return torch.tensor(prompt_tokens).to("cuda")

            @torch.inference_mode()
            def hf_generate(prompt, sampling_params):
                if self.chat:
                    input_ids = tokenize_dialog(prompt)
                else:
                    input_ids = self.tokenizer(
                        prompt, return_tensors="pt"
                    ).input_ids.to("cuda")
                prompt_len = input_ids.shape[-1]
                output_dict = self.model.generate(
                    input_ids,
                    pad_token_id=self.tokenizer.eos_token_id,  # max_length=prompt_len + sampling_params['max_new_tokens'],
                    **sampling_params,
                )
                generated_samples = self.tokenizer.batch_decode(
                    output_dict.sequences[:, prompt_len:]
                )
                generated_samples = [s.strip() for s in generated_samples]
                generated_samples = [
                    s[:-4] if "</s>" in s[-4:] else s for s in generated_samples
                ]
                if self.debug:
                    print(generated_samples)
                return generated_samples, 0

            def _generate(prompt, sampling_params):
                usage = 0
                if source == "openai":
                    return openai_generate(prompt, sampling_params)
                elif self.source == "hf":
                    return hf_generate(prompt, sampling_params)

                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None

        #COBEL - zhimin 初始化信念
        self.belief_rules = self.zero_order_rules + "\n" + self.first_order_rules #整个任务不变
        # self.zero = self.init_beliefs(self.belief_rules, self.goal_desc, self.rooms)  # 初始化信念

    def reset(self, rooms_name, goal_objects):
        """
        重置模型状态

        参数:
            rooms_name: 房间名称列表
            goal_objects: 目标物体
        """
        self.rooms = rooms_name
        self.goal_desc = self.goal2description(goal_objects)
        #COBEL - zhimin
        initial_zero_beliefs, initial_first_beliefs = self.init_beliefs(self.belief_rules,self.goal_desc,self.rooms)
        
        self.tokens = 0
        self.communication_cost = 0
        self.api = 0
        self.total_cost = 0
        return initial_zero_beliefs, initial_first_beliefs
    def goal2description(self, goals):  # {predicate: count}
        """
        将目标转换为描述文本

        参数:
            goals: 目标字典 {predicate: count}

        返回:
            目标描述文本
        """
        s = "Transport "
        r = None
        for object_name, count in goals.items():
            s += f"{count} {object_name}{'s' if count > 1 else ''}, "

        s = s[:-2] + f" to the bed."
        return s

    def parse_answer(self, available_actions, text):
        """
        解析模型回答

        参数:
            available_actions: 可用动作列表
            text: 模型生成的文本

        返回:
            解析后的动作
        """
        flags = "AC"
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message:"):
                action = "send a message"
                flags = "COMMUNICATION"
            if action.lower() in text.lower():
                return available_actions[i], flags
        sents = text.split("\n")  # Split by space
        words = []
        for sent in sents:
            words.extend(sent.split(" "))
        words = list(filter(None, words))  # Remove empty strings from the result

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord("A") + i)
            # txt = text.lower()
            if (
                f"option {option}" in text
                or f"{option}." in words
                or f"{option}," in words
                or f"{option}\n" in text.split(" ")
                or f"Option {option}" in text
                or f"({option})" in words
                or f"action {option}" in text
                or (len(text) <= 2 and option in text)
            ):
                return action, flags
        print("WARNING! Fuzzy match!")
        flags = "Fuzzy match"
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith("go to"):
                # act = 'go to'
                name = action.split(" ")[-2][1:-1]
                id = action.split(" ")[-1][1:-1]
            elif action.startswith("explore"):
                act = "explore"
                name = action.split(" ")[-2][1:-1]
                id = action.split(" ")[-1][1:-1]
            elif action.startswith("go grasp"):
                act = "grasp"
                name = action.split(" ")[-2][1:-1]
                id = action.split(" ")[-1][1:-1]
            elif action.startswith("put"):
                act = "put"
            elif action.startswith("transport"):
                act = "transport"
            option = chr(ord("A") + i)
            if name in text and id in text:
                return action, flags
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith("go to"):
                # act = 'go to'
                name = action.split(" ")[-2][1:-1]
                id = action.split(" ")[-1][1:-1]
            elif action.startswith("explore"):
                act = "explore"
                name = action.split(" ")[-2][1:-1]
                id = action.split(" ")[-1][1:-1]
            elif action.startswith("go grasp"):
                act = "grasp"
                name = action.split(" ")[-2][1:-1]
                id = action.split(" ")[-1][1:-1]
            elif action.startswith("put"):
                act = "put"
            elif action.startswith("transport"):
                act = "transport"
            option = chr(ord("A") + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action, flags
        if len(text) == 1:
            i = ord(text) - ord("A")
            if i in range(len(available_actions)):
                return available_actions[i]
        print("WARNING! No available action parsed!!! Random choose one")#TODO: verify that if the parse function works
        flags = "failed to parse"
        return random.choice(available_actions), flags

    def progress2text(
        self,
        current_step,
        satisfied,
        opponent_grabbed_objects,
        opponent_last_room,
    ):
        """
        将进度转换为文本描述

        参数:
            current_step: 当前步骤
            satisfied: 已完成的物体
            opponent_grabbed_objects: 对手抓取的物体
            opponent_last_room: 对手最后所在的房间

        返回:
            进度描述文本
        """
        s = f"I've taken {current_step}/3000 steps. "

        sss = {}
        for room, obj_list in self.obj_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            s_bed = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ", ".join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ", ".join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss
            if len(obj_list[2]) > 0:
                s_bed = "the goal position bed"
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += "nothing"
            elif s_obj != "" and s_con != "" and s_bed == "":
                sr += s_obj + ", and " + s_con
            elif s_obj != "" and s_con == "" and s_bed != "":
                sr += s_obj + ", and " + s_bed
            elif s_obj == "" and s_con != "" and s_bed != "":
                sr += s_con + ", and " + s_bed
            elif s_obj != "" and s_con != "" and s_bed != "":
                sr += s_obj + ", " + s_con + ", and " + s_bed
            else:
                sr += s_obj + s_con + s_bed
            sss[room] = sr

        if len(satisfied) == 0:
            if len(self.object_list[2]) == 0:
                s += "I haven't found the goal position bed. "
            else:
                s += ""
        else:
            s += f"{'I' if self.single else 'We'}'ve already transported "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x["type"] == 0]) == 0:
                s += "nothing"
            s += ", ".join(
                [
                    f"<{x['name']}> ({x['id']})"
                    for x in unique_satisfied
                    if x["type"] == 0
                ]
            )
            s += " to the bed. "

        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj["type"] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj["type"] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj["contained"]):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = "nothing"
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = (
                    f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "
                )

        if (
            self.holding_objects[0]["type"] == 0
            and self.holding_objects[1]["type"] == 0
        ):
            s += f"I'm holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}). "
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "I'm holding nothing. "
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"I'm holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"I'm holding {s_hold[0]}{s_hold[1]}"

        # print(self.current_room, self.obj_per_room)
        if self.current_room not in self.rooms_explored:
            pred_room = "none"
        else:
            pred_room = self.rooms_explored[self.current_room]
        if pred_room != "all" and sss[self.current_room] == "nothing":
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it. "
        else:
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it and found {sss[self.current_room]}. "
        ### opponent modeling
        if not self.single:
            s_hold = ["", ""]
            for i, obj in enumerate(opponent_grabbed_objects):
                if obj["type"] == 0:
                    s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
                elif obj["type"] == 1:
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj["contained"]):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = "nothing"
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    s_hold[i] = (
                        f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "
                    )
            if (
                opponent_grabbed_objects[0]["type"] == 0
                and opponent_grabbed_objects[1]["type"] == 0
            ):
                ss = f"two target objects <{opponent_grabbed_objects[0]['name']}> ({opponent_grabbed_objects[0]['id']}) and <{opponent_grabbed_objects[1]['name']}> ({opponent_grabbed_objects[1]['id']}). "
            if s_hold[0] == "" and s_hold[1] == "":
                ss = "nothing. "
            elif s_hold[0] != "" and s_hold[1] != "":
                ss = f"{s_hold[0][:-2]}, and {s_hold[1]}"
            else:
                ss = f"{s_hold[0]}{s_hold[1]}"

            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == self.current_room:
                s += f"I also see {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == self.current_room:
                continue
            # s += f"I've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
            if room not in self.rooms_explored:
                pred_room = "none"
            else:
                pred_room = self.rooms_explored[room]
            if pred_room != "all" and sss[room] == "nothing":
                s += f"I've explored {pred_room} of the {room}. "
            else:
                s += f"I've explored {pred_room} of the {room}, and I found {sss[room]} there. "

        return s

    def get_available_plans(self, message):#plans according to the state
        """
        获取可用的规划

        参数:
            message: 消息文本

        返回:
            可用规划列表
        """
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        send a message: ""
        """
        available_plans = []
        if self.communication and message is not None:
            available_plans.append(f"send a message: {message}")
        if (
            self.holding_objects[0]["type"] is None
            or self.holding_objects[1]["type"] is None
        ):
            for obj in self.object_list[0]:
                available_plans.append(
                    f"go grasp target object <{obj['name']}> ({obj['id']})"
                )
            if not (
                self.holding_objects[0]["type"] == 1
                or self.holding_objects[1]["type"] == 1
            ):
                for obj in self.object_list[1]:
                    available_plans.append(
                        f"go grasp container <{obj['name']}> ({obj['id']})"
                    )
        else:
            if (
                self.holding_objects[0]["type"] == 1
                and self.holding_objects[0]["contained"][-1] is None
                and self.holding_objects[1]["type"] == 0
            ):
                available_plans.append(
                    f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})"
                )
            elif (
                self.holding_objects[1]["type"] == 1
                and self.holding_objects[1]["contained"][-1] is None
                and self.holding_objects[0]["type"] == 0
            ):
                available_plans.append(
                    f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})"
                )
        if (
            any(obj["type"] is not None for obj in self.holding_objects)
            and len(self.object_list[2]) != 0
        ):
            available_plans.append(f"transport objects I'm holding to the bed")
        for room in self.rooms:
            if room == self.current_room or room is None or room == "None":
                continue
            available_plans.append(f"go to {room}")
        if (
            self.current_room not in self.rooms_explored
            or self.rooms_explored[self.current_room] != "all"
        ):
            available_plans.append(f"explore current room {self.current_room}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans
    #COBEL -shaokang available_plans 
    def get_available_plans_cobel(self):#plans according to the state
        """
        获取可用的规划

        参数:
            message: 消息文本

        返回:
            可用规划列表
        """
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        """
        available_plans = []
        if (
            self.holding_objects[0]["type"] is None
            or self.holding_objects[1]["type"] is None
        ):
            for obj in self.object_list[0]:
                available_plans.append(
                    f"go grasp target object <{obj['name']}> ({obj['id']})"
                )
            if not (
                self.holding_objects[0]["type"] == 1
                or self.holding_objects[1]["type"] == 1
            ):
                for obj in self.object_list[1]:
                    available_plans.append(
                        f"go grasp container <{obj['name']}> ({obj['id']})"
                    )
        else:
            if (
                self.holding_objects[0]["type"] == 1
                and self.holding_objects[0]["contained"][-1] is None
                and self.holding_objects[1]["type"] == 0
            ):
                available_plans.append(
                    f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})"
                )
            elif (
                self.holding_objects[1]["type"] == 1
                and self.holding_objects[1]["contained"][-1] is None
                and self.holding_objects[0]["type"] == 0
            ):
                available_plans.append(
                    f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})"
                )
        if (
            any(obj["type"] is not None for obj in self.holding_objects)
            and len(self.object_list[2]) != 0
        ):
            available_plans.append(f"transport objects I'm holding to the bed")
        for room in self.rooms:
            if room == self.current_room or room is None or room == "None":
                continue
            available_plans.append(f"go to {room}")
        if (
            self.current_room not in self.rooms_explored
            or self.rooms_explored[self.current_room] != "all"
        ):
            available_plans.append(f"explore current room {self.current_room}")
        
        #COBEL subgoal finish
        available_plans.append("SUBGOAL DONE")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    #COBEL-zhimin
    def update_first_order_beliefs(self,first_order_beliefs,visual_observation, message, belief_rules):
        """
        更新信念状态

        参数:
            zero_order_beliefs: 零阶信念
            first_orderbeliefs: 一阶信念
            belief_rules: 信念规则

        返回:
            更新后的信念状态
        """
        #TODO completed the prompt
        prompt = (
            self.cobel_prompts_df["prompt"][0]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$BELIEF_RULES$", self.first_order_rules)
            .replace("$FIRST_ORDER_BELIEFS$", first_order_beliefs)
            .replace("$MESSAGE$", message)
            .replace("$VISUAL_OBSERVATION$", visual_observation)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        updated_first_order_beliefs = output[0]
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========updated_first_order_beliefs=============: \n{updated_first_order_beliefs}")
        return updated_first_order_beliefs

    #COBEL-zhimin
    def update_zero_order_beliefs(self,zero_order_beliefs, visual_observation, message, belief_rules):
        """
        更新信念状态

        参数:
            zero_order_beliefs: 零阶信念
            first_orderbeliefs: 一阶信念
            belief_rules: 信念规则

        返回:
            更新后的信念状态
        """
        #TODO completed the prompt
        prompt = (
            self.cobel_prompts_df["prompt"][1]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$BELIEF_RULES$", self.zero_order_rules)
            .replace("$ZERO_ORDER_BELIEFS$", zero_order_beliefs)
            .replace("$MESSAGE$", message)
            .replace("$VISUAL_OBSERVATION$", visual_observation)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        updated_zero_order_beliefs = output[0]
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========updated_zero_order_beliefs=============: \n{updated_zero_order_beliefs}")
        return updated_zero_order_beliefs

    #COBEL-zhimin
    def prediction_first_order(self, first_order_beliefs):
        prompt = (
            self.cobel_prompts_df["prompt"][2]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$FIRST_ORDER_BELIEFS$", first_order_beliefs)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        #这里的结果是Reason: ... Subgoal: ... 


        opponent_subgoal = self.extract_subgoal_content(output[0])
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========opponent_subgoal=============: \n{output[0]}")

        return opponent_subgoal
    
    #COBEL-zhimin
    def prediction_zero_order(self, first_order_beliefs, zero_order_beliefs):
        prompt = (
            self.cobel_prompts_df["prompt"][3] 
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$FIRST_ORDER_BELIEFS$", first_order_beliefs)
            .replace("$ZERO_ORDER_BELIEFS$", zero_order_beliefs)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        # match = re.search(r'Subgoal[：:]\s*(.+?)(?:\n\S|\Z)', output[0], re.DOTALL)
        # if match:
        #     output = match.group(1).strip()


        my_subgoal = self.extract_subgoal_content(output[0])
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========my_subgoal=============: \n{output[0]}")
  

        return my_subgoal
    
    #COBEL - zhimin
    def belief_awareness(self, first_order_beliefs, zero_order_beliefs):
        """
        信念意识

        参数:
            first_order_beliefs: 一阶信念
            zero_order_beliefs: 零阶信念

        返回:
            信念差异分数
            信念差异文本
        """
        prompt = (
            self.cobel_prompts_df["prompt"][4]
            .replace("$FIRST_ORDER_BELIEFS$", first_order_beliefs)
            .replace("$ZERO_ORDER_BELIEFS$", zero_order_beliefs)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                )   
        
        # Extract Difference score
        difference_score_match = re.search(r'- Difference score:\s*(\d+)', output[0])
        difference_score = difference_score_match.group(1) if difference_score_match else None

        # Extract Difference content
        difference_content_match = re.search(r'- Different content:\s*(.*?)^-*', output[0], re.DOTALL | re.MULTILINE)
        difference_content = difference_content_match.group(1).strip() if difference_content_match else None

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========difference=============: \n{output[0]}")

        return difference_score, difference_content
    
    #COBEL - zhimin
    def init_beliefs(self, belief_rules:str, goal:str, room_list:List[str]):


        room_des = ""
        for room in room_list:
            room_des += f"{room}, "
        room_des = room_des[:-2] + ". "
        prompt = (
            self.cobel_prompts_df["prompt"][6]  #-> init_beliefs
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$BELIEF_RULES$", belief_rules)
            .replace("$GOAL$", goal)
            .replace("$ROOM_LIST$", room_des)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        initial_beliefs = output[0]
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========initial_beliefs=============: \n{initial_beliefs}")
        
        pattern1 = r'(Zero\s+order\s+beliefs:.+?)(?=First\s+order\s+beliefs:)'
        zero_order_match = re.search(pattern1, initial_beliefs, re.DOTALL | re.IGNORECASE)

        # 匹配从 "First order beliefs:" 开始到文本结尾
        pattern2 = r'(First\s+order\s+beliefs:.*)'
        first_order_match = re.search(pattern2, initial_beliefs, re.DOTALL | re.IGNORECASE)

        # 提取匹配的字符串
        zero_order_beliefs = zero_order_match.group(1) if zero_order_match else ""
        first_order_beliefs = first_order_match.group(1) if first_order_match else ""

        return zero_order_beliefs, first_order_beliefs
      
      
    #COBEL  -shaokang
    def intuitive_planning(self,zero_order_beliefs,subgaol,action_history):#TODO:subgoal need tobe change to formally the combination of the available plans
        # Done:action will be cleaned once the subgoal is done
        available_plans, num, available_plans_list = self.get_available_plans_cobel()
        prompt = (
            self.cobel_prompts_df['prompt'][5]
            .replace('$AGENT_NAME$',self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace('$MYSTATE$',zero_order_beliefs)
            .replace('$SUBGOAL$',subgaol)
            .replace('$PREVIOUSACTIONS$',action_history)
            .replace('$AVAILABLEACTIONS$',available_plans)
        )
        chat_prompt = [{'role':'user','content':prompt}]
        output,usage = self.generator(
            chat_prompt,self.sampling_params
        )
        #TODO: COBEL parse checking the efficiency
        plan, flags = self.parse_answer(available_plans_list, output[0])
        return plan
      
    #COBEL - zhimin
    def message_generation(self, difference_content):

            # 使用正则表达式提取zero_order_belief后面的内容
        zero_match = re.search(r'Zero_order_belief:\s*(.*?)(?:\n|$)', difference_content)
        zero_order_belief = zero_match.group(1) if zero_match else ""

        # 使用正则表达式提取first_order_belief后面的内容
        first_match = re.search(r'First_order_belief:\s*(.*?)(?:\n|$)', difference_content)
        first_order_belief = first_match.group(1) if first_match else ""

        prompt = (
            self.cobel_prompts_df["prompt"][6]  #-> init_beliefs
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$FIRST_ORDER_BELIEF_DIFFERENCE$", first_order_belief)
            .replace("$ZERO_ORDER_BELIEF_DIFFERENCE$", zero_order_belief)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        mes_list = output[0]
        if self.belief_debug:
            print(f"=========message_prompt===========: \n{prompt}")
            print(f"=========mes_list=============: \n{mes_list}")
        try:
            result_dict = ast.literal_eval(mes_list)
            print(result_dict)
        except Exception as e:
            print(f"Error: {e}")
        return result_dict
    
    def run(
        self,
        current_step,
        current_room,
        rooms_explored,
        holding_objects,
        satisfied,
        object_list,
        obj_per_room,
        action_history,
        dialogue_history,
        opponent_grabbed_objects=None,
        opponent_last_room=None,
        episode_logger = None
    ):
        """
        运行模型生成规划

        参数:
            current_step: 当前步骤
            current_room: 当前房间
            rooms_explored: 已探索的房间
            holding_objects: 持有的物体
            satisfied: 已完成的物体
            object_list: 物体列表
            obj_per_room: 每个房间的物体
            action_history: 动作历史
            dialogue_history: 对话历史
            opponent_grabbed_objects: 对手抓取的物体
            opponent_last_room: 对手最后所在的房间

        返回:
            生成的规划和相关信息
        """
        info = {}
        print("current_step", current_step)
        # llm_logger.info(f"当前步骤: {current_step}")
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        progress_desc = self.progress2text(
            current_step, satisfied, opponent_grabbed_objects, opponent_last_room
        )
        action_history_desc = ", ".join(
            action_history[-10:] if len(action_history) > 10 else action_history
        )
        dialogue_history_desc = "\n".join(
            dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history
        )
        prompt = self.prompt_template.replace("$GOAL$", self.goal_desc)
        prompt = prompt.replace("$PROGRESS$", progress_desc)
        prompt = prompt.replace("$ACTION_HISTORY$", action_history_desc)
        message = None

        if self.communication:
            prompt = prompt.replace("$DIALOGUE_HISTORY$", dialogue_history_desc)
            if not action_history[-1].startswith("send a message"):
                gen_prompt = self.generator_prompt_template.replace(
                    "$GOAL$", self.goal_desc
                )
                gen_prompt = gen_prompt.replace("$PROGRESS$", progress_desc)
                gen_prompt = gen_prompt.replace("$ACTION_HISTORY$", action_history_desc)
                gen_prompt = gen_prompt.replace(
                    "$DIALOGUE_HISTORY$", dialogue_history_desc
                )
                gen_prompt = gen_prompt + f"\n{self.agent_name}:"
                chat_prompt = [{"role": "user", "content": gen_prompt}]
                outputs, usage = self.generator(
                    chat_prompt if self.chat else gen_prompt, self.sampling_params
                ) # usage token cost
                self.total_cost += usage
                message = outputs[0]
                if len(message) > 0 and message[0] != '"':
                    message = re.search(r'"([^"]+)"', message)
                    if message:
                        message = '"' + message.group(1) + '"'
                info["prompt_comm"] = gen_prompt
                info["output_comm"] = outputs
                info["usage_comm"] = usage
                if self.debug:
                    print(f"prompt_comm:\n{gen_prompt}")
                print(f"output_comm:\n{message}")
                episode_logger.info(f"output_message:\n{message}")

        available_plans, num, available_plans_list = self.get_available_plans(message) #因为要传入消息,only for message in the available plans
        if num == 0 or (message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num, "plan": None})
            return plan, info

        prompt = prompt.replace("$AVAILABLE_ACTIONS$", available_plans)

        if self.cot:
            prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(
                chat_prompt if self.chat else prompt, self.sampling_params
            )
            output = outputs[0]
            ## truncate the unfinished cot
            last_index = output.rfind(".")
            if last_index != -1:
                output = output[: last_index + 1]
            else:
                output += "."
            self.total_cost += usage
            # info['outputs_cot'] = outputs
            # info['usage_plan_stage_1'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
            chat_prompt = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
                {
                    "role": "user",
                    "content": "Answer with only one best next action. So the answer is option",
                },
            ]
            normal_prompt = (
                prompt
                + " "
                + output
                + " Answer with only one best next action. So the answer is option"
            )
            episode_logger.info(f"input_prompt:\n{normal_prompt}")
            outputs, usage = self.generator(
                chat_prompt if self.chat else normal_prompt, self.sampling_params
            )
            output = outputs[0]
            episode_logger.info(f"output_plan:\n{output}")
            self.total_cost += usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            normal_prompt = prompt
            episode_logger.info(f"input_prompt:\n{normal_prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            outputs, usage = self.generator(
                chat_prompt if self.chat else normal_prompt, self.sampling_params
            )
            output = outputs[0]
            episode_logger.info(f"output_plan:\n{output}")
            self.total_cost += usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
        plan, flags = self.parse_answer(available_plans_list, output)
        #这里plan就是包含消息的动作
        if flags == "COMMUNICATION":
            self.communication_cost += 1
            self.characters += len(plan.split(" ")) #send a message: "xxxxx" character
            # # 新增：记录通信内容
            # if plan.startswith("send a message:"):
            #     message_content = plan[len("send a message:"):].strip()
                # llm_logger.info(f"{self.agent_name} 发送消息内容: {message_content}\n当前通信次数{self.communication_cost}")
            # llm_logger.info(f"{self.agent_name}:当前计划:\n{plan}")
        if self.debug:
            print(f"plan: {plan}\n")
        info.update(
            {
                "num_available_actions": num,
                "prompt_plan_stage_2": normal_prompt,
                "output_plan_stage_2": output,
                "parse_exception": flags,
                "plan": plan,
                "total_cost": self.total_cost,
            }
        )
        return plan, info
    

    #COBEL-zhimin
    def extract_subgoal_content(self, text):
        """
        提取最后一个 subgoal: 之后的内容
        """
        # 简单匹配 Subgoal: 后面的任何内容直到行尾
        pattern = r'Subgoal:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text)
        
        # 返回最后一个匹配的内容，清理空白字符
        if matches:
            return matches[-1].strip()
        else:
            return None

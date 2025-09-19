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
# from transformers import ( #COBEL - zhimin 更换主机后报错暂时
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     LlamaForCausalLM,
#     LlamaTokenizer,
# )
from openai import AzureOpenAI
from openai import OpenAIError
from openai import OpenAI
from datetime import datetime
import ast
from LLM.bert_consine_calculate import BeliefSimilarityCalculator



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
        self.my_rooms_explored = None
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
        df = pd.read_csv(self.prompt_template_path, quotechar='"', quoting=1)

        
        #COBEL - zhimin
        with open("./LLM/belief_symbolic_language_no_conf.txt", "r", encoding="utf-8") as f:
            self.belief_symbolic_language = f.read()
        self.cobel_prompts_df = pd.read_csv(self.prompt_template_path)
        self.total_tokens = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.api = 0
        self.comm_chars = 0
        self.comm_counts = 0
        with open("./LLM/rules_tdw_replace.txt", "r", encoding="utf-8") as f:
            self.belief_rules = f.read()
        # with open("./LLM/rules_tdw_no_conf.txt", "r", encoding="utf-8") as f:
        #     self.belief_rules = f.read()
        # 添加token统计字典
        self.token_stats = {}
        for call_name in ['small_model',"large_model","init_beliefs","update_beliefs","prediction_zero_order","prediction_first_order","intuitive_planning","cooradination_aware","communication"]:
            self.token_stats[call_name] = {
                "prompt": 0,
                "completion": 0,
                "call_counts": 0
            }
        




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
        self.chat = True
        self.OPENAI_KEY = None  # OpenAI API密钥
        self.total_cost = 0  # 总花费
        self.communication_cost = 0  # 通信花费
        # 根据不同来源初始化模型
        if self.source == "openai":
            # OpenAI模型初始化
            # api_key=os.environ.get("CHATANYWHERE_API_KEY")
            # base_url=os.environ.get("CHATANYWHERE_URL")

            api_key=os.environ.get("CHATANYWHERE_API_KEY")
            base_url=os.environ.get("CHATANYWHERE_URL")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
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
        elif self.source == "aliyun":
            # DeepSeek模型初始化
            client = OpenAI(
                api_key=os.environ.get("ALIYUN_API_KEY"),
                base_url=os.environ.get("ALIYUN_URL"),
            )
            if self.chat:
                self.sampling_params = {
                    "extra_body": {"enable_thinking": False},
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "extra_body": {"enable_thinking": False},
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
            def openai_generate(prompt, sampling_params, model_size="large"):
                
                for attempt in range(10):
                    try:
                        # 根据model_size参数选择不同的模型
                        if model_size == "small":
                            # 使用小参数模型
                            model_to_use = "qwen2.5-7b-instruct" #TODO
                        else:
                            # 使用大参数模型（默认）
                            model_to_use = self.lm_id #TODO
                        
                        if self.chat:
                            response = client.chat.completions.create(
                                model=model_to_use, messages=prompt, **sampling_params
                            )
                            self.api += 1
                            usage = [response.usage.prompt_tokens,response.usage.completion_tokens]
                            
                            # 获取token数量
                            prompt_tokens = usage[0]
                            completion_tokens = usage[1]
                            self.total_tokens += (prompt_tokens + completion_tokens)
                            self.completion_tokens += completion_tokens
                            # 根据模型大小记录token使用情况
                            if model_size == "small":
                                # 同时记录到token_stats中
                                self.token_stats["small_model"]["prompt"] += prompt_tokens
                                self.token_stats["small_model"]["completion"] += completion_tokens
                                self.token_stats["small_model"]["call_counts"] += 1
                            else:
                                self.token_stats["large_model"]["prompt"] += prompt_tokens
                                self.token_stats["large_model"]["completion"] += completion_tokens
                                self.token_stats["large_model"]["call_counts"] += 1
                            
                            # 总token计数
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
                                model=model_to_use, prompt=prompt, **sampling_params
                            )
                            
                            # 根据模型大小记录token使用情况
                            if model_size == "small":
                                self.small_model_tokens_in += response.usage.prompt_tokens
                                self.small_model_tokens_out += response.usage.completion_tokens
                            else:
                                self.large_model_tokens_in += response.usage.prompt_tokens
                                self.large_model_tokens_out += response.usage.completion_tokens

                            # 总token计数
                            

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
                        return generated_samples, usage
                    except OpenAIError as e:
                        if attempt == 5:
                            print(e)
                            raise e
                    

            
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

            def _generate(prompt, sampling_params, model_size="large"):
                usage = 0
                if source == "openai" or source == 'aliyun':
                    return openai_generate(prompt, sampling_params, model_size)
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
        self.belief_calculator = BeliefSimilarityCalculator()
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
        
        self.total_tokens = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.api = 0
        self.comm_chars = 0
        self.comm_counts = 0
        self.total_cost = 0
        # 重置token统计字典
        self.token_stats = {}
        for call_name in ['small_model',"large_model","init_beliefs","update_beliefs","prediction_zero_order","prediction_first_order","intuitive_planning","cooradination_aware","communication"]:
            self.token_stats[call_name] = {
                "prompt": 0,
                "completion": 0,
                "call_counts": 0
            }

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
        for room, obj_list in self.my_objects_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            s_bed = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object {x}"
                else:
                    ss = ", ".join([f"{x}" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container {x}"
                else:
                    ss = ", ".join([f"{x}" for x in cons])
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
        if self.current_room not in self.my_rooms_explored:
            pred_room = "none"
        else:
            pred_room = self.my_rooms_explored[self.current_room]
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
            # s += f"I've explored {self.my_rooms_explored[room] if room in self.my_rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
            if room not in self.my_rooms_explored:
                pred_room = "none"
            else:
                pred_room = self.my_rooms_explored[room]
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
        #整体的问题就是不允许不看到然后抓取 只能是别人跟我说了 所以我去那个房间找到 看到再抓取
        available_plans = []
        if (
            self.holding_objects[0]["type"] is None #COBEL - zhimin 出现了 Nonetype报错的情况
            or self.holding_objects[1]["type"] is None
        ):
            for obj in self.object_list[0]: #TODO 看看这里的object_list是代表的可见物体吗 - 不是 是在地图里存的信息 0是普通物体 1是容器 2是床
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
            and len(self.object_list[2]) != 0 #看到床才能运输 TODO 这个地方可以判断床是不是可以用 并且两个手都是满的 （可能不被填满就走了）
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
        
        #COBEL subplan finish
        available_plans.append("SUBPLAN DONE")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    #COBEL-zhimin
    def update_beliefs(self,received_messages,dialogues):
        print("======更新信念======")
        updated_zero_order_beliefs = None
        updated_first_order_beliefs = None
        pattern_zero = r"zero order belief rules:\s*(.*?)\s*(?=first order belief rules:)"
        match_zero = re.search(pattern_zero, self.belief_rules, re.IGNORECASE | re.DOTALL)
        if not match_zero:
            raise ValueError("Failed to extract updated beliefs from output.")
        zero_order_belief_rules = match_zero.group(1).strip()
        zero_order_belief_rules = zero_order_belief_rules.replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        
        
        
        pattern_first = r"first order belief rules:\s*(.*)"
        match_first = re.search(pattern_first, self.belief_rules, re.IGNORECASE | re.DOTALL)
        if not match_first:
            raise ValueError("Failed to extract updated beliefs from output.")
        first_order_belief_rules = match_first.group(1).strip()
        first_order_belief_rules = first_order_belief_rules.replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        #first
        if dialogues != "None":
            prompt = (
                self.cobel_prompts_df["prompt"][0]
                .replace("$AGENT_NAME$", self.agent_name)
                .replace("$OPPO_NAME$", self.oppo_name)
                .replace("$MESSAGE$", dialogues)
                .replace("$RULE$", first_order_belief_rules)
            )
            system_prompt = "You MUST answer strictly in this format:\n$OPPO_NAME$ knows:\nfirst order beliefs\n$OPPO_NAME$'s plan:".replace("$OPPO_NAME$", self.oppo_name)
            chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
            # chat_prompt = [{"role": "user", "content": prompt}]
            first_output, usage = self.generator(
                        chat_prompt, self.sampling_params
                    ) # usage token cost
            # 记录token消耗
            method_name = "update_beliefs"
            # 使用usage.prompt_tokens和usage.completion_tokens
            prompt_tokens = usage[0]
            completion_tokens = usage[1]
            self.token_stats[method_name]["prompt"] += prompt_tokens
            self.token_stats[method_name]["completion"] += completion_tokens
            self.token_stats[method_name]["call_counts"] += 1



            if self.belief_debug:
                # print(f"=========prompt===========: \n{prompt}")
                print(f"=========updated_first_beliefs=============: \nfirst:{first_output[0]}")

            pattern_first = rf"first order beliefs:\s*(.*?)\s*(?={re.escape(self.oppo_name)}'s plan:)"
            first_match = re.search(pattern_first, first_output[0], re.IGNORECASE | re.DOTALL)
            if not first_match:
                updated_first_order_beliefs = None
            else:
                updated_first_order_beliefs = first_match.group(1).strip()
            
        else:
            updated_first_order_beliefs = None
            if self.belief_debug:
                print(f"=========no first update==========: \n")
            
        if received_messages != "None":
            prompt = (
                self.cobel_prompts_df["prompt"][1]
                .replace("$AGENT_NAME$", self.agent_name)
                .replace("$OPPO_NAME$", self.oppo_name)
                .replace("$MESSAGE$", received_messages)
                .replace("$RULE$", zero_order_belief_rules)
            )
            #zero
            system_prompt = "You MUST answer strictly in this format:\n$AGENT_NAME$ knows:\nzero order beliefs:\n$OPPO_NAME$'s plan:".replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
            chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]

            zero_output, usage = self.generator(
                        chat_prompt, self.sampling_params 
                    ) # usage token cost
            
            # 记录token消耗
            method_name = "update_beliefs"
            # 使用usage.prompt_tokens和usage.completion_tokens
            prompt_tokens = usage[0]
            completion_tokens = usage[1]
            self.token_stats[method_name]["prompt"] += prompt_tokens
            self.token_stats[method_name]["completion"] += completion_tokens
            self.token_stats[method_name]["call_counts"] += 1
            # pattern_zero = r'zero.*?:\s*(.*?)(?=first.*?:|$)'
            
            pattern_zero = rf"zero order beliefs:\s*(.*?)\s*(?={re.escape(self.oppo_name)}'s plan:)"
            zero_match = re.search(pattern_zero, zero_output[0], re.IGNORECASE | re.DOTALL)



            pattern_plan = rf"{re.escape(self.oppo_name)}'s plan:\s*(.*)"
            plan_match = re.search(pattern_plan, zero_output[0], re.IGNORECASE | re.DOTALL)

            
            
            if self.belief_debug:
                # print(f"=========prompt===========: \n{prompt}")
                print(f"=========updated_zero_beliefs=============: \nzero:{zero_output[0]}")
            
            if not zero_match or not plan_match:
                updated_zero_order_beliefs = None
                oppo_subplan = None
            else:
                updated_zero_order_beliefs = zero_match.group(1).strip()
                oppo_subplan = plan_match.group(1).strip()
        else:
            updated_zero_order_beliefs = None
            oppo_subplan = None
            if self.belief_debug:
                print(f"=========no zero update==========: \n")
        

        

        
        return updated_zero_order_beliefs, updated_first_order_beliefs , oppo_subplan
        

        

    #COBEL-zhimin
    def prediction_first_order(self, oppo_progress):

        prompt = (
            self.cobel_prompts_df["prompt"][2]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$OPPO_PROGRESS$", oppo_progress)
            .replace("$GOAL$", self.goal_desc)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasoning:\nsubplans: \nsubplan1: \nsubplan2: \nsubplan3:"
        chat_prompt = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        
        # 记录token消耗
        method_name = "prediction_first_order"
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1

        # 使用正则表达式提取 $OPPO_NAME$'s Subplans
        # 使用正则表达式提取 Updated Beliefs
        pattern_beliefs = r'reasoning:\s*(.*?)(?=' + re.escape(f"subplans:") + r'|$)'
        match_beliefs = re.search(pattern_beliefs, output[0], re.IGNORECASE | re.DOTALL)
        if not match_beliefs:
            raise ValueError("Failed to extract updated beliefs from output.")
        reason = match_beliefs.group(1).strip()

        # 使用正则表达式提取 $OPPO_NAME$'s Subplans
        pattern_subplan = rf"subplans:\s*(.*)"
        match_subplan = re.search(pattern_subplan, output[0], re.IGNORECASE | re.DOTALL)
        if not match_subplan:
            raise ValueError("Failed to extract opponent subplans from output.")
        opponent_subplans = match_subplan.group(1).strip()
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========predict_first=============: \n{output[0]}")


        return reason, opponent_subplans
    
    def prediction_zero_order(self, my_progress):
        prompt = (
            self.cobel_prompts_df["prompt"][3]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MY_PROGRESS$", my_progress)
            .replace("$GOAL$", self.goal_desc)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasoning:\nsubplan:"
        chat_prompt = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        # chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        
        # 记录token消耗
        method_name = "prediction_zero_order"
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1

        # 使用正则表达式提取 $OPPO_NAME$'s SubPlans
        # 使用正则表达式提取 Updated Beliefs
        # pattern_beliefs = r'updated zero order beliefs:\s*(.*?)(?=' + re.escape("reasons:") + r'|$)'
        pattern_beliefs = r'reasoning:\s*(.*?)(?=' + re.escape("subplan:") + r'|$)'
        match_beliefs = re.search(pattern_beliefs, output[0], re.IGNORECASE | re.DOTALL)
        if not match_beliefs:
            raise ValueError("Failed to extract updated beliefs from output.")
        reason = match_beliefs.group(1).strip()

        # 使用正则表达式提取 $OPPO_NAME$'s Subplans
        # pattern_subplan = r'reasons:\s*(.*?)(?=' + re.escape("subplan:") + r'|$)'
        pattern_subplan = rf"subplan:\s*(.*)"
        match_subplan = re.search(pattern_subplan, output[0], re.IGNORECASE | re.DOTALL)
        if not match_subplan:
            raise ValueError("Failed to extract subplan from output.")
        my_subplan = match_subplan.group(1).strip()
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========predict_zero=============: \n{output[0]}")

        # episode_logger.info(
        #     f"\n{self.agent_name}predict_first_order:\n{output[0]}"
        # )
        return reason, my_subplan

    def passive_prediction_zero_order(self, my_progress, oppo_subplan):
        prompt = (
            self.cobel_prompts_df["prompt"][4]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MY_PROGRESS$", my_progress)
            .replace("$OPPO_SUBPLAN$", oppo_subplan)
            .replace("$GOAL$", self.goal_desc)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasoning:\nsubplan:"
        chat_prompt = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        # chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        
        # 记录token消耗
        method_name = "prediction_zero_order"
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1

        # 使用正则表达式提取 $OPPO_NAME$'s SubPlans
        # 使用正则表达式提取 Updated Beliefs
        # pattern_beliefs = r'updated zero order beliefs:\s*(.*?)(?=' + re.escape("reasons:") + r'|$)'
        pattern_beliefs = r'reasoning:\s*(.*?)(?=' + re.escape("subplan:") + r'|$)'
        match_beliefs = re.search(pattern_beliefs, output[0], re.IGNORECASE | re.DOTALL)
        if not match_beliefs:
            raise ValueError("Failed to extract updated beliefs from output.")
        reason = match_beliefs.group(1).strip()

        # 使用正则表达式提取 $OPPO_NAME$'s Subplans
        # pattern_subplan = r'reasons:\s*(.*?)(?=' + re.escape("subplan:") + r'|$)'
        pattern_subplan = rf"subplan:\s*(.*)"
        match_subplan = re.search(pattern_subplan, output[0], re.IGNORECASE | re.DOTALL)
        if not match_subplan:
            raise ValueError("Failed to extract subplan from output.")
        my_subplan = match_subplan.group(1).strip()
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========predict_zero=============: \n{output[0]}")

        # episode_logger.info(
        #     f"\n{self.agent_name}predict_first_order:\n{output[0]}"
        # )
        return reason, my_subplan
    
    #COBEL - zhimin
    def coordination_aware(self, my_progress, oppo_progress, my_subplan, opponent_subplans):
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
            self.cobel_prompts_df["prompt"][5]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MY_PROPGRESS$", my_progress)
            .replace("$OPPO_PROGRESS$", oppo_progress)
            .replace("$MY_SUBPLAN$", my_subplan)
            .replace("$OPPO_SUBPLAN$", opponent_subplans)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasons:\nanswer:\nmisaligned information:\n"
        chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                )   
        
        # 记录token消耗
        method_name = "cooradination_aware"
        # 使用usage.prompt_tokens和usage.completion_tokens
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1
        

        if self.belief_debug:
            # print(f"=========prompt===========: \n{prompt}")
            print(f"=========coordination_aware=============: \n{output[0]}")

        pattern_reason = r'reasons:\s*(.*?)(?=' + re.escape("answer:") + r'|$)'
        match_reason = re.search(pattern_reason, output[0], re.IGNORECASE | re.DOTALL)
        if not match_reason:
            raise ValueError("Failed to extract reason from output.")
        reason = match_reason.group(1).strip()


        pattern_answer = r'answer:\s*(.*?)(?=' + re.escape("misaligned information:") + r'|$)'
        match_answer = re.search(pattern_answer, output[0], re.IGNORECASE | re.DOTALL)
        if not match_answer:
            raise ValueError("Failed to extract answer from output.")
        answer = match_answer.group(1).strip()

        if "NO" not in answer.upper():

            
            pattern_difference = rf"misaligned information:\s*(.*)"
            match_difference = re.search(pattern_difference, output[0], re.IGNORECASE | re.DOTALL)
            if not match_difference:
                raise ValueError("Failed to extract difference from output.")
            difference = match_difference.group(1).strip()
        else:
            answer = "NO MISCOORDINATION"
            reason = "None"
            difference = "None"

        
        return answer, reason, difference

    #COBEL  -shaokang - zhimin update 因为原本是通过run规划，所以会在run传入很多信息来更新状态来提供available action


    def comm(self, difference, my_subplan):
        prompt = (
            self.cobel_prompts_df["prompt"][6]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MISALIGNED INFORMATION$", difference)
            .replace("$MY_SUBPLAN$", my_subplan)
        )

        system_prompt = "Just output the message content without any additional analysis, quotes or reasons. Just output the message. "
        chat_prompt = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                )
        
        # 记录token消耗
        method_name = "communication"
        # 使用usage.prompt_tokens和usage.completion_tokens
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1
        
        # pattern_message = rf"message:\s*(.*)"
        # match_message = re.search(pattern_message, output[0], re.IGNORECASE | re.DOTALL)
        # if not match_message:
        #     raise ValueError("Failed to extract message from output.")
        # message = match_message.group(1).strip()
        message = output[0]
        if self.belief_debug:
            # print(f"=========prompt===========: \n{prompt}")
            print(f"=========message=============: \n{output[0]}")
        return message
    




    def intuitive_planning(self,
                           my_subplan,
                           action_history,
                           progress_desc,
                           episode_logger = None, 
                           ):

        
        available_plans, num, available_plans_list = self.get_available_plans_cobel()
        prompt = (
            self.cobel_prompts_df["prompt"][7]
            .replace('$AGENT_NAME$',self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$GOAL$", self.goal_desc)
            .replace('$MY_SUBPLAN$',my_subplan)
            .replace('$PREVIOUS_ACTIONS$',action_history)
            .replace('$PROGRESS$',progress_desc)
            .replace('$ACTION_LIST$',available_plans)
        )

        system_prompt = "You MUST follow the output format strictly.Format: answer:\nreasons:\nanswer:"
        chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
        # chat_prompt = [{'role':'user','content':prompt}]
        output,usage = self.generator(
            chat_prompt,self.sampling_params
        )
        
        # 记录token消耗
        method_name = "intuitive_planning"
        # 使用usage.prompt_tokens和usage.completion_tokens
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1
        
        
        if self.belief_debug:
            print(f"=========plan_prompt===========: \n{prompt}")
            print(f"=========intuitive_planning=============: \n{output[0]}")
            episode_logger.info(f"{self.agent_name}: 选动作 {output[0]}")
        
        episode_logger.info(
            f"\n{self.agent_name} intuitive_planning:\n{output[0]}"
        )

        pattern_answer = rf"answer:\s*(.*)"
        match_answer = re.search(pattern_answer, output[0], re.IGNORECASE | re.DOTALL)
        if not match_answer:
            raise ValueError("Failed to extract opponent answers from output.")
        answer = match_answer.group(1).strip()

        #TODO: COBEL parse checking the efficiency
        plan, flags = self.parse_answer(available_plans_list, answer)
        return plan
    
    def random_planning(self):

        
        available_plans, num, available_plans_list = self.get_available_plans_cobel()
        filtered_plans = [item for item in available_plans_list if "SUBPLAN DONE" not in item]
        random_plan = random.choice(filtered_plans)
        return random_plan
    

    

    def get_progress_description(
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
        episode_logger = None,
        oppo_holding_objects_zero = None,
        oppo_current_room_zero = None,
        my_rooms_explored = None,
        my_objects_per_room = None,
    ):
        info = {}
        print("current_step", current_step)
        self.current_room = current_room
        self.rooms_explored = rooms_explored #COBEL这里用全局的 但是动作要用局部的
        self.my_rooms_explored = my_rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        self.my_objects_per_room = my_objects_per_room

        #优先还是用看到的信息
        #如果没看到 就用消息维护的
        if not opponent_grabbed_objects:
            opponent_grabbed_objects = oppo_holding_objects_zero #check
        if not opponent_last_room:
            opponent_last_room = oppo_current_room_zero

        #COBEL - zhimin 这里会涉及初始化
        progress_desc = self.progress2text(
            current_step, satisfied, opponent_grabbed_objects, opponent_last_room
        )

        return progress_desc
    
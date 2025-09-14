import random

import openai
import torch
import json
import os
import pandas as pd
from openai import OpenAIError
import backoff
from openai import OpenAI
import os

class LLM_cobel:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_name = "Alice" if agent_id == 2 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 2 else "he"
        # self.debug = sampling_parameters.debug
        self.debug = True
        self.goal_location = None
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        if communication:
            self.generator_prompt_template = df['prompt'][1].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        else:
            self.generator_prompt_template = None

        self.communication = communication
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = True
        self.OPENAI_KEY = None
        self.total_cost = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #COBEL
        self.completion_tokens = 0
        self.total_tokens = 0
        self.comm_tokens = 0
        self.api_num = 0

        self.token_stats = {}
        for call_name in ['small_model',"large_model","init_beliefs","update_beliefs","prediction_zero_order","prediction_first_order","intuitive_planning","cooradination_aware","communication"]:
            self.token_stats[call_name] = {
                "prompt": 0,
                "completion": 0,
                "call_counts": 0
            }
        if self.source == 'openai':
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
        elif source == 'huggingface':
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                # 'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'early_stopping': True,
            }
        elif source == "debug":
            self.sampling_params = sampling_parameters
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id, device):
            if source == 'huggingface':
                from transformers import AutoModelForCausalLM, AutoTokenizer, LLaMATokenizer, LLaMAForCausalLM
                print(f"loading huggingface model {lm_id}")
                if 'llama' in lm_id or 'alpaca' in lm_id:
                    tokenizer = LLaMATokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce') # '/gpfs/u/scratch/AICD/AICDhnng/.cache')
                    model = LLaMAForCausalLM.from_pretrained(lm_id, # device_map="balanced_low_0",
                                                             # max_memory = {0: "10GB", 1: "20GB", 2: "20GB", 3: "20GB",4: "20GB",5: "20GB",6: "20GB",7: "20GB"},
                                                             torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                                load_in_8bit=False,
                                                             cache_dir='/work/pi_chuangg_umass_edu/.cahce')\
                                                                .to(device)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce')
                    model = AutoModelForCausalLM.from_pretrained(lm_id, torch_dtype=torch.float16,
                                                                 pad_token_id=tokenizer.eos_token_id,
                                                                 cache_dir='/work/pi_chuangg_umass_edu/.cahce').to(
                        device)
                print(f"loaded huggingface model {lm_id}")

            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            response = client.chat.completions.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response.to_dict(), indent=4))
                                    f.write('\n')
                            generated_samples = [
                                choice.message.content 
                                for choice in response.choices  # 直接遍历 choices 对象
                            ]
                            self.completion_tokens += response.usage.completion_tokens
                            self.total_tokens += response.usage.total_tokens
                            self.api_num += 1
                            #COBEL usage = completion token
                            usage = [response.usage.prompt_tokens,response.usage.completion_tokens]
                            # if 'gpt-4' in self.lm_id:
                            # 	usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage']['completion_tokens'] * 0.06 / 1000
                            # elif 'gpt-3.5' in self.lm_id:
                            # 	usage = response['usage']['total_tokens'] * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [
                                choice.message.content 
                                for choice in response.choices  # 直接遍历 choices 对象
                            ]
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 			  range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == 'huggingface':
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    # print(sampling_params)
                    output_dict = model.generate(input_ids, # max_length=prompt_len + sampling_params['max_new_tokens'],
                                                 **sampling_params)
                    generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                    # vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)
                    # token_log_probs = torch.gather(vocab_log_probs, 2,
                    # 							   output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()
                    for i, sample in enumerate(generated_samples):
                        stop_idx = sample.index('\n') if '\n' in sample else None
                        generated_samples[i] = sample[:stop_idx]
                    # 	token_log_probs[i] = token_log_probs[i][:stop_idx]
                    # mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
                elif source == "debug":
                    return ["navigation"]
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id, self.device)


    def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
        self.rooms = rooms_name
        self.roomname2id = roomname2id
        self.completion_tokens = 0
        self.comm_tokens = 0
        self.total_tokens = 0
        self.api_num = 0
        self.goal_location = goal_location
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)
        self.token_stats = {}
        for call_name in ['small_model',"large_model","init_beliefs","update_beliefs","prediction_zero_order","prediction_first_order","intuitive_planning","cooradination_aware","communication"]:
            self.token_stats[call_name] = {
                "prompt": 0,
                "completion": 0,
                "call_counts": 0
            }


    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        # print(goals)
        map_rel_to_pred = {
            'inside': 'into',
            'on': 'onto',
        }
        s = "Find and put "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_')
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
                # s += f"Alice holds a book, "
            elif relation == 'sit':
                continue
                # s += f"Alice sits in {obj2}, "
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return "None."

        s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."
        # if type(goal_location_room) is not list:
        # 	s += f" in the {goal_location_room}."
        # else:
        # 	ss = ' or '.join([f'{room}' for room in goal_location_room])
        # 	s += f", which may be in the {ss}."
        return s, f"{map_rel_to_pred[r]} the {self.goal_location}"




    def update_beleifs(self,message,dialogue):
        pass



    def plan_zero(self,my_progress):
        pass


    def plan_first(self,oppo_progress):
        pass



    def coordination_aware(self,my_progress,oppo_progress):
        pass



    def passive_plan(self, my_progress, oppo_subplan):
        pass

    # def get_obj(self, obs, text, k=1):
    # 	id2node = {node['id']: node for node in obs['nodes']}
    # 	cnt = 0
    # 	for x, node in id2node.items():
    # 		if f'({x})' in text:
    # 			cnt += 1
    # 			if cnt != k: continue
    # 			return f"<{node['class_name']}> ({x})"
    # 	print("WARNING! No object correctly parsed!!! Random choose one")
    # 	x, node = random.choice(list(id2node.items()))
    # 	return f"<{node['class_name']}> ({x})"
    #
    #
    # def get_action(self, obs, text):
    # 	if '[open]' in text or '[close]' in text or '[grab]' in text or '[walktowards]' in text:
    # 		return f"[{text.split(']')[0].split('[')[-1]}] {self.get_obj(obs, text)}"
    # 	elif 'putback' in text or 'putin' in text:
    # 		obj1 = self.get_obj(obs, text)
    # 		obj2 = self.get_obj(obs, text, 2)
    # 		return f"[{text.split(']')[0].split('[')[-1]}] {obj1} {obj2}"

    def parse_answer(self, available_actions, text):
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
                return action
        print("WARNING! Fuzzy match!")
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act, name, id = action.split(' ')
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions) if len(available_actions) > 0 else "[waiting]"  ##may cause exception

    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
        sss = {}
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"<{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        if len(satisfied) == 0:
            s = ""
        else:
            s = f"{'I' if self.single else 'We'}'ve already found and put "
            s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
            s += ' ' + self.goal_location_with_r + '. '

        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
        ### opponent modeling
        if not self.single:
            ss = ""
            if len(opponent_grabbed_objects) == 0:
                ss += "nothing. "
            else:
                ss += f"<{opponent_grabbed_objects[0]['class_name']}> ({opponent_grabbed_objects[0]['id']}). "
                if len(opponent_grabbed_objects) == 2:
                    ss = ss[:-2] + f" and <{opponent_grabbed_objects[1]['class_name']}> ({opponent_grabbed_objects[1]['id']}). "
            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "

        return s


    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        [send_message] <"">
        """
        available_plans = []
        if self.communication and message is not None:
            available_plans.append(f"[send_message] <{message}>")
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")
        
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

            
    def run(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):
        info = {}
        # goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history)
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        message = None

        if self.communication:
            prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
            if not action_history[-1].startswith('[send_message]'):
                gen_prompt = self.generator_prompt_template.replace('$GOAL$', self.goal_desc)
                gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
                gen_prompt = gen_prompt.replace('$ACTION_HISTORY$', action_history_desc)
                gen_prompt = gen_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
                gen_prompt = gen_prompt + f"\n{self.agent_name}:"
                system_prompt = "Just output the message content without any additional analysis, quotes or reasons. Just output the message. "
                chat_prompt = [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": gen_prompt}]
                outputs, usage= self.generator(chat_prompt if self.chat else gen_prompt, self.sampling_params)
                self.comm_tokens += usage
                self.total_cost += usage #COBEL 这里的usage等于token
                message = outputs[0]
                info['message_generator_prompt'] = gen_prompt
                info['message_generator_outputs'] = outputs
                info['message_generator_usage'] = usage
                if self.debug:
                    print(f"message_generator_prompt:\n{gen_prompt}")
                    print(f"message_generator_outputs:\n{message}")

        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored)
        if num == 0 or (message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                     "plan": None})
            return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        if self.cot:
            prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            info['cot_outputs'] = outputs
            info['cot_usage'] = usage
            if self.debug:
                print(f"cot_output:\n{output}")
            chat_prompt = [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user", "content": "Answer with only one best next action. So the answer is"}]
            normal_prompt = prompt + output + ' So the answer is'
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            info['output_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            system_prompt = "Don't output any additional analysis and reasons, just output the answer chosen from available actions. A standard response is: Answer:A. go to a room."
            chat_prompt = [{"role": "system", "content": system_prompt},
                           {"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            info['cot_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
        plan = self.parse_answer(available_plans_list, output)
        if self.debug:
            print(f"plan: {plan}\n")
        info.update({"num_available_actions": num,
                     "prompts": prompt,
                     "outputs": outputs,
                     "plan": plan,
                     "total_cost": self.total_cost})
        return plan, info



    
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

    def init_beliefs(self, init_challenge_descs, goal_objects):
        #TASK DESCRIPTION
        #PROGRESS
        #OPPO PROGRESS
        #BELIEF SYMBOLIC LANGUAGE
        #GOAL
        # room_des = ""
        # for room in rooms:
        #     room_des += f"{room} is explored None. "

        self.goal_desc = self.goal2description(goal_objects)
        prompt = (
            self.cobel_prompts_df["prompt"][8]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$GOAL$", self.goal_desc)
            .replace("$MY_PROGRESS$", init_challenge_descs[self.agent_id])
            .replace("$OPPO_PROGRESS$", init_challenge_descs[1- self.agent_id])
            .replace("$BELIEF_RULES$", self.belief_rules)
            .replace("$LANGUAGE$", self.belief_symbolic_language)
        )
        system_prompt = "You MUST follow the output format strictly.Format: $AGENT_NAME$ knows:\n$OPPO_NAME$ knows:\nzero order beliefs: GENERATE CONTENT\nfirst order beliefs:GENERATE CONTENT"
        chat_prompt = [{"role":"system", "content":system_prompt},{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        
        # 记录token消耗
        method_name = "init_beliefs"
        # 使用usage.prompt_tokens和usage.completion_tokens
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1
        
        pattern_zero = r'zero.*?:\s*(.*?)(?=first.*?:|$)'
        pattern_first = r'first.*?:\s*(.*)'
        zero_match = re.search(pattern_zero, output[0], re.IGNORECASE | re.DOTALL)
        first_match = re.search(pattern_first, output[0], re.IGNORECASE | re.DOTALL)
        if not zero_match or not first_match:
            raise ValueError("Failed to extract beliefs from output.")
        init_zero_order_beliefs = zero_match.group(1).strip()
        init_first_order_beliefs = first_match.group(1).strip()

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            # print(f"=========init_beliefs=============: \nzero:{init_zero_order_beliefs}\nfirst:{init_first_order_beliefs}")
            print(f"=========init_beliefs=============: \n{output[0]}")
                  
        return init_zero_order_beliefs, init_first_order_beliefs

    #COBEL-zhimin
    def update_beliefs(self, old_zero_order_beliefs,old_first_order_beliefs,visual_observation, message, oppo_obs):
        """
        更新信念状态

        参数:
            zero_order_beliefs: 零阶信念
            first_orderbeliefs: 一阶信念
            belief_rules: 信念规则

        返回:
            更新后的信念状态
        """
        pattern_zero = r"zero order belief rules:\s*(.*?)\s*(?=first order belief rules:)"
        match_zero = re.search(pattern_zero, self.belief_rules, re.IGNORECASE | re.DOTALL)
        if not match_zero:
            raise ValueError("Failed to extract updated beliefs from output.")
        zero_order_belief_rules = match_zero.group(1).strip()

        pattern_first = r"first order belief rules:\s*(.*)"
        match_first = re.search(pattern_first, self.belief_rules, re.IGNORECASE | re.DOTALL)
        if not match_first:
            raise ValueError("Failed to extract updated beliefs from output.")
        first_order_belief_rules = match_first.group(1).strip()
        
        if not (message == "None" and oppo_obs is None):
            if message == "None":
                message = "No message received."
            if oppo_obs is None:
                oppo_obs = "No new observation."
            #first
            
            prompt = (
                self.cobel_prompts_df["prompt"][1]
                .replace("$AGENT_NAME$", self.agent_name)
                .replace("$OPPO_NAME$", self.oppo_name)
                .replace("$BELIEFS$", old_first_order_beliefs)
                .replace("$MESSAGE$", message)
                .replace("$VISUAL_OBSERVATION$", oppo_obs)
                .replace("$RULE$", first_order_belief_rules)
            )
            chat_prompt = [{"role": "user", "content": prompt}]
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

            pattern_first = r'updated beliefs:\s*(.*)'
            first_match = re.search(pattern_first, first_output[0], re.IGNORECASE | re.DOTALL)
            if not first_match:
                raise ValueError("Failed to extract beliefs from output.")
            first_order_beliefs = first_match.group(1).strip()
            if self.belief_debug:
                print(f"=========prompt===========: \n{prompt}")
                print(f"=========updated_first_beliefs=============: \nfirst:{first_order_beliefs}")
        else:
            first_order_beliefs = old_first_order_beliefs
            if self.belief_debug:
                print(f"=========no first update==========: \n")
            

        prompt = (
            self.cobel_prompts_df["prompt"][0]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$BELIEFS$", old_zero_order_beliefs)
            .replace("$MESSAGE$", message)
            .replace("$VISUAL_OBSERVATION$", visual_observation)
            .replace("$RULE$", zero_order_belief_rules)
        )
        #zero
        chat_prompt = [{"role": "user", "content": prompt}]
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
        pattern_zero = r'updated beliefs:\s*(.*)'
        
        zero_match = re.search(pattern_zero, zero_output[0], re.IGNORECASE | re.DOTALL)
        
        if not zero_match:
            raise ValueError("Failed to extract beliefs from output.")
        zero_order_beliefs = zero_match.group(1).strip()
        

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========updated_zero_beliefs=============: \nzero:{zero_order_beliefs}")
        return zero_order_beliefs, first_order_beliefs

    #COBEL-zhimin
    def update_beliefs_in_one(self, old_zero_order_beliefs,old_first_order_beliefs,visual_observation, message):
        """
        更新信念状态

        参数:
            zero_order_beliefs: 零阶信念
            first_orderbeliefs: 一阶信念
            belief_rules: 信念规则

        返回:
            更新后的信念状态
        """

        old_beliefs = "old zero order beliefs\n"+ old_zero_order_beliefs + "\n" + "old first order beliefs\n" + old_first_order_beliefs

        
        
        system_prompt = "You are an expert skilled at extracting and analyzing the distinct information known by different individuals, drawn separately from multi-party dialogues and personal self-descriptions. You must precisely differentiate the information accessible to each participant and present it in a Theory of Mind Reasoning, clearly delineating each person’s cognitive boundaries, sources of information, and their respective reasoning pathways."
        prompt = (
            self.cobel_prompts_df["prompt"][0]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$BELIEFS$", old_beliefs)
            .replace("$MESSAGE$", message)
            .replace("$VISUAL_OBSERVATION$", visual_observation)
            .replace("$RULE$", self.belief_rules)
        )
        #zero
        chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
        output, usage = self.generator(
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
        pattern_zero = r"updated zero order beliefs:\s*(.*?)\s*(?=updated first order beliefs:)"
        zero_match = re.search(pattern_zero, output[0], re.IGNORECASE | re.DOTALL)
        
        if not zero_match:
            raise ValueError("Failed to extract beliefs from output.")
        zero_order_beliefs = zero_match.group(1).strip()

        pattern_first = r'updated first order beliefs:\s*(.*)'
        first_match = re.search(pattern_first, output[0], re.IGNORECASE | re.DOTALL)
        
        if not first_match:
            raise ValueError("Failed to extract beliefs from output.")
        first_order_beliefs = first_match.group(1).strip()
        

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========updated_in_one=============: \n {output[0]}")
        return zero_order_beliefs, first_order_beliefs

    #COBEL-zhimin
    def prediction_first_order(self, first_order_beliefs):
        if first_order_beliefs == "None":
            first_order_beliefs = "Know nothing about opponent's belief."
        prompt = (
            self.cobel_prompts_df["prompt"][2]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$FIRST_ORDER_BELIEFS$", first_order_beliefs)
            .replace("$GOAL$", self.goal_desc)
            .replace("$LANGUAGE$", self.belief_symbolic_language)
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
        
        # episode_logger.info(
        #     f"\n{self.agent_name}predict_first_order:\n{output[0]}"
        # )
        return reason, opponent_subplans
    
    def prediction_zero_order(self, zero_order_beliefs):
        prompt = (
            self.cobel_prompts_df["prompt"][3]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$ZERO_ORDER_BELIEFS$", zero_order_beliefs)
            .replace("$GOAL$", self.goal_desc)
            .replace("$LANGUAGE$", self.belief_symbolic_language)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
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
    def coordination_aware(self, first_order_beliefs, zero_order_beliefs, my_subplan, opponent_subplans):
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
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$FIRST_ORDER_BELIEFS$", first_order_beliefs)
            .replace("$ZERO_ORDER_BELIEFS$", zero_order_beliefs)
            .replace("$MY_SUBPLAN$", my_subplan)
            .replace("$OPPO_SUBPLAN$", opponent_subplans)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasons:\nanswer:\ndifferent beliefs:\n, with no bold format."
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
        
        pattern_reason = r'reasons:\s*(.*?)(?=' + re.escape("answer:") + r'|$)'
        match_reason = re.search(pattern_reason, output[0], re.IGNORECASE | re.DOTALL)
        if not match_reason:
            raise ValueError("Failed to extract reason from output.")
        reason = match_reason.group(1).strip()


        pattern_answer = r'answer:\s*(.*?)(?=' + re.escape("different beliefs:") + r'|$)'
        match_answer = re.search(pattern_answer, output[0], re.IGNORECASE | re.DOTALL)
        if not match_answer:
            raise ValueError("Failed to extract answer from output.")
        answer = match_answer.group(1).strip()

        if "NO" not in answer.upper():

            
            pattern_difference = rf"different beliefs:\s*(.*)"
            match_difference = re.search(pattern_difference, output[0], re.IGNORECASE | re.DOTALL)
            if not match_difference:
                raise ValueError("Failed to extract difference from output.")
            difference = match_difference.group(1).strip()
        else:
            answer = "NO MISCOORDINATION"
            reason = "None"
            difference = "None"

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========coordination_aware=============: \n{output[0]}")
        return answer, reason, difference

    #COBEL  -shaokang - zhimin update 因为原本是通过run规划，所以会在run传入很多信息来更新状态来提供available action


    def comm(self, difference, my_subplan):
        prompt = (
            self.cobel_prompts_df["prompt"][5]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$DIFFERENCE$", difference)
            .replace("$MY_SUBPLAN$", my_subplan)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                )
        
        pattern_message = rf"message:\s*(.*)"
        match_message = re.search(pattern_message, output[0], re.IGNORECASE | re.DOTALL)
        if not match_message:
            raise ValueError("Failed to extract message from output.")
        message = match_message.group(1).strip()

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========message=============: \n{output[0]}")
        return message
    


    def passive_coordination(self, oppo_message,zero_order_beliefs, my_subplan):
        prompt = (
            self.cobel_prompts_df["prompt"][6]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MESSAGE$", oppo_message)
            .replace("$ZERO_ORDER_BELIEFS$", zero_order_beliefs)
            .replace("$SUBPLAN$", my_subplan)
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                )
        
        pattern_answer = r'answer:\s*(.*?)(?=' + re.escape("reasons:") + r'|$)'
        match_answer = re.search(pattern_answer, output[0], re.IGNORECASE | re.DOTALL)
        if not match_answer:
            raise ValueError("Failed to extract answer from output.")
        answer = match_answer.group(1).strip()

        pattern_reasons = r'reasons:\s*(.*?)(?=' + re.escape("new subplan:") + r'|$)'
        match_reasons = re.search(pattern_reasons, output[0], re.IGNORECASE | re.DOTALL)
        if not match_reasons:
            raise ValueError("Failed to extract reasons from output.")
        reasons = match_reasons.group(1).strip()


        pattern_subplan = rf"new subplan:\s*(.*)"
        match_subplan = re.search(pattern_subplan, output[0], re.IGNORECASE | re.DOTALL)
        if not match_subplan:
            raise ValueError("Failed to extract subplan from output.")
        subplan = match_subplan.group(1).strip()
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"========passive coordination=============: \n{output[0]}")
        return answer, reasons, subplan


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
        chat_prompt = [{'role':'user','content':prompt}]
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
        
        episode_logger.info(
            f"\n{self.agent_name}intuitive_planning:\n{output[0]}"
        )

        pattern_answer = rf"answer:\s*(.*)"
        match_answer = re.search(pattern_answer, output[0], re.IGNORECASE | re.DOTALL)
        if not match_answer:
            raise ValueError("Failed to extract opponent answers from output.")
        answer = match_answer.group(1).strip()

        #TODO: COBEL parse checking the efficiency
        plan, flags = self.parse_answer(available_plans_list, answer)
        return plan

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
        episode_logger = None
    ):
        info = {}
        print("current_step", current_step)
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room

        #COBEL - zhimin 这里会涉及初始化
        progress_desc = self.progress2text(
            current_step, satisfied, opponent_grabbed_objects, opponent_last_room
        )

        return progress_desc
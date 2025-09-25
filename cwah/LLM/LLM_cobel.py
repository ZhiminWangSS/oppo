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
import re

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
        self.belief_debug = False

        self.cobel_prompts_df = pd.read_csv(self.prompt_template_path)
        with open("./LLM/rules_cwah_no_conf.txt", "r", encoding="utf-8") as f:
            self.belief_rules = f.read()
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
        for call_name in ['total',"update_beliefs","prediction_zero_order","prediction_first_order","intuitive_planning","cooradination_aware","communication"]:
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
        elif self.source == "aliyun":
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
                usage = [0,0]
                if source == 'openai' or source == 'aliyun':
                    for attempt in range(6):
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
                                self.api_num += 1
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
                            return generated_samples, usage
                        except OpenAIError as e:
                            if attempt == 5:
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
        for call_name in ['total',"update_beliefs","prediction_zero_order","prediction_first_order","intuitive_planning","cooradination_aware","communication"]:
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






    def update_beliefs(self, received_messages, dialogues,agent_names):
        
        
        updated_zero_order_beliefs = {}
        updated_first_order_beliefs = {}
        oppo_subplans = {}
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
        
        #first
        if dialogues != {}:
            print("======update belief======")
            for agent_id, agent_name in enumerate(agent_names):
                if agent_name == self.agent_name:
                    continue
                my_oppo_dialogue = ""
                for agent_name1 in agent_names:
                    if agent_name1 in dialogues.keys(): 
                        my_oppo_dialogue += f"{agent_name1}: {dialogues[agent_name1]}"
                    prompt = (
                        self.cobel_prompts_df["prompt"][0]
                        .replace("$AGENT_NAME$", self.agent_name)
                        .replace("$OPPO_NAME$", agent_name) #oppo
                        .replace("$MESSAGE$", my_oppo_dialogue)
                        .replace("$RULE$", first_order_belief_rules)
                    )
                    system_prompt = "You MUST answer strictly in this format:\n$OPPO_NAME$ knows:\nfirst order beliefs\n$OPPO_NAME$'s plan:".replace("$OPPO_NAME$", self.oppo_name)
                    chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
                    # chat_prompt = [{"role": "user", "content": prompt}]
                    first_output, usage = self.generator(
                                chat_prompt, self.sampling_params
                            ) # usage token cost
            
                    method_name = "update_beliefs"
               
                    prompt_tokens = usage[0]
                    completion_tokens = usage[1]
                    self.token_stats[method_name]["prompt"] += prompt_tokens
                    self.token_stats[method_name]["completion"] += completion_tokens
                    self.token_stats[method_name]["call_counts"] += 1



                    if self.belief_debug:
                        print(f"=========prompt===========: \n{prompt}")
                        print(f"=========updated_first_beliefs=============: \nfirst:{first_output[0]}")

                    pattern_first = rf"first order beliefs:\s*(.*?)\s*(?={re.escape(self.oppo_name)}'s plan:)"
                    first_match = re.search(pattern_first, first_output[0], re.IGNORECASE | re.DOTALL)
                    if not first_match:
                        continue
                    else:
                        updated_first_order_beliefs.update({agent_name:first_match.group(1).strip()})

        else:
            updated_first_order_beliefs = {}
            if self.belief_debug:
                print(f"=========no first update==========: \n")
            
        if received_messages != {}:
            print("======update belief======")
            for agent_name in agent_names:
                if agent_name in received_messages.keys():
                    prompt = (
                        self.cobel_prompts_df["prompt"][1]
                        .replace("$AGENT_NAME$", self.agent_name)
                        .replace("$OPPO_NAME$", agent_name)
                        .replace("$MESSAGE$", received_messages[agent_name])
                        .replace("$RULE$", zero_order_belief_rules)
                    )
                    #zero
                    system_prompt = "You MUST answer strictly in this format:\n$AGENT_NAME$ knows:\nzero order beliefs:\n$OPPO_NAME$'s plan:".replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
                    chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]

                    zero_output, usage = self.generator(
                                chat_prompt, self.sampling_params 
                            ) # usage token cost
                    
               
                    method_name = "update_beliefs"
  
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
                        print(f"=========prompt===========: \n{prompt}")
                        print(f"=========updated_zero_beliefs=============: \nzero:{zero_output[0]}")
                    
                    if not zero_match or not plan_match:
                        continue
                    else:
                        updated_zero_order_beliefs.update({agent_name:zero_match.group(1).strip()})
                        oppo_subplans.update({agent_name:plan_match.group(1).strip()})
        else:
            updated_zero_order_beliefs = {}
            oppo_subplans = {}
            if self.belief_debug:
                print(f"=========no zero update==========: \n")
        return updated_zero_order_beliefs, updated_first_order_beliefs , oppo_subplans

    def prediction_zero_order(self,my_progress):
        prompt = (
            self.cobel_prompts_df["prompt"][3]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MY_PROGRESS$", my_progress)
            .replace("$GOAL$", self.goal_desc)
        )
        print(self.goal_desc)
        system_prompt = "You MUST follow the output format strictly.Format: reasoning:\nsubplan:"
        chat_prompt = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        # chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        

        method_name = "prediction_zero_order"
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1

    
        pattern_beliefs = r'reasoning:\s*(.*?)(?=' + re.escape("subplan:") + r'|$)'
        match_beliefs = re.search(pattern_beliefs, output[0], re.IGNORECASE | re.DOTALL)
        if not match_beliefs:
            raise ValueError("Failed to extract updated beliefs from output.")
        reason = match_beliefs.group(1).strip()

       
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


    def prediction_first_order(self,oppo_progress):
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

        method_name = "prediction_first_order"
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1

        pattern_beliefs = r'reasoning:\s*(.*?)(?=' + re.escape(f"subplans:") + r'|$)'
        match_beliefs = re.search(pattern_beliefs, output[0], re.IGNORECASE | re.DOTALL)
        if not match_beliefs:
            raise ValueError("Failed to extract updated beliefs from output.")
        reason = match_beliefs.group(1).strip()

      
        pattern_subplan = rf"subplans:\s*(.*)"
        match_subplan = re.search(pattern_subplan, output[0], re.IGNORECASE | re.DOTALL)
        if not match_subplan:
            raise ValueError("Failed to extract opponent subplans from output.")
        opponent_subplans = match_subplan.group(1).strip()
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========predict_first=============: \n{output[0]}")


        return reason, opponent_subplans



    def coordination_aware(self,my_progress, oppo_progress, my_subplan, opponent_subplans):
        
        oppo_progress_str = ""
        for agent_name, progress in oppo_progress.items():
            oppo_progress_str += f"{agent_name}'s progress: {progress}\n"
        oppo_progress_str = oppo_progress_str.strip()
        opponent_subplans_str = ""
        for agent_name, subplan in opponent_subplans.items():
            opponent_subplans_str += f"{agent_name}'s subplan: {subplan}\n"
        opponent_subplans_str = opponent_subplans_str.strip()
        prompt = (
            self.cobel_prompts_df["prompt"][5]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MY_PROPGRESS$", my_progress)
            .replace("$OPPO_PROGRESS$", oppo_progress_str)
            .replace("$MY_SUBPLAN$", my_subplan)
            .replace("$OPPO_SUBPLAN$", opponent_subplans_str)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasons:\nanswer:\nmisaligned information:\n"
        chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                )   
        
        
        method_name = "cooradination_aware"
    
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1
        

        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
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



    def passive_prediction_zero_order(self, my_progress, oppo_subplan):
        
        oppo_subplan_str = ""

        for agent_name,agent_subplan in oppo_subplan.items():
            oppo_subplan_str += f"{agent_name}'s subplan: "
            oppo_subplan_str += agent_subplan

        prompt = (
            self.cobel_prompts_df["prompt"][4]
            .replace("$AGENT_NAME$", self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$MY_PROGRESS$", my_progress)
            .replace("$OPPO_SUBPLAN$", oppo_subplan_str)
            .replace("$GOAL$", self.goal_desc)
        )
        system_prompt = "You MUST follow the output format strictly.Format: reasoning:\nsubplan:"
        chat_prompt = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        # chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(
                    chat_prompt, self.sampling_params
                ) # usage token cost
        
     
        method_name = "prediction_zero_order"
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1

       
        pattern_beliefs = r'reasoning:\s*(.*?)(?=' + re.escape("subplan:") + r'|$)'
        match_beliefs = re.search(pattern_beliefs, output[0], re.IGNORECASE | re.DOTALL)
        if not match_beliefs:
            raise ValueError("Failed to extract updated beliefs from output.")
        reason = match_beliefs.group(1).strip()

       
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
        
        method_name = "communication"
       
        prompt_tokens = usage[0]
        completion_tokens = usage[1]
        self.token_stats[method_name]["prompt"] += prompt_tokens
        self.token_stats[method_name]["completion"] += completion_tokens
        self.token_stats[method_name]["call_counts"] += 1
        
        
        message = output[0]
        if self.belief_debug:
            print(f"=========prompt===========: \n{prompt}")
            print(f"=========message=============: \n{output[0]}")
        return message
    


    def intuitive_planning(self,
                           my_subplan,
                           action_history,
                           my_progress,
                           available_plans,
                           available_plans_list,
                           episode_logger = None, 
                           
                           ):

    
        prompt = (
            self.cobel_prompts_df["prompt"][7]
            .replace('$AGENT_NAME$',self.agent_name)
            .replace("$OPPO_NAME$", self.oppo_name)
            .replace("$GOAL$", self.goal_desc)
            .replace('$MY_SUBPLAN$',my_subplan)
            .replace('$PREVIOUS_ACTIONS$',action_history)
            .replace('$PROGRESS$',my_progress)
            .replace('$ACTION_LIST$',available_plans)
        )

        system_prompt = "You MUST follow the output format strictly.Format: answer:\nreasons:\nanswer:"
        chat_prompt = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
        # chat_prompt = [{'role':'user','content':prompt}]
        output,usage = self.generator(
            chat_prompt,self.sampling_params
        )
        
       
        method_name = "intuitive_planning"
    
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
        plan = self.parse_answer(available_plans_list, answer)
        return plan

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
        if 'DONE' in text:
            action = 'SUBPLAN DONE'
            return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text or f"{option}" in text:
                return action
        print("WARNING! Fuzzy match!")
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            if action ==  'SUBPLAN DONE':
                act = 'SUBPLAN DONE'
                name = 'SUBPLAN'
                id = 'DONE'
            else:
                act, name, id = action.split(' ')
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions) if len(available_actions) > 0 else "[waiting]"  ##may cause exception



    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
        sss = {}
        print("ungrab",ungrabbed_objects)
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
            if opponent_last_room['class_name'] is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room['class_name'] == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room['class_name']}, {self.oppo_pronoun} was holding {ss}"



        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "
        for room,state in room_explored.items():
            if state == None:
                s += f"I've explored none of the {room}. "
            elif state == 'all':
                s += f"I've explored all of the {room}. "
        return s

    def oppo_progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
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
            if opponent_last_room['class_name'] is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room['class_name'] == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room['class_name']}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "
        for room,state in room_explored.items():
            if state == None:
                s += f"I've explored none of the {room}. "
            elif state == 'all':
                s += f"I've explored all of the {room}. "
        return s

    def get_available_plans_cobel(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        [send_message] <"">
        """
        available_plans = []
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

        available_plans.append("SUBPLAN DONE")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

            
    def get_my_progress(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):
        #opponent_grabbed_objects, opponent_last_room -》 {name:[]}
        # goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        return progress_desc

    def get_available_plans(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):

        # goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
        message = None
        available_plans, num, available_plans_list = self.get_available_plans_cobel(grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored)
        return available_plans, num, available_plans_list

    def get_oppo_progress(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):

        # goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
        progress_desc = self.oppo_progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        

        return progress_desc
    

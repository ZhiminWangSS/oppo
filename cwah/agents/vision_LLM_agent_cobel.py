import copy
import math
from pathlib import Path

import numpy as np
import re
from . import vision_pipeline
from LLM.LLM_cobel import *
# from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
# from maskrcnn_benchmark.config import cfg

class vision_LLM_agent_cobel:
    """
    LLM agent class
    """
    def __init__(self, agent_id, char_index, args):
        self.vision_pipeline = None
        self.config = vision_pipeline.agent_vision_config(
            agent_type = 'LLM_vision',
            char_index=char_index,
            agent_id=agent_id,
            )
        self.debug = args.debug
        self.agent_type = 'LLM_vision'
        self.agent_names = ["Zero", "Alice", "Bob"]
        self.work_agents = {"Zero":0, "Alice":1, "Bob":1}
        self.agent_id = agent_id
        self.opponent_agent_id = 3 - agent_id
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM_cobel(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}
        self.unsatisfied = {}
        self.steps = 0
        self.location = None
        self.last_location = None
        self.plan = None
        self.stuck = 0
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None
        self.opponent_grabbed_objects = []
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id2node = {}
        self.id_inside_room = {}
        self.done_time = 0 
        self.satisfied = []
        self.reachable_objects = []
        self.con =False 
        if 'qwen' in self.lm_id:
            self.con = False
        self.room_explored = {
            "livingroom": False,
            "kitchen": False,
            "bedroom": False,
            "bathroom": False,
        }
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.task_id = 0
        self.record_dir = os.path.join(args.record_dir, str(self.task_id),'image', self.agent_names[self.agent_id])
        Path(self.record_dir).mkdir(parents=True, exist_ok=True)
        self.obs = None
        self.rotated = 0
        self.comm_chars = 0
        self.comm_counts = 0

        #Cobel:
        # = con_per_room
        #zero + first
        self.team_unchecked_con = {
            _:{
            "livingroom": [],
            "kitchen": [],
            "bedroom": [],
            "bathroom": [],
        } for _ in self.agent_names
        }
    
        # = obj per room
        # 为每个智能体创建一个字典列表来跟踪未抓取的物体
        #zero + first
        self.team_ungrasped_obj = {
            _:{
            "livingroom": [],
            "kitchen": [],
            "bedroom": [],
            "bathroom": [],
        } for _ in self.agent_names
        }

        # grasped objects {'id': None,'name':None}
        #team [[{hand},{hand2}]]

        self.team_explored_rooms = { #探索了就改成"all"
            _ : {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
           } for _ in self.agent_names
        }

        self.team_subplan = { _ : None for _ in self.agent_names}

        

        self.team_grasped_obj = {
            _ : {
                _ : [] for _ in self.agent_names
            } for _ in self.agent_names
        }


        self.team_current_room = {
            _ : {
                _ : None for _ in self.agent_names
            } for _ in self.agent_names
        }

        
        self.message_received = {}
        self.dialogue = {}
        self.action_history_max_length = 3
        self.my_subplan = None
        self.max_message_time = 2

    def goexplore(self):
        target_room_id = int(self.plan.split(' ')[-1][1:-1])
        if self.rotated == 12:
            assert self.current_room['id'] == target_room_id
            self.rotated = 0
            self.plan = None
            return None
        if self.rotated > 0:
            self.rotated += 1
            return '[TurnLeft]'
        if self.current_room['id'] != target_room_id or (math.dist(self.location, self.last_location) > 0.3 and self.stuck < 10): # still not at the center
            if self.debug:
                print(f"{self.agent_names[self.agent_id]}'s current location: {self.location}")
            return self.plan.replace('[goexplore]', '[walktowards]')
        else:
            self.rotated = 1
            return '[TurnLeft]'


    def gocheck(self):
        assert len(self.grabbed_objects) < 2 # must have at least one free hands
        target_container_id = int(self.plan.split(' ')[-1][1:-1])
        target_container_name = self.plan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]
        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        if self.location == self.last_location and not target_container_id in self.vision_pipeline.see_this_step and f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            if target_container in self.unchecked_containers[self.current_room['class_name']]:
                self.unchecked_containers[self.current_room['class_name']].remove(target_container)
            target_container['states'].append('OPEN') # must already be opened
            if 'CLOSED' in target_container['states']: target_container['states'].remove('CLOSED')

        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        if (target_container_id in self.vision_pipeline.see_this_step or self.location == self.last_location) and f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan.replace('[gocheck]', '[open]')
        else:
            return self.plan.replace('[gocheck]', '[walktowards]')


    def gograb(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        target_object_name = self.plan.split(' ')[1]
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan = None
            return None
        assert len(self.grabbed_objects) < 2 # must have at least one free hands

        target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.plan = None
            return None
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan.replace('[gograb]', '[grab]')
        else:
            return self.plan.replace('[gograb]', '[walktowards]')

    def goput(self):
        # if len(self.progress['goal_location_room']) > 1: # should be ruled out
        if len(self.grabbed_objects) == 0:
            self.plan = None
            print("手上没东西")
            return None
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
            if self.current_room['class_name'] != target_room_name:
                return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            if len(self.id_inside_room[self.goal_location_id]) > 1:
                self.plan = f"[goexplore] <{target_room_name}> ({self.roomname2id[target_room_name]})"
                self.room_explored[target_room_name] = True
                print("无返回值")
                return None
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"

        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and 'CLOSED' in y['states']:
                return self.plan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'
        x = self.id2node[self.grabbed_objects[0]]
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"


    
    # def LLM_plan(self):
    #     if len(self.grabbed_objects) == 2:
    #         return f"[goput] {self.goal_location}", {}

    #     return self.LLM.run(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id], self.room_explored)
    def get_my_progress(self):
        
        # return self.LLM.get_my_progress(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id])
        return self.LLM.get_my_progress(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.team_unchecked_con[self.agent_names[self.agent_id]], self.team_ungrasped_obj[self.agent_names[self.agent_id]], self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.team_grasped_obj[self.agent_names[self.agent_id]][self.agent_names[self.opponent_agent_id]], self.team_current_room[self.agent_names[self.agent_id]][self.agent_names[self.opponent_agent_id]],self.room_explored)
    
    def get_oppo_progress(self): #zhe
        # print("=========================对手progress==")
        # print(self.team_current_room)
        # return self.LLM.get_my_progress(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id])
        return self.LLM.get_oppo_progress(self.team_current_room[self.agent_names[self.opponent_agent_id]][self.agent_names[self.opponent_agent_id]], self.team_grasped_obj[self.agent_names[self.opponent_agent_id]][self.agent_names[self.opponent_agent_id]], self.satisfied, self.team_unchecked_con[self.agent_names[self.opponent_agent_id]], self.team_ungrasped_obj[self.agent_names[self.opponent_agent_id]], self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.team_grasped_obj[self.agent_names[self.opponent_agent_id]][self.agent_names[self.agent_id]], self.team_current_room[self.agent_names[self.opponent_agent_id]][self.agent_names[self.agent_id]],self.team_explored_rooms[self.agent_names[self.opponent_agent_id]])



    def get_available_plan(self):

        
        return self.LLM.get_available_plans(
            self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id], self.room_explored
        )
        # return self.LLM.get_available_plans(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.team_unchecked_con[self.agent_names[self.agent_id]], self.team_ungrasped_obj[self.agent_names[self.agent_id]], self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.team_grasped_obj[self.agent_names[self.agent_id]][self.agent_names[self.opponent_agent_id]], self.team_current_room[self.agent_names[self.agent_id]][self.agent_names[self.opponent_agent_id]],self.team_explored_rooms[self.agent_names[self.agent_id]])
    

    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
                    # if self.debug:
                    # 	print(satisfied)
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied



    def intuitive_planning(self):

        my_progress = self.get_my_progress()
        available_plans, num, available_plans_list = self.get_available_plan()
        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} action_with_message_history:{self.action_history_w_mes}")
        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} action_history:{self.action_history}\nmy_subplan:{self.my_subplan}")
        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} action_with_message_history:{self.action_history_w_mes}")
        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} action_history:{self.action_history}")
        action_history_desc = ", ".join(self.action_history)
        print(available_plans)
        plan = self.LLM.intuitive_planning(self.my_subplan,
                                           action_history_desc,
                                           my_progress,
                                           available_plans,
                                           available_plans_list,
                                          self.episode_logger)
        
        return plan

    def get_action_cobel(self, obs, goal):
        """
        :param obs: {
                'bgr': bgr_images,  # 1 * w * h * 3
                'depth': depth_images,  # 1 * w * h * 3
                'camera_info': camera_info,
                'seg_info': self.id_map[agent_id], 1 * w * h
                'room_info': self.room_info,
                'messages': self.message_said,
                'location': agent_location,  # [x, y, z]
                **self.agent_env_api(agent_id),
                **self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
                        agent_id + 1))),
            }
        :param goal:{predicate:[count, True, 2]}
        :return:
        """
        self.vision_pipeline.deal_with_obs(obs, self.last_action)
        symbolic_obs = self.vision_pipeline.get_graph()

        if self.communication:
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i + 1]}: {obs['messages'][i]}")
                    self.dialogue.update(
                        {self.agent_names[i + 1]: obs['messages'][i]}
                        )
                    if (i+1) != self.agent_id:
                        self.message_received.update(
                            {self.agent_names[i + 1]: obs['messages'][i]} #改成字典
                        )
        self.obs = obs
        # print(obs)
        self.location = obs['location']
        self.location[1] = 1 # fix env bug
        self.current_room = self.vision_pipeline.object_info[obs['current_room']]
        # unexplored_room = False
        # if self.unchecked_containers[self.current_room['class_name']] is None:
        # 	unexplored_room = True

        satisfied, unsatisfied = self.check_progress(symbolic_obs, goal)
        self.observe_new = False
        print(f"satisfied: {satisfied}")
        if self.satisfied != satisfied:
            self.observe_new = True
        # print(f"satisfied: {satisfied}")
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied

        if self.debug:
            # colorids = np.stack(((ids % 10) * 10, ((ids // 10) % 10) * 10, ((ids // 100) % 10) * 10), axis=3)
            import cv2
            for i in range(len(obs['camera_info'])):
                cv2.imwrite(os.path.join(self.record_dir, f"{self.steps:03}_img.png"), np.rot90(obs['bgr'][0], axes=(0,1)))
            # print(f"symbolic_graph:\n{symbolic_obs['nodes']}\n{json.dumps(symbolic_obs['edges'], indent=4)}")

        self.grabbed_objects = []
        self.team_grasped_obj[self.agent_names[self.agent_id]][self.agent_names[self.agent_id]] = []
        opponent_grabbed_objects = []
        self.reachable_objects = []
        self.id2node = {x['id']: x for x in symbolic_obs['nodes']}
        for e in symbolic_obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.opponent_agent_id:
                if r == 'INSIDE':
                    self.team_current_room[self.agent_names[self.agent_id]][self.agent_names[self.opponent_agent_id]] = self.id2node[y]
            if x == self.agent_id:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y] #id2node返回那个大字典
                    self.team_current_room[self.agent_names[self.agent_id]][self.agent_names[self.agent_id]] = self.id2node[y]
                    #####
                elif r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                    #######
                    self.team_grasped_obj[self.agent_names[self.agent_id]][self.agent_names[self.agent_id]].append({'id': y, 'class_name': self.id2node[y]['class_name']})
                
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
                opponent_grabbed_objects.append(self.id2node[y])
        if opponent_grabbed_objects != []:
            self.team_grasped_obj[self.agent_names[self.agent_id]][self.agent_names[self.opponent_agent_id]] = opponent_grabbed_objects
        
        unchecked_containers = []
        ungrabbed_objects = []

        for x in symbolic_obs['nodes']:
            if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x['id'] == ungrab['id']:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x['id']] = self.current_room['class_name']
            if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
                unchecked_containers.append(x)
            if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x['id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
                ungrabbed_objects.append(x)

        if self.room_explored[self.current_room['class_name']] and type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
            
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        ####
        self.team_explored_rooms[self.agent_names[self.agent_id]][self.current_room['class_name']] = 'all'
        #####
        self.team_unchecked_con[self.agent_names[self.agent_id]][self.current_room['class_name']] = unchecked_containers[:]
        #####
        self.team_ungrasped_obj[self.agent_names[self.agent_id]][self.current_room['class_name']] = ungrabbed_objects[:]
        
        if self.ungrabbed_objects[self.current_room['class_name']] != ungrabbed_objects[:]: #新物体
            print(self.ungrabbed_objects)
            self.observe_new = True
        
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': symbolic_obs,
                "obs": {
                    "location": self.location,
                    "grabbed_objects": self.grabbed_objects,
                    "opponent_grabbed_objects": self.opponent_grabbed_objects,
                    "reachable_objects": self.reachable_objects,
                    "progress": {
                        "unchecked_containers": self.unchecked_containers,
                        "ungrabbed_objects": self.ungrabbed_objects,
                        },
                    "satisfied": self.satisfied,
                    "goal_position_room": self.id_inside_room[self.goal_location_id],
                    "with_character_id": self.vision_pipeline.with_character_id,
                    "current_room": self.current_room['class_name'],
                    "see_this_step": self.vision_pipeline.see_this_step,
                    }
                }
        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects
        action = None
        LM_times = 0
        # if unexplored_room:
        # 	self.action_history.pop()
        # 	self.plan = f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"
        # 	self.action_history.append(self.plan)
        # 	self.last_location = [0, 0, 0]
        self.done_time = 0
        while action is None:

            #=========================satisfied 更新 和房间地图 ================
            #完成的物品 从所有手上删除
            for obj in self.satisfied:
                for agent_name_host, agents_graped in self.team_grasped_obj.items(): #2个人
                    for agent_name, agent_graped in agents_graped.items(): #各自对彼此
                        for hand_id, hand_obj in enumerate(agent_graped): #手
                            if obj['id'] == agent_graped[hand_id]['id']: #[{},{}]
                                self.team_grasped_obj[agent_name_host][agent_name][hand_id] = {
                                    'id': None, 'class_name':None
                                }
                        
            
            #完成的物品 从所有地图上删除
            for obj in self.satisfied:
                for agent_name_host, agent_ungraped in self.team_ungrasped_obj.items(): #2个人#各自对彼此
                    for room_name, con_list in agent_ungraped.items(): #手
                        to_remove = []
                        for idx1, obj_ungrasped in enumerate(con_list):
                            if obj_ungrasped['id'] == obj['id']:
                                to_remove.append(idx1)

                        for idx2 in sorted(to_remove, reverse=True):
                                self.team_ungrasped_obj[agent_name_host][room_name].pop(idx2)

                        
                    
            #拿到的物体从所有地图上删除 分开信念处理 grasp_zero 
            hold_obj_ids = []
            for agent_name, agent_grasp_obj in self.team_grasped_obj[self.agent_names[self.agent_id]].items():
                for hand_id, hand_obj in enumerate(agent_grasp_obj): #手循环
                    hold_obj_ids.append(hand_obj['id'])
            

            for agent_name_host, agent_ungrasped_obj in self.team_ungrasped_obj.items():
                for room_name, objs in agent_ungrasped_obj.items():
                    to_remove = []
                    for idx1, obj_ungrasped in enumerate(objs):
                            if obj_ungrasped['id'] in hold_obj_ids:
                                to_remove.append(idx1)

                    for idx2 in sorted(to_remove, reverse=True):
                                self.team_ungrasped_obj[agent_name_host][room_name].pop(idx2)

            
            
            #===========================结束=========================
            
            if (self.plan is None or self.observe_new) and (self.rotated == 0): #or new obj or message
                if self.observe_new:
                    self.subplan = None
                    self.episode_logger.info("=======新物体=======")
                    self.plan_logger.info("=======新物体=======")
                if self.plan == None:
                    print("=======没计划触发重新规划=======")
                    self.plan_logger.info("=======没计划触发重新规划=======")
                # if LM_times > 0:
                if LM_times > 10:
                    print(info)
                    break
                plan = None
                work_agents_name = []
                for agent_name,agent_state in self.work_agents.items():
                    if agent_state == 1:
                        work_agents_name.append(agent_name)
                #measurement update
                updated_zero_order_beliefs,updated_first_order_beliefs,self.opponent_subplans = self.LLM.update_beliefs(self.message_received,self.dialogue,work_agents_name)

                # print("=========updated_first_beliefs==========")
                # print(updated_first_order_beliefs)
                # print("=========updated_zero_beliefs==========")
                # print(updated_zero_order_beliefs)
                if self.con == False:
                    # print("==============不带容器更新==============")
                    if updated_first_order_beliefs != {}:
                        self.parse_belief_line_con('first',updated_first_order_beliefs)
                    if updated_zero_order_beliefs != {}:
                        self.parse_belief_line_con('zero',updated_zero_order_beliefs)
                else:
                    if updated_first_order_beliefs != {}:
                        self.parse_belief_line('first',updated_first_order_beliefs)
                    if updated_zero_order_beliefs != {}:
                        self.parse_belief_line('zero',updated_zero_order_beliefs)

                
                
                self.dialogue = {} #处理完就清空
                self.message_received = {}

                self.episode_logger.info(f"\nzero update:{updated_zero_order_beliefs}\nfirst update{updated_first_order_beliefs}")

                plan = None

                if len(self.opponent_subplans) == len(self.work_agents) - 1: #说明所有智能体计划都知道
                    my_progress = self.get_my_progress()
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_progress:{my_progress}")
                    print(f"\n{self.agent_names[self.agent_id]} my_progress:{my_progress}")
                    zero_reason, self.my_subplan = self.LLM.passive_prediction_zero_order(my_progress,self.opponent_subplans)
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_zero:{zero_reason}")
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                    self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                    print("=========被动更新==========")
                    self.plan_logger.info("=========被动更新==========")
                    print(f"{self.agent_names[self.agent_id]}: {self.my_subplan}\n")
                    # print(f"{self.agent_names[self.opponent_agent_id]}: {self.opponent_subplans}")
                    self.action_history = []
                    self.action_history_w_mes = []
                    self.opponent_subplans = None

                #===== 只在更新计划的时候走 =====
                if self.my_subplan is None or len(self.action_history) >= self.action_history_max_length:#TODO 发现新物体
                    oppo_progress = self.get_oppo_progress()
                    my_progress = self.get_my_progress()
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} oppo_progress:{oppo_progress}")
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_progress:{my_progress}")
                    print(self.goal_objects_name)
                    print(f"\n{self.agent_names[self.agent_id]} my_progress:{my_progress}")

                    #这个基本不可能到这


                    if self.opponent_subplans != {}: #说明消息发送了计划
                        zero_reason, self.my_subplan = self.LLM.passive_prediction_zero_order(my_progress,self.opponent_subplans)
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_zero:{zero_reason}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")


                    else: #就主动
                        zero_reason, self.my_subplan = self.LLM.prediction_zero_order(my_progress)
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_zero:{zero_reason}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")

                        total_progress = {}
                        for agent_name in self.agent_names:
                            if agent_name == self.agent_names[self.agent_id]:
                                continue
                            if agent_name not in self.opponent_subplans.keys() and self.work_agents[agent_name] == 1:
                                oppo_progress = self.get_oppo_progress()
                                first_reason, oppo_subplan = self.LLM.prediction_first_order(oppo_progress)
                                self.opponent_subplans.update({agent_name:oppo_subplan})
                                self.episode_logger.info(f"\n{agent_name} predict_first:{first_reason}")
                                self.episode_logger.info(f"\n{agent_name} oppo_subplan:{oppo_subplan}")
                                self.plan_logger.info(f"\n{agent_name} oppo_subplan:{oppo_subplan}")
                                total_progress.update({agent_name:oppo_progress})

                            print("=========主动更新==========")
                            self.plan_logger.info("=========主动更新==========")
                        print(f"{self.agent_names[self.agent_id]}: {self.my_subplan}\n")
                        # print(f"{self.agent_names[self.opponent_agent_id]}: {self.opponent_subplans}")
                        answer, reason, difference = self.LLM.coordination_aware(my_progress,total_progress,self.my_subplan,self.opponent_subplans)
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} answer:{answer}")
                        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} answer:{answer}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} reason:{reason}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} difference:{difference}")
                        #有必要就更新
                        if "YES" in answer.upper() and self.message_time < self.max_message_time:
                            message = self.LLM.comm(difference,self.my_subplan)
                            plan =  "[send_message]" + "<" + message + ">"
                            self.comm_counts += 1
                            self.comm_chars += len(message)
                            self.message_time += 1

                    #subplan更新后清空
                    self.action_history = [] #COBEL clean the action history
                    self.action_history_w_mes = []


                # ======subplan规划结束=======
                if plan is None:
                    plan = self.intuitive_planning()

                
                    
                if "SUBPLAN DONE" in plan: #TODO:have to program a fuzzy match in parse
                    self.my_subplan = None
                    if self.done_time > 2:
                        print("subplan done 太多次")
                        available_plans, num, available_plans_list = self.get_available_plan()
                        filtered_plans = [item for item in available_plans_list if "SUBPLAN DONE" not in item]
                        if filtered_plans == []:
                            my_progress = self.get_my_progress()
                            plan = "[send_message]" + "<" + my_progress + ">"
                        else:
                            plan = random.choice(filtered_plans)
                    else:
                        self.plan = None
                        self.done_time += 1
                        # self.plan = None #其实不需要
                        
                        continue

                

                if plan is None:  # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"

                if len(self.grabbed_objects) == 2:
                    self.subplan = None
                    print("强制搬运")
                    plan =  f"[goput] {self.goal_location}"
                unsatisfied_num = sum(self.unsatisfied.values())
                if len(self.grabbed_objects) == unsatisfied_num:
                    print("最后几个直接放")
                    self.subplan = None
                    plan =  f"[goput] {self.goal_location}"
                if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"

                self.plan_logger.info(
                            f"\n{self.agent_names[self.agent_id]}: low-level-plan:{plan}"
                        )
                
                self.episode_logger.info(
                            f"\n{self.agent_names[self.agent_id]}: low-level-plan:{plan}"
                        )

                print(f"\n{self.agent_names[self.agent_id]}: low-level-plan:{plan}")

                
                self.plan = plan
                self.action_history.append('[send_message]' if plan.startswith('[send_message]') else plan)
                # a_info.update({"steps": self.steps})
                # info.update({"LLM": a_info})
                LM_times += 1

                self.observe_new = False
            
            if self.plan.startswith('[goexplore]'):
                action = self.goexplore()
            elif self.plan.startswith('[gocheck]'):
                action = self.gocheck()
            elif self.plan.startswith('[gograb]'):
                action = self.gograb()
            elif self.plan.startswith('[goput]'):
                print("goput动作")
                action = self.goput()
            elif self.plan.startswith('[send_message]'):
                action = self.plan[:]
                self.comm_chars += len(self.plan) - len('[send_message] ')
                self.comm_counts += 1
                self.plan = None
                
            else:
                raise ValueError(f"unavailable plan {self.plan}")
        print("step+1 返回action")
        self.steps += 1
        info.update({"plan": self.plan,
                     })
        if action == self.last_action and self.current_room['class_name'] == self.last_room:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_action = action
        self.last_location = self.location
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.plan_logger.info("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.plan = None
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walktowards] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0
        print(f"================={self.agent_names[self.agent_id]}{action}==============")
        
        return action, info


    def reset(self, obs, containers_name, goal_objects_name, rooms_name, goal, episode_logger, task_id,plan_logger):
        self.vision_pipeline = vision_pipeline.Vision_Pipeline(self.config, obs)
        self.steps = 0
        self.containers_name = containers_name
        self.goal_objects_name = goal_objects_name
        self.rooms_name = rooms_name
        self.roomname2id = {x['class_name']: x['id'] for x in obs['room_info']}
        self.id2node = {}
        self.stuck = 0
        self.last_room = None
        self.unsatisfied = {k: v[0] for k, v in goal.items()}
        self.satisfied = []
        self.goal_location = list(goal.keys())[0].split('_')[-1]
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.id_inside_room = {self.goal_location_id: self.rooms_name[:], self.opponent_agent_id: None}
        self.done_time = 0
        self.task_id = task_id
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.opponent_grabbed_objects = []

        self.room_explored = {
            "livingroom": False,
            "kitchen": False,
            "bedroom": False,
            "bathroom": False,
        }
        self.task_id = task_id

        self.current_room = self.vision_pipeline.object_info[obs['current_room']]
        self.plan = None
        self.action_history = [f"[goto] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []
        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)
        self.episode_logger = episode_logger

        self.location = obs['location']
        self.last_location = [0, 0, 0]
        self.last_action = None
        self.rotated = 0
        self.done_time = 0 

        self.comm_chars = 0
        self.comm_counts = 0
        self.plan_logger = plan_logger
        #Cobel:
        # = con_per_room
        #zero + first
        self.team_unchecked_con = {
            _:{
            "livingroom": [],
            "kitchen": [],
            "bedroom": [],
            "bathroom": [],
        } for _ in self.agent_names
        }
    
        # = obj per room
        # 为每个智能体创建一个字典列表来跟踪未抓取的物体
        #zero + first
        self.team_ungrasped_obj = {
            _:{
            "livingroom": [],
            "kitchen": [],
            "bedroom": [],
            "bathroom": [],
        } for _ in self.agent_names
        }

        # grasped objects {'id': None,'name':None}
        #team [[{hand},{hand2}]]

        self.team_explored_rooms = { #探索了就改成"all"
            _ : {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
           } for _ in self.agent_names
        }

        self.team_subplan = { _ : None for _ in self.agent_names}

        

        self.team_grasped_obj = {
            _ : {
                _ : [] for _ in self.agent_names
            } for _ in self.agent_names
        }


        self.team_current_room = {
            _ : {
                _ : None for _ in self.agent_names
            } for _ in self.agent_names
        }

        
        self.dialogue = {} #处理完就清空
        self.message_received = {}
        self.my_subplan = None
        self.message_time = 0
        

    def get_api_num(self):
        return self.LLM.api_num


    def get_comm_tokens(self):
        return self.LLM.comm_tokens
    
    def get_token_stats(self):
        return self.LLM.token_stats
    
    def parse_belief_line(self,belief_type,beliefs):
        print("======开始解析信念============")
        
        for agent_name, update_belief in beliefs.items(): #逐个智能体更新 一阶更新各自的 0阶全部更新自己的
            formatted_beliefs = []
            unchecked_container = {
                room:[] for room in self.rooms_name
            }
            belief_lines = beliefs[agent_name]
            if belief_lines is None:
                continue
            for line in belief_lines.splitlines():
                line = line.strip()
                #<Office> (1000) -> <Office>(1000)
                # line = re.sub(r'>([^<]*?)\(', r'>\(', line)
                line = line.replace('> (', '>(')
                if not line:
                    continue
                
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                
                tokens = [t.lower() for t in tokens]

                
                if belief_type == "first": 
                    if agent_name == self.agent_names[self.agent_id]: #自己的不更新
                        pass
                    if tokens.count('believe') < 2:
                        continue
                    first_believe_idx = tokens.index('believe')
                    second_believe_idx = tokens.index('believe', first_believe_idx + 1)

                    if first_believe_idx == 0:
                        continue  # 没有前一个 token


                    # if tokens[first_believe_idx - 1] != self.agent_names[self.agent_id].lower():
                    #     continue  # 不匹配 agent_name
                    
                    if second_believe_idx == 0:
                        continue  # 不可能，但安全检查

                    if tokens[second_believe_idx - 1] != self.agent_names[self.opponent_agent_id].lower():
                        continue  # 不匹配 oppo_name
                    
                    belief_tokens = tokens[second_believe_idx + 1:]
                    if len(belief_tokens) < 3:
                        continue
                    

                    subject = belief_tokens[0]
                    predicate = belief_tokens[1]
                    obj = belief_tokens[2]

                    if 'in' in predicate:
                        if self.parse_room(obj) is None:
                            continue
                        obj = self.parse_room(obj)
                        if obj in self.rooms_name:
                            if self.parse_obj(subject) == None:
                                continue
                            obj_str,obj_name,obj_id = self.parse_obj(subject)
                            if obj_name in self.goal_objects_name:
                                #检查是否有了
                                for obj_dict in self.team_ungrasped_obj[agent_name][obj]:
                                    if obj_dict['id'] == obj_id:
                                        continue #有了就跳出
                                    #room
                                self.team_ungrasped_obj[agent_name][obj].append({'id':obj_id,'class_name':obj_name})
                                formatted_beliefs.append(f"{obj_name} is in {obj}")
                            else:
                                #检测到的肯定都是未完成的
                                for obj_dict in self.team_unchecked_con[agent_name][obj]:
                                    if obj_dict['id'] == obj_id:
                                        continue #有了就跳出
                                unchecked_container[obj].append({'id':obj_id,'class_name':obj_name}) #需要清除检查过的的容器 这里是消息告诉我有这个容器，对但是我其实过去一下就更新了。
                                formatted_beliefs.append(f"{obj_str} is in {obj}")     
                                #containers_name 有一个全局的容器信息

                    if 'at' in predicate:
                        if self.parse_room(obj) is None:
                            continue
                        obj = self.parse_room(obj)
                        obj = obj
                        if subject.capitalize() in self.agent_names:
                            self.team_current_room[agent_name][subject.capitalize()] = {'class_name':obj}
                        formatted_beliefs.append(f"{subject.capitalize()} is at {obj}")    
                    if 'hold' in predicate:
                        if self.parse_obj(obj) == None:
                                continue
                        obj_str,obj_name,obj_id = self.parse_obj(obj)
                        if subject.capitalize() in self.agent_names:
                            for hold_obj in self.team_grasped_obj[agent_name][subject.capitalize()]:
                                if hold_obj['id'] == obj_id:
                                    continue
                            self.team_grasped_obj[agent_name][subject.capitalize()].append({'id':obj_id,'class_name':obj_name})
                        formatted_beliefs.append(f"{subject.capitalize()} is holding {obj_name}")
                        
                    # if 'explore' in predicate:
                    #     if self.parse_room(subject) is None:
                    #         continue
                    #     subject = self.parse_room(subject)
                    #     if subject in self.rooms_name:
                    #         if 'yes' in obj:
                    #             self.team_explored_rooms[agent_name][subject] = 'all'
                    #             formatted_beliefs.append(f"{subject} explored all")

                else:
                    try:
                        believe_idx = tokens.index('believe')  # 不区分大小写
                    except ValueError:
                        continue
                    belief_tokens = tokens[believe_idx + 1:]      # 用原始 tokens 提取内容

                    if len(belief_tokens) < 3:
                        continue
                    subject = belief_tokens[0]
                    predicate = belief_tokens[1]
                    obj = belief_tokens[2]
                    if 'in' in predicate:
                        if self.parse_room(obj) is None:
                                continue
                        obj = self.parse_room(obj)
                        if obj in self.rooms_name:
                            unchecked_container_room = []
                            if self.parse_obj(subject) == None:
                                continue
                            obj_str,obj_name,obj_id = self.parse_obj(subject)
                            if obj_name in self.goal_objects_name:
                                #检查是否有了
                                for obj_dict in self.team_ungrasped_obj[agent_name][obj]:
                                    if obj_dict['id'] == obj_id:
                                        continue #有了就跳出
                                    #room
                                self.team_ungrasped_obj[agent_name][obj].append({'id':obj_id,'class_name':obj_name})
                                formatted_beliefs.append(f"{obj_name} is in {obj}")
                            else:
                                #检测到的肯定都是未完成的容器
                                for obj_dict in self.team_unchecked_con[agent_name][obj]:
                                    if obj_dict['id'] == obj_id:
                                        continue #有了就跳出
                                unchecked_container[obj].append({'id':obj_id,'class_name':obj_name}) 
                                formatted_beliefs.append(f"{obj_str} is in {obj}")  
                                #需要清除检查过的的容器 这里是消息告诉我有这个容器，对但是我其实过去一下就更新了。
                                #containers_name 有一个全局的容器信息
                    if 'at' in predicate:
                        if self.parse_room(obj) is None:
                            continue
                        obj = self.parse_room(obj)
                        if subject.capitalize() in self.agent_names:
                            self.team_current_room[agent_name][subject.capitalize()] = {'class_name':obj}
                        formatted_beliefs.append(f"{subject.capitalize()} is at {obj}") 
                    if 'hold' in predicate:
                        if self.parse_obj(obj) == None:
                                continue
                        obj_str,obj_name,obj_id = self.parse_obj(obj)
                        if subject.capitalize() in self.agent_names:
                            if subject.capitalize() == agent_name:
                                continue
                            for hold_obj in self.team_grasped_obj[agent_name][subject.capitalize()]:
                                if hold_obj['id'] == obj_id:
                                    continue
                            self.team_grasped_obj[agent_name][subject.capitalize()].append({'id':obj_id,'class_name':obj_name})
                        formatted_beliefs.append(f"{subject.capitalize()} is holding {obj_name}")  
                        
                    # if 'explore' in predicate:
                    #     if self.parse_room(subject) is None:
                    #         continue
                    #     subject = self.parse_room(subject)
                    #     if subject in self.rooms_name:
                    #         if 'yes' in obj:
                    #             self.team_explored_rooms[agent_name][subject] = 'all'
                    #             formatted_beliefs.append(f"{subject} explored all")
            for room_name,room_con in self.team_unchecked_con[agent_name].items():
                if room_name not in unchecked_container.keys():
                    continue
                self.team_unchecked_con[agent_name][room_name] = unchecked_container[room_name]
                formatted_beliefs.append(f"{room_name} unchecked containers {unchecked_container[room_name]}")
            print(formatted_beliefs)
            self.episode_logger.info(f"{agent_name} beliefs: {formatted_beliefs}")

    def parse_belief_line_con(self,belief_type,beliefs):
        print("======开始解析信念============")
        
        for agent_name, update_belief in beliefs.items(): #逐个智能体更新 一阶更新各自的 0阶全部更新自己的
            formatted_beliefs = []
            unchecked_container = {
                room:[] for room in self.rooms_name
            }
            belief_lines = beliefs[agent_name]
            if belief_lines is None:
                continue
            for line in belief_lines.splitlines():
                line = line.strip()
                #<Office> (1000) -> <Office>(1000)
                # line = re.sub(r'>([^<]*?)\(', r'>\(', line)
                line = line.replace('> (', '>(')
                if not line:
                    continue
                
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                
                tokens = [t.lower() for t in tokens]

                
                if belief_type == "first": 
                    if agent_name == self.agent_names[self.agent_id]: #自己的不更新
                        pass
                    if tokens.count('believe') < 2:
                        continue
                    first_believe_idx = tokens.index('believe')
                    second_believe_idx = tokens.index('believe', first_believe_idx + 1)

                    if first_believe_idx == 0:
                        continue  # 没有前一个 token


                    # if tokens[first_believe_idx - 1] != self.agent_names[self.agent_id].lower():
                    #     continue  # 不匹配 agent_name
                    
                    if second_believe_idx == 0:
                        continue  # 不可能，但安全检查

                    if tokens[second_believe_idx - 1] != self.agent_names[self.opponent_agent_id].lower():
                        continue  # 不匹配 oppo_name
                    
                    belief_tokens = tokens[second_believe_idx + 1:]
                    if len(belief_tokens) < 3:
                        continue
                    

                    subject = belief_tokens[0]
                    predicate = belief_tokens[1]
                    obj = belief_tokens[2]

                    if 'in' in predicate:
                        if self.parse_room(obj) is None:
                            continue
                        obj = self.parse_room(obj)
                        if obj in self.rooms_name:
                            if self.parse_obj(subject) == None:
                                continue
                            obj_str,obj_name,obj_id = self.parse_obj(subject)
                            if obj_name in self.goal_objects_name:
                                #检查是否有了
                                for obj_dict in self.team_ungrasped_obj[agent_name][obj]:
                                    if obj_dict['id'] == obj_id:
                                        continue #有了就跳出
                                    #room
                                self.team_ungrasped_obj[agent_name][obj].append({'id':obj_id,'class_name':obj_name})
                                formatted_beliefs.append(f"{obj_name} is in {obj}")
                            # else:
                            #     #检测到的肯定都是未完成的
                            #     for obj_dict in self.team_unchecked_con[agent_name][obj]:
                            #         if obj_dict['id'] == obj_id:
                            #             continue #有了就跳出
                            #     unchecked_container[obj].append({'id':obj_id,'class_name':obj_name}) #需要清除检查过的的容器 这里是消息告诉我有这个容器，对但是我其实过去一下就更新了。
                            #     formatted_beliefs.append(f"{obj_str} is in {obj}")     
                            #     #containers_name 有一个全局的容器信息

                    if 'at' in predicate:
                        if self.parse_room(obj) is None:
                            continue
                        obj = self.parse_room(obj)
                        obj = obj
                        if subject.capitalize() in self.agent_names:
                            self.team_current_room[agent_name][subject.capitalize()] = {'class_name':obj}
                        formatted_beliefs.append(f"{subject.capitalize()} is at {obj}")    
                    if 'hold' in predicate:
                        if self.parse_obj(obj) == None:
                                continue
                        obj_str,obj_name,obj_id = self.parse_obj(obj)
                        if subject.capitalize() in self.agent_names:
                            for hold_obj in self.team_grasped_obj[agent_name][subject.capitalize()]:
                                if hold_obj['id'] == obj_id:
                                    continue
                            self.team_grasped_obj[agent_name][subject.capitalize()].append({'id':obj_id,'class_name':obj_name})
                        formatted_beliefs.append(f"{subject.capitalize()} is holding {obj_name}")
                        
                    # if 'explore' in predicate:
                    #     if self.parse_room(subject) is None:
                    #         continue
                    #     subject = self.parse_room(subject)
                    #     if subject in self.rooms_name:
                    #         if 'yes' in obj:
                    #             self.team_explored_rooms[agent_name][subject] = 'all'
                    #             formatted_beliefs.append(f"{subject} explored all")



                            # else:
                            #     #检测到的肯定都是未完成的
                            #     for obj_dict in self.team_unchecked_con[agent_name][obj]:
                            #         if obj_dict['id'] == obj_id:
                            #             continue #有了就跳出
                            #     unchecked_container[obj].append({'id':obj_id,'class_name':obj_name}) 
                            #     formatted_beliefs.append(f"{obj_str} is in {obj}")  
                            #     #需要清除检查过的的容器 这里是消息告诉我有这个容器，对但是我其实过去一下就更新了。
                            #     #containers_name 有一个全局的容器信息
                    if 'at' in predicate:
                        if self.parse_room(obj) is None:
                            continue
                        obj = self.parse_room(obj)
                        if subject.capitalize() in self.agent_names:
                            self.team_current_room[agent_name][subject.capitalize()] = {'class_name':obj}
                        formatted_beliefs.append(f"{subject.capitalize()} is at {obj}") 
                    if 'hold' in predicate:
                        if self.parse_obj(obj) == None:
                                continue
                        obj_str,obj_name,obj_id = self.parse_obj(obj)
                        if subject.capitalize() in self.agent_names:
                            if subject.capitalize() == agent_name:
                                continue
                            for hold_obj in self.team_grasped_obj[agent_name][subject.capitalize()]:
                                if hold_obj['id'] == obj_id:
                                    continue
                            self.team_grasped_obj[agent_name][subject.capitalize()].append({'id':obj_id,'class_name':obj_name})
                        formatted_beliefs.append(f"{subject.capitalize()} is holding {obj_name}")  
                        
                    # if 'explore' in predicate:
                    #     if self.parse_room(subject) is None:
                    #         continue
                    #     subject = self.parse_room(subject)
                    #     if subject in self.rooms_name:
                    #         if 'yes' in obj:
                    #             self.team_explored_rooms[agent_name][subject] = 'all'
                    #             formatted_beliefs.append(f"{subject} explored all")
            # for room_name,room_con in self.team_unchecked_con[agent_name].items():
            #     if room_name not in unchecked_container.keys():
            #         continue
            #     self.team_unchecked_con[agent_name][room_name] = unchecked_container[room_name]
            #     formatted_beliefs.append(f"{room_name} unchecked containers {unchecked_container[room_name]}")
            print(formatted_beliefs)
            self.episode_logger.info(f"{agent_name} beliefs: {formatted_beliefs}")
    def parse_obj(self, text):
        """
        将类似 <livingroom>(1000) 的字符串：
        1. 格式化为 <Livingroom> (1000)
        2. 提取出 name 和 id

        :param text: str, 如 "<livingroom>(1000)"
        :return: tuple (formatted_str, name, id_int)
                如 ('<Livingroom> (1000)', 'Livingroom', 1000)
        """
        # 使用正则匹配 <...>(数字)
        match = re.match(r'<([^>]+)>\s*\((\d+)\)', text.strip())
        if not match:
            # raise ValueError(f"无法解析格式: {text}")
            return None

        name_raw = match.group(1)   # 'livingroom'
        id_str = match.group(2)     # '1000'

        # 名字首字母大写（其他字母保持原样，如 livingRoom → LivingRoom）
        # 如果希望每个单词首字母大写，用 .title()；如果只第一个字母，用 .capitalize()
        name_raw = name_raw.lower() # "livingroom" → "Livingroom"
        id = int(id_str)
        # 格式化输出字符串
        formatted = f"<{name_raw}> ({id_str})"

        return formatted, name_raw, id
    

    def parse_room(self, text):
        """
        将类似 <livingroom>(1000) 的字符串：
        1. 格式化为 <Livingroom> (1000)
        2. 提取出 name 和 id

        :param text: str, 如 "<livingroom>(1000)"
        :return: tuple (formatted_str, name, id_int)
                如 ('<Livingroom> (1000)', 'Livingroom', 1000)
        """
        for room in self.rooms_name:
            if room in text:
                return room

        return None
##
import json
import os
import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
import copy
from PIL import Image
from agent_memory import AgentMemory
import re
import ast
from concurrent.futures import ThreadPoolExecutor

from LLM.LLM_cobel import LLM_cobel

CELL_SIZE = 0.125
ANGLE = 15
import logging
# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns

class lm_agent_cobel:
    """
    大模型驱动的智能体类
    主要功能：
    1. 使用大模型进行决策规划
    2. 处理环境观察和状态
    3. 执行导航和物体操作
    4. 管理智能体记忆和状态
    """

    def __init__(self, agent_id, logger, max_frames, args, output_dir="results"):
        """
        初始化大模型智能体

        参数:
            agent_id: 智能体ID
            logger: 日志记录器
            max_frames: 最大帧数
            args: 配置参数
            output_dir: 输出目录
        """
                #counting
        self.characters = 0 # model-generated-characters
        self.comm_num = 0 # agent-communication-times
                # 环境状态相关变量
        self.with_oppo = None  # 对手持有的物体
        self.oppo_pos = None  # 对手位置
        self.with_character = None  # 角色持有的物体
        self.color2id = None  # 颜色到ID的映射
        self.satisfied = None  # 已完成的物体
        self.object_list = None  # 物体列表
        self.container_held = None  # 持有的容器
        self.gt_mask = None  # 是否使用真实掩码
        self.episode = None

        self.one_update = False

        self.single = False
        

        # 物体信息存储
        self.object_info = (
            {}
        )  # 物体详细信息 {id: {id: xx, type: 0/1/2, name: sss, position: x,y,z}}
        self.object_per_room = (
            {}
        )
        self.my_object_per_room = None
          # 每个房间的物体 {room_name: {0/1/2: [{id: xx, type: 0/1/2, name: sss, position: x,y,z}]}}
        self.oppo_object_per_room = None
        # 地图相关
        self.id_map = None  # ID地图
        self.object_map = None  # 物体地图

        # 智能体基本信息
        self.agent_id = agent_id
        self.agent_type = "lm_agent_cobel"
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id

        # 环境配置
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 130) #COBEL size该过了
        self.save_img = True

        # 场景边界
        self._scene_bounds = {"x_min": -15, "x_max": 15, "z_min": -7.5, "z_max": 7.5}

        # 导航参数
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.logger = logger
        random.seed(1024)
        self.debug = True

        # 观察和状态相关
        self.new_object_list = None  # 新发现的物体
        self.visible_objects = None  # 可见物体
        self.num_frames = None  # 当前帧数
        self.steps = None  # 步数
        self.obs = None  # 当前观察
        self.local_step = 0  # 局部步数

        # 动作历史
        self.last_action = None  # 上一个动作
        self.pre_action = None  # 前一个动作

        # 目标相关
        self.goal_objects = None  # 目标物体
        self.dropping_object = None  # 正在放置的物体

        # 大模型配置
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM_cobel(
            self.source,
            self.lm_id,
            self.prompt_template_path,
            self.communication,
            self.cot,
            self.args,
            self.agent_id,
        )
        self.action_history = []  # 动作历史
        self.dialogue_history = []  # 对话历史
        self.dialogue = []
        self.plan = None  # 当前计划
        self.visible_obj = {}
        self.last_hold = None

        # 房间和位置相关
        self.rooms_name = None  # 房间名称
        self.rooms_explored = {}  # 已探索的房间
        self.my_rooms_explored = {}
        self.oppo_rooms_explored = {}
        self.new_room_explored = {}
        self.position = None  # 当前位置
        self.forward = None  # 朝向
        self.current_room = None  # 当前房间
        self._objects_id = None  # 持有的物体ID TODO 打印看一下结构
        self.oppo_holding_objects_id = None  # 对手持有的物体ID
        self.oppo_last_room = None  # 对手最后所在的房间
        self.rotated = None  # 旋转状态
        self.navigation_threshold = 5  # 导航阈值
        self.detection_threshold = 5  # 检测阈值

        # 通信相关配置
        self.communication = args.communication  # 是否启用通信功能
        # print(f"是否启用通信：{self.communication}")
        self.episode_logger = None  # 记录当前episode的日志
        self.plan_logger = None #记录subplan plan等等
        # COBEL-zhimin
        self.belief_rules = None
        self.zero_order_beliefs = "None"
        self.first_order_beliefs = "None"
        self.subplan_done = True  # 是否完成子目标
        self.obs_not_updated = True  # 是否更新了观测
        self.max_message_time = 2
        self.message_time = 0
        self.action_history_max_length = 3
        self.action_history_w_mes = []
        self.my_subplan = None
        self.comm_counts = 0
        self.comm_chars = 0
        self.message_received = []
        self.oppo_holding_objects_first = [
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]},
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]}
        ]
        self.oppo_holding_objects_zero = [
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]},
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]}
        ]
        self.oppo_current_room_first = None
        self.oppo_current_room_zero = None
        self.my_current_room_first = None
        self.my_holding_first = [
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]},
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]}
        ]
        self.oppo_last_room_mes = None
        self.init_challenge_descs = None

    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        """
        获取指定颜色的点云数据

        参数:
            color: 目标颜色

        返回:
            点云数据
        """
        depth = self.obs["depth"].copy()
        for i in range(len(self.obs["seg_mask"])):
            for j in range(len(self.obs["seg_mask"][0])):
                if (self.obs["seg_mask"][i][j] != color).any():
                    depth[i][j] = 1e9
        # camera info
        FOV = self.obs["FOV"]
        W, H = depth.shape
        cx = W / 2.0
        cy = H / 2.0
        fx = cx / np.tan(math.radians(FOV / 2.0))
        fy = cy / np.tan(math.radians(FOV / 2.0))

        # Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)

        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].copy().reshape(-1)
        yy = yy[index].copy().reshape(-1)
        depth = depth[index].copy().reshape(-1)

        pc = np.stack((xx, yy, depth, np.ones_like(xx)))

        pc = pc.reshape(4, -1)

        E = self.obs["camera_matrix"]
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)
        return rpc[:3]

    def cal_object_position(self, o_dict):
        """
        计算物体位置

        参数:
            o_dict: 物体信息字典

        返回:
            物体位置坐标
        """
        pc = self.get_pc(o_dict["seg_color"])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]

    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o["type"] is not None and o["type"] < 4:
                visible_obj.append(o)
        return visible_obj

    def get_object_list(self):
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        
        for object_type in [0, 1, 2]:
            obj_map_indices = np.where(self.object_map == object_type + 1)##object_map update from getting new object
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.id_map[i, j]#check4 :id_map where it comes from? A:from updating new objects
                if (
                    id in self.satisfied
                    or id in self.holding_objects_id
                    or id in self.oppo_holding_objects_id
                    or self.object_info[id] in object_list[object_type]
                ):
                    continue
                object_list[object_type].append(self.object_info[id])
                
                #id:{'id': int,'type': int, 'name':'str', 'position'"array"}
                room = self.env_api["belongs_to_which_room"](
                    self.object_info[id]["position"]#check5 object_info where the position comes from?   A:from getting new object
                )
                if room is None:
                    self.logger.warning(f"obj {self.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                self.object_per_room[room][object_type].append(self.object_info[id])
                if f"<{self.object_info[id]['name']}>({self.object_info[id]['id']})" not in self.my_object_per_room[room][object_type]:
                    self.my_object_per_room[room][object_type].append(f"<{self.object_info[id]['name']}>({self.object_info[id]['id']})")
        self.object_list = object_list #TODO
    #act刚开始更新
    def get_new_object_list(self):## key function
        self.visible_objects = self.obs["visible_objects"] #self.visible_objects是未经过筛选的
        self.new_object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict["id"] is None:
                continue
            self.color2id[o_dict["seg_color"]] = o_dict["id"]
            if (
                o_dict["id"] is None
                or o_dict["id"] in self.satisfied
                or o_dict["id"] in self.with_character
                or o_dict["type"] == 4
            ):
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            object_id = o_dict["id"]
            new_obj = False

            if object_id not in self.object_info:#can know that the object_info is personlize
                self.object_info[object_id] = {}
                new_obj = True

            if object_id not in self.visible_obj.keys(): #COBEL
                if o_dict["type"] < 4:
                    self.visible_obj[object_id] = {}

            self.object_info[object_id]["id"] = object_id #=o_dict["id"]
            self.object_info[object_id]["type"] = o_dict["type"]
            self.object_info[object_id]["name"] = o_dict["name"]
            if o_dict["type"] == 3:  # the agent'information updating
                if o_dict["id"] == self.opponent_agent_id:
                    self.last_hold = self.obs['oppo_held_objects']
                    position = self.cal_object_position(o_dict)
                    self.oppo_pos = position#update the partner's position
                    if position is not None:
                        oppo_last_room = self.env_api["belongs_to_which_room"](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                            self.oppo_current_room_zero = oppo_last_room #Cobel
                            self.oppo_current_room_first = oppo_last_room
                            self.visible_obj[object_id]["id"] = object_id
                            self.visible_obj[object_id]["type"] = o_dict["type"]
                            self.visible_obj[object_id]["name"] = o_dict["name"]
                            self.visible_obj[object_id]["position"] = str(self.num_frames) + " at " + oppo_last_room #COBEL 这里应该是
                continue
            if object_id in self.satisfied or object_id in self.with_character:
                continue
            self.object_info[object_id]["position"] = position
            if o_dict["type"] < 3:
                self.visible_obj[object_id]["id"] = object_id
                self.visible_obj[object_id]["type"] = o_dict["type"]
                self.visible_obj[object_id]["name"] = o_dict["name"]
                self.visible_obj[object_id]["position"] = self.env_api["belongs_to_which_room"](position)
            if o_dict["type"] == 0:
                x, y, z = self.object_info[object_id]["position"]

                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[0].append(object_id)

            elif o_dict["type"] == 1:
                x, y, z = self.object_info[object_id]["position"]
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[1].append(object_id)
            elif o_dict["type"] == 2:
                x, y, z = self.object_info[object_id]["position"]
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[2].append(object_id)

    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any():
                return -100  # wall
            else:
                return self.agent_id  # agent
        else:
            return self.color2id[color]

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold=1.0):# check if agent reach the pos
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        if self.plan.startswith("transport"):
            if self.env_api["belongs_to_which_room"](
                np.array([x, 0, z])
            ) != self.env_api["belongs_to_which_room"](np.array([gx, 0, gz])):
                return False
        return d < threshold




        # 配置日志
    
    # def setup_logger(self,name, log_file, level=logging.INFO):
    #     """设置日志记录器"""
    #     formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    #     handler = logging.FileHandler(log_file)
    #     handler.setFormatter(formatter)

    #     logger = logging.getLogger(name)
    #     logger.setLevel(level)
    #     logger.addHandler(handler)

    #     return logger


    # # 创建日志目录
    # if not os.path.exists("logs"):
    #     os.makedirs("logs")

    # # 创建llm日志记录器
    # log_filename = f"logs/coela_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # llm_logger = setup_logger("llm_logger", log_filename)

    def reset(
        self,
        obs,
        goal_objects=None,
        output_dir=None,
        env_api=None,
        rooms_name=None,
        agent_color=[-1, -1, -1],
        agent_id=0,
        gt_mask=True,
        save_img=True,
        episode=None,
        episode_logger=None,
        plan_logger=None
    ):
        self.force_ignore = []
        self.characters = 0 
        self.comm_num = 0 
        self.agent_memory = AgentMemory(
            agent_id=self.agent_id,
            agent_color=agent_color,
            output_dir=output_dir,
            gt_mask=self.gt_mask,
            gt_behavior=True,
            env_api=env_api,
            constraint_type=None,
            map_size=self.map_size,
            scene_bounds=self._scene_bounds,
        )
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self.oppo_pos = None
        goal_count = sum([v for k, v in goal_objects.items()])
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}#personlized attribution
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0
        # print(self.obs.keys())
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api["belongs_to_which_room"](self.position)
        self.rotated = None
        self.rooms_explored = {}
        self.my_rooms_explored = {}
        self.oppo_rooms_explored = {}
        self.last_hold = None
        self.message_time = 0
        # COBEL detect new exploration extend 
        self.new_room_explored = {} 


        self.oppo_object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        self.my_object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        for name in self.rooms_name:
            self.new_room_explored.update(
                {
                    name:'None'
                }
            )



        # self.zero_order_beliefs, self.first_order_beliefs = self.LLM.init_beliefs(self.rooms_name,self.goal_objects)


        self.my_subplan = None
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.dialogue = []
        self.gt_mask = gt_mask
        if self.gt_mask == True:##check6 what is gt_mask?
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection


            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
        self.navigation_threshold = 5

        self.episode_logger = episode_logger

        self.plan_logger = plan_logger
        # print(self.rooms_name)
        #COBEL - zhimin begin 修改了reset的返回 改为了初始化的信念模版
        self.my_subplan = None
        self.action_history_w_mes = []
        self.init_challenge_descs = None
        self.oppo_holding_objects_first = [
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]},
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]}
        ]
        self.oppo_holding_objects_zero = [
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]},
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]}
        ]
        self.oppo_current_room_first = None
        self.oppo_current_room_zero = None
        self.my_current_room_first = None
        self.my_holding_first = [
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]},
            {'type': None, 'id': None, 'name': None, 'contained':[None,None,None],'contained_name':[None,None,None]}
        ]
        self.oppo_last_room_mes = None
        #COBEL - zhimin end 每个episode初始化一次
        self.LLM.reset(self.rooms_name, self.goal_objects)
        self.save_img = save_img
        self.episode = episode
        self.message_received = []

        self.visible_obj = {}

    def move(self, target_pos):
        self.local_step += 1
        action, path_len = self.agent_memory.move_to_pos(target_pos)#check7 what is the action here? A:like the action down here
        return action

    def gotoroom(self):
        target_room = " ".join(self.plan.split(" ")[2:4])
        if target_room[-1] == ",":
            target_room = target_room[:-1]
        if self.debug:
            print(target_room)
        target_pos = self.env_api["center_of_room"](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # # add an interruption if anything new happens
        # if (
        #     len(self.new_object_list[0])
        #     + len(self.new_object_list[1])
        #     + len(self.new_object_list[2])
        #     > 0
        # ):
        #     self.action_history[-1] = self.action_history[-1].replace(
        #         self.plan, f"go to {self.current_room}"
        #     )
        #     self.new_object_list = {0: [], 1: [], 2: []}
        #     self.plan = None
        #     return None
        return self.move(target_pos)

    def goexplore(self):
        target_room = " ".join(self.plan.split(" ")[-2:])
        # assert target_room == self.current_room, f"{target_room} != {self.current_room}"
        target_pos = self.env_api["center_of_room"](target_room)
        self.explore_count += 1
        dis_threshold = 1 + self.explore_count / 50
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)
        if self.rotated is None:
            self.rotated = 0
        if self.rotated == 16:
            self.roatated = 0
            self.rooms_explored[target_room] = "all"#every direction going through
            self.my_rooms_explored[target_room] = "all"
            self.plan = None
            return None
        self.rotated += 1
        action = {"type": 1}
        return action

    def gograsp(self):
        target_object_id = int(self.plan.split(" ")[-1][1:-1])
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.object_map[np.where(self.id_map == target_object_id)] = 0
            self.id_map[np.where(self.id_map == target_object_id)] = 0
            self.plan = None
            return None

        if self.target_pos is None:
            self.target_pos = copy.deepcopy(
                self.object_info[target_object_id]["position"]
            )
        target_object_pos = self.target_pos

        if (
            target_object_id not in self.object_info
            or target_object_id in self.with_oppo
        ):
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)
        action = {
            "type": 3,
            "object": target_object_id,
            "arm": "left" if self.obs["held_objects"][0]["id"] is None else "right",
        }
        return action

    def goput(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            self.with_character = [self.agent_id]
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_list[2][0]["position"])
        target_pos = self.target_pos

        if not self.reach_target_pos(target_pos, 1.5):
            return self.move(target_pos)
        if self.obs["held_objects"][0]["type"] is not None:
            self.dropping_object += [self.obs["held_objects"][0]["id"]]
            if self.obs["held_objects"][0]["type"] == 1:
                self.dropping_object += [
                    x for x in self.obs["held_objects"][0]["contained"] if x is not None
                ]
            return {"type": 5, "arm": "left"}
        else:
            self.dropping_object += [self.obs["held_objects"][1]["id"]]
            if self.obs["held_objects"][1]["type"] == 1:
                self.dropping_object += [
                    x for x in self.obs["held_objects"][1]["contained"] if x is not None
                ]
            return {"type": 5, "arm": "right"}

    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action

    def detect(self):
        """
        执行目标检测

        返回:
            obj_infos: 检测到的物体信息
            curr_seg_mask: 分割掩码
        """
        detect_result = self.detection_model(self.obs["rgb"][..., [2, 1, 0]])[
            "predictions"
        ][0]
        obj_infos = []
        curr_seg_mask = np.zeros(
            (self.obs["rgb"].shape[0], self.obs["rgb"].shape[1], 3)
        ).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result["labels"])):
            if detect_result["scores"][i] < 0.3:
                continue
            mask = detect_result["masks"][:, :, i]
            label = detect_result["labels"][i]
            curr_info = self.env_api["get_id_from_mask"](
                mask=mask, name=self.detection_model.cls_to_name_map(label)
            ).copy()
            if curr_info["id"] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info["seg_color"]
        curr_with_seg, curr_seg_flag = self.env_api["get_with_character_mask"](
            character_object_ids=self.with_character
        )
        curr_seg_mask = curr_seg_mask * (
            ~np.expand_dims(curr_seg_flag, axis=-1)
        ) + curr_with_seg * np.expand_dims(curr_seg_flag, axis=-1)
        return obj_infos, curr_seg_mask

    def get_progress_description(self):
        

        return self.LLM.get_progress_description(
            self.num_frames,
            self.current_room, #current room
            self.rooms_explored, #room CHECK
            self.obs["held_objects"],
            [self.object_info[x] for x in self.satisfied if x in self.object_info],
            self.object_list,
            self.object_per_room, #check
            self.action_history,
            self.dialogue_history,
            self.obs["oppo_held_objects"],
            self.oppo_last_room,
            self.logger,
            self.oppo_holding_objects_zero,
            self.oppo_current_room_zero,
            self.my_rooms_explored,
            self.my_object_per_room
        )



    #COBEL - zhimin
    def init_beliefs(self):
        self.zero_order_beliefs, self.first_order_beliefs = self.LLM.init_beliefs(self.init_challenge_descs,self.goal_objects)


            
    def comm(self, difference, my_subplan):
        message = self.LLM.comm(difference, my_subplan)

        return message
    #COBEL-zhimin
    def intuitive_planning(self):

        progress_desc = self.get_progress_description()

        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} action_with_message_history:{self.action_history_w_mes}")
        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} action_history:{self.action_history}\nmy_subplan:{self.my_subplan}")
        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} action_with_message_history:{self.action_history_w_mes}")
        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} action_history:{self.action_history}")
        action_history_desc = ", ".join(self.action_history)
        plan = self.LLM.intuitive_planning(self.my_subplan,
                                           action_history_desc,
                                           progress_desc,
                                          self.episode_logger)
        
        return plan


    #COBEL -shaokang
    def observation2text(self,info):
        measurement_observation = {}
        current_frames = info['obs']['current_frames']
        current_room = info["current_room"]
        holding = ['','']
        container = ['',''] #应该是三个
        oppo_holding = ['','']
        oppo_container = ['','']
        visible_ids = []
        satisfied = ""
        satisfied_list = [self.object_info[x] for x in self.satisfied if x in self.object_info]
        if len(satisfied_list) == 0:
            if len(self.object_list[2]) == 0:
                satisfied += "I haven't found the goal position bed. "
            else:
                satisfied += ""
        else:
            satisfied += f"{'I' if self.single else 'We'}'ve already transported "
            unique_satisfied = []
            for x in satisfied_list:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x["type"] == 0]) == 0:
                satisfied += "nothing"
            satisfied += ", ".join(
                [
                    f"<{x['name']}> ({x['id']})"
                    for x in unique_satisfied
                    if x["type"] == 0
                ]
            )
            satisfied += " to the bed. "
        #自己拿的
        for id ,item in enumerate(info['obs']['held_objects']):
            if item['id'] is not None:
                holding[id] = '<' + item["name"] + "> " + '(' + str(item["id"]) + ")"
                if item['contained'] != [None, None, None]:
                    container[id] = 'with'
                    for index,obj in enumerate(item["contained"]):
                        if obj != None:
                            container[id] += '<' + item['contained_name'][index] + '> ' + '(' + str(obj) + '),  '
                    container[id] += "in it. "
        if holding[0] == '' and holding[1] == '':
            pro_holding = "holding nothing."
        else:
            pro_holding = " holding" + holding[0] + container[0] 
            pro_holding += ' and ' if holding[0] != None and holding[1] != None else ''
            pro_holding += holding[1] + container[1]
        seeing = 'I see '
        last_agent_position = None
        last_see_frame = None
        
        see_oppo = False
        for item in info["visible_objects"].values():
            if item.get("type") == 3:
                see_oppo = True
                # 找到type为3的字典，可以在这里处理

        #看到合作者了 再解析
        if self.last_hold != None and see_oppo:
            for id ,item in enumerate(self.last_hold):
                if item['id'] is not None: #协作者拿的东西
                    visible_ids.append(item['id'])
                    oppo_holding[id] = '<' + item["name"] + "> " + '(' + str(item["id"]) + ")"
                    if item['contained'] != [None, None, None]:  #拿的东西是否含东西
                        oppo_container[id] = 'with'
                        
                        for index,obj in enumerate(item["contained"]):
                            visible_ids.append(obj)
                            if obj != None:
                                oppo_container[id] += '<' + item['contained_name'][index] + '> ' + '(' + str(obj) + '), '
                        oppo_container[id] += "in it. "
        if oppo_holding[0] == '' and oppo_holding[1] == '':
            oppo_pro_holding = " holding nothing."
        else:
            oppo_pro_holding = " holding" + oppo_holding[0] + oppo_container[0]
            oppo_pro_holding += " and " if oppo_holding[0] != None and oppo_holding[1] != None else ''
            oppo_pro_holding += oppo_holding[1] + oppo_container[1]

         





        # 看到的物体
        for item in info["visible_objects"].values():
            if item["type"] != 3 and item["id"] not in visible_ids: #不是合作者且没被记录过的物体
                visible_ids.append(item['id'])
                seeing += '<' + item["name"] + '> ' + '(' + str(item["id"]) + ')' + " in " + item["position"] + '. '
                # seeing += '<' + item["name"] + '> ' + '(' + str(item["id"]) + ')' + " in " + info['current_room'] + '. '
            elif item["type"] == 3:
                # last_agent_position = item['position'][5:]
                # last_see_frame = item['position'][:2]
                # 提取第一个数字（帧数）
                frame_match = re.match(r"(\d+)", item['position'])
                last_see_frame = frame_match.group(1) if frame_match else None

                # 提取 at 后面的所有内容
                at_match = re.search(r"at\s+(.+)", item['position'])
                last_agent_position = at_match.group(1) if at_match else None
        seeing = seeing if seeing != 'I see ' else ''
        
        observation = "At " + str(current_frames) + " frames, I'm in " + current_room + ', ' + pro_holding + seeing 
        oppo_obs = None
        if last_see_frame is not None:
            observation += ("I saw "+ self.agent_names[self.opponent_agent_id] + " at " + last_see_frame + ' frames at ' + last_agent_position + oppo_pro_holding)
            oppo_obs = ("I saw "+ self.agent_names[self.opponent_agent_id] + " at " + last_see_frame + ' frames at ' + last_agent_position + oppo_pro_holding)
        else:
            oppo_obs = None
        explored_extend = 'I have '
        for item in self.new_room_explored.keys():
            if item in info['room_explored'].keys():
                if self.new_room_explored[item] != info['room_explored'][item]:
                    explored_extend += 'explored ' + info['room_explored'][item] + ' of the ' + item + ', '
                    self.new_room_explored[item] = info['room_explored'][item]
        
                
        explored_extend = explored_extend if explored_extend != 'I have ' else ""
        observation += explored_extend
        observation += satisfied
        measurement_observation['observation'] = observation
        # measurement_observation['messages'] = self.obs['messages']
        # 先把上次自己发的消息加上
        measurement_observation['messages'] = {}
        for receiver_name in self.agent_names:
            if self.obs['messages'][self.agent_id] is not None:
                measurement_observation['messages'][receiver_name] = f"{receiver_name}:" #str
                measurement_observation['messages'][receiver_name] += self.obs['messages'][self.agent_id]
                measurement_observation['messages'][receiver_name] += '\n'
            else:
                measurement_observation['messages'][receiver_name] = ""
        # 再把收到的消息加上
        # for idx, sender_name in enumerate(self.agent_names):
        #     if self.obs['messages'][idx] is not None:
        #         measurement_observation["messages"] += self.obs['messages'][idx]
                # mes_dic = ast.literal_eval(self.obs['messages'][idx])
                # for receiver_name in self.agent_names:
                #     for key,value in mes_dic.items():
                #         if receiver_name in key:
                #             measurement_observation['messages'][receiver_name] += f"{sender_name}:" #str
                #             measurement_observation['messages'][receiver_name] += value #str
                #             measurement_observation['messages'][receiver_name] += '\n'

        # for idx, receiver_name in enumerate(self.agent_names):
        #     if measurement_observation['messages'][receiver_name] == "":
        #         measurement_observation['messages'][receiver_name] = "None"


        #Done send to specific agent
        # self.plan_logger.info(
        #                 f"\nAt {self.steps} steps, {self.agent_names[self.agent_id]}:\nvisual_obs:\n{measurement_observation['observation']}\nMessages:\n{self.dialogue}"
        #             )

        return measurement_observation,oppo_obs


    #COBEL - zhimin 这里用来以后当作自己的act 我会先copy一个 coela的 act用                 
    def act_cobel(self, obs):
        """
        执行动作

        参数:
            obs: 环境观察

        返回:
            action: 要执行的动作 / 发送消息
        """
        # if self.zero_order_beliefs == 'None' and self.first_order_beliefs == 'None':
        #     self.init_beliefs()
        # if "EXPLORED" not in self.zero_order_beliefs or "EXPLORED" not in self.first_order_beliefs:
        #     raise ValueError("Failed to init beliefs.")
        
        # self.logger.info(f"zero:\n{self.zero_order_beliefs}\nfirst:\n{self.first_order_beliefs}")
        self.obs = obs.copy()
        self.obs["rgb"] = self.obs["rgb"].transpose(1, 2, 0)
        self.num_frames = obs["current_frames"]
        self.steps += 1

        if not self.gt_mask:
            self.obs["visible_objects"], self.obs["seg_mask"] = self.detect()

        if obs["valid"] == False:#how to be invalid?
            if self.last_action is not None and "object" in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action["object"])] = 0
                self.id_map[np.where(self.id_map == self.last_action["object"])] = 0
                self.satisfied.append(self.last_action["object"])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"

        if self.communication:

            # 遍历所有接收到的消息
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    # 将消息添加到对话历史中，格式为"智能体名称: 消息内容"
                    # 使用copy.deepcopy确保消息内容不会被意外修改
                    self.dialogue_history.append(
                        f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}"
                    )
                    self.dialogue.append(
                        f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}"
                    )
                    if i != self.agent_id:
                        self.message_received.append(
                            f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}"
                        )

        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api["belongs_to_which_room"](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api["get_room_distance"](self.position)
        if (
            self.current_room not in self.rooms_explored
            or self.rooms_explored[self.current_room] != "all"
        ):
            self.rooms_explored[self.current_room] = "part"

        if (
            self.current_room not in self.my_rooms_explored
            or self.my_rooms_explored[self.current_room] != "all"
        ):
            self.my_rooms_explored[self.current_room] = "part"
        
        if self.agent_id not in self.with_character:
            self.with_character.append(
                self.agent_id
            )  # DWH: buggy env, need to solve later.
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        hand = 0
        for x in self.obs["held_objects"]:
            if x['type'] != None:
                self.my_holding_first[hand] = x
                hand += 1
            if x["type"] == 0:
                self.holding_objects_id.append(x["id"])
                if x["id"] not in self.with_character:
                    self.with_character.append(
                        x["id"]
                    )  # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
            elif x["type"] == 1:
                self.holding_objects_id.append(x["id"])
                if x["id"] not in self.with_character:
                    self.with_character.append(
                        x["id"]
                    )  # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
                for y in x["contained"]:
                    if y is None:
                        break
                    if y not in self.with_character:
                        self.with_character.append(y)
                    # self.with_character.append(y)
        oppo_name = {}
        oppo_type = {}
        oppo_hand = 0
        for x in self.obs["oppo_held_objects"]:
            if x['type'] != None:
                self.oppo_holding_objects_first[oppo_hand] = x
                self.oppo_holding_objects_zero[oppo_hand] = x
                oppo_hand += 1
            if x["type"] == 0:
                self.oppo_holding_objects_id.append(x["id"])
                self.oppo_holding_objects_first
                self.oppo_holding_objects_zero
                self.with_oppo.append(x["id"])
                oppo_name[x["id"]] = x["name"]
                oppo_type[x["id"]] = x["type"]
            elif x["type"] == 1:
                self.oppo_holding_objects_id.append(x["id"])
                self.with_oppo.append(x["id"])
                oppo_name[x["id"]] = x["name"]
                oppo_type[x["id"]] = x["type"]
                for i, y in enumerate(x["contained"]):
                    if y is None:
                        break
                    self.with_oppo.append(y)
                    oppo_name[y] = x["contained_name"][i]
                    oppo_type[y] = 0
        for obj in self.with_oppo:
            if obj not in self.satisfied:
                self.satisfied.append(obj)
                self.object_info[obj] = {
                    "name": oppo_name[obj],
                    "id": obj,
                    "type": oppo_type[obj],
                }
                self.object_map[np.where(self.id_map == obj)] = 0
                self.id_map[np.where(self.id_map == obj)] = 0
        if not self.obs["valid"]:  # invalid, the object is not there
            if self.last_action is not None and "object" in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action["object"])] = 0
                self.id_map[np.where(self.id_map == self.last_action["object"])] = 0
        if len(self.dropping_object) > 0 and self.obs["status"] == 1:
            self.logger.info(f"Drop object: {self.dropping_object}")
            self.satisfied += self.dropping_object
            self.dropping_object = []
            if len(self.holding_objects_id) == 0:
                self.logger.info("successful drop!")
                self.plan = None

        ignore_obstacles = []
        ignore_ids = []
        self.with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x is None or x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        for x in self.force_ignore:
            self.with_character.append(x)

        for x in self.obs["oppo_held_objects"]:
            if x is None or x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        satisfied_obj_info = [self.object_info[x] for x in self.satisfied if x in self.object_info]

         

        
        self.agent_memory.update(
            obs,
            ignore_ids=ignore_ids,
            ignore_obstacles=ignore_obstacles,
            save_img=self.save_img,
        )

        if self.obs["status"] == 0:  # ongoing###
            return {"type": "ongoing"}

        self.get_new_object_list()
        # print(self.new_object_list)
        self.get_object_list()



        
        info = {
            "satisfied": self.satisfied,
            #"object_list": self.object_list,
            #"new_object_list": self.new_object_list,
            "current_room": self.current_room,
            #"visible_objects": self.filtered(self.obs["visible_objects"]),
            "visible_objects":self.visible_obj, #self.visible_obj是自己定义的
            "room_explored":self.rooms_explored,
            "obs": {
                k: v
                for k, v in self.obs.items()
                if k
                not in ["rgb", "depth", "seg_mask", "camera_matrix", "visible_objects"]
            },
        }

        # print(self.visible_obj)

        action = None
        lm_times = 0
        
        while action is None: #SUBPLAN DONE
            
            # if self.plan is None:
            if self.plan is None or self.message_received != []: #cobel
                
                # ==================================================================================
                        # 处理 oppo_holding_objects_zero
                for idx, obj in enumerate(self.oppo_holding_objects_zero):
                    # 从房间中移除该物品
                    for room in self.rooms_name:
                        to_remove = []
                        for idx2, obj_str in enumerate(self.my_object_per_room[room][0]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj['id'] == obj_id:
                                to_remove.append(idx2)
                        # 倒序删除，避免索引错乱
                        for idx2 in sorted(to_remove, reverse=True):
                            self.my_object_per_room[room][0].pop(idx2)


                        to_remove = []
                        for idx2, obj_str in enumerate(self.my_object_per_room[room][1]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj['id'] == obj_id:
                                to_remove.append(idx2)
                        # 倒序删除，避免索引错乱
                        for idx2 in sorted(to_remove, reverse=True):
                            self.my_object_per_room[room][1].pop(idx2)

                    # 清空逻辑：物品本身或包含物满足条件则清空
                    
                    if obj['id'] in self.satisfied or any(cid in self.satisfied for cid in obj['contained']):
                        self.oppo_holding_objects_zero[idx] = {
                            'type': None, 'id': None, 'name': None,
                            'contained': [None, None, None],
                            'contained_name': [None, None, None]
                        }


                # 处理 holding_objects_id（注意：这里是 ID 列表）
                for idx, hand_id in enumerate(self.with_character):
                    for room in self.rooms_name:
                        to_remove = []
                        for idx2, obj_str in enumerate(self.my_object_per_room[room][0]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj_id == hand_id:  # 对比 ID
                                to_remove.append(idx2)
                        for idx2 in sorted(to_remove, reverse=True):
                            self.my_object_per_room[room][0].pop(idx2)

                        to_remove = []
                        for idx2, obj_str in enumerate(self.my_object_per_room[room][1]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj_id == hand_id:  # 对比 ID
                                to_remove.append(idx2)
                        for idx2 in sorted(to_remove, reverse=True):
                            self.my_object_per_room[room][1].pop(idx2)



                # 处理 my_holding_first
                for idx, obj in enumerate(self.my_holding_first):
                    # 从对手房间中移除
                    for room in self.rooms_name:
                        to_remove = []
                        for idx2, obj_str in enumerate(self.oppo_object_per_room[room][0]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj['id'] == obj_id:
                                to_remove.append(idx2)
                        for idx2 in sorted(to_remove, reverse=True):
                            self.oppo_object_per_room[room][0].pop(idx2)

                        to_remove = []
                        for idx2, obj_str in enumerate(self.oppo_object_per_room[room][1]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj['id'] == obj_id:
                                to_remove.append(idx2)
                        for idx2 in sorted(to_remove, reverse=True):
                            self.oppo_object_per_room[room][1].pop(idx2)

                    # 清空逻辑
                    if obj['id'] in self.satisfied or any(cid in self.satisfied for cid in obj['contained']):
                        self.my_holding_first[idx] = {
                            'type': None, 'id': None, 'name': None,
                            'contained': [None, None, None],
                            'contained_name': [None, None, None]
                        }


                # 处理 oppo_holding_objects_first
                for idx, obj in enumerate(self.oppo_holding_objects_first):
                    # 从对手房间中移除
                    for room in self.rooms_name:
                        to_remove = []
                        for idx2, obj_str in enumerate(self.oppo_object_per_room[room][0]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj['id'] == obj_id:
                                to_remove.append(idx2)
                        for idx2 in sorted(to_remove, reverse=True):
                            self.oppo_object_per_room[room][0].pop(idx2)
                        to_remove = []
                        for idx2, obj_str in enumerate(self.oppo_object_per_room[room][1]):
                            obj_type, name, obj_id = self.parse_obj(obj_str)  # 修复命名
                            if obj['id'] == obj_id:
                                to_remove.append(idx2)
                        for idx2 in sorted(to_remove, reverse=True):
                            self.oppo_object_per_room[room][1].pop(idx2)

                    # 清空逻辑
                    if obj['id'] in self.satisfied or any(cid in self.satisfied for cid in obj['contained']):
                        self.oppo_holding_objects_first[idx] = {
                            'type': None, 'id': None, 'name': None,
                            'contained': [None, None, None],
                            'contained_name': [None, None, None]
                        }


                print("===========清理手中check===========")
                print(self.satisfied)
                print(self.oppo_holding_objects_first[0])
                #==============================更ixn===================================

                
                self.target_pos = None
                if lm_times > 0:
                    #print(info)
                    pass
                if lm_times > 3:
                    raise Exception(f"retrying LM_plan too many times")
                
                #COBEL - zhimin begin 从这里开始维护belief
                

                # ===== 第一步 观测更新 subplan done跳过 =====
                # observation,oppo_obs = self.observation2text(info)
                # self.visible_obj = {}
                # print("========visual and message=======\n")
                # print(observation['observation'])
                # print(oppo_obs)
                print(self.dialogue)
                # visual_observation = observation['observation'] #视觉描述
                
                dialogues = "" if self.dialogue != [] else "None"
                for mes in self.dialogue:
                    dialogues += mes + '\n'

                messages_received = "" if self.message_received != [] else "None"
                for mes in self.message_received:
                    messages_received += mes + '\n'

                #measurement update
                updated_zero_order_beliefs,updated_first_order_beliefs,self.opponent_subplans = self.LLM.update_beliefs(messages_received,dialogues)
                print("=========updated_first_beleifs==========")
                print(updated_first_order_beliefs)
                print("=========updated_zero_beleifs==========")
                print(updated_zero_order_beliefs)
                if updated_first_order_beliefs:
                    self.parse_belief_line('first',updated_first_order_beliefs)
                if updated_zero_order_beliefs:
                    self.parse_belief_line('zero',updated_zero_order_beliefs)
                
                print(self.goal_objects)

                self.dialogue = [] #处理完就清空
                self.message_received = []

                self.episode_logger.info(f"\nzero update:{updated_zero_order_beliefs}\nfirst update{updated_first_order_beliefs}")

                plan = None


                if self.opponent_subplans is not None: #说明消息发送了计划
                    my_progress = self.get_progress_description()
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_progress:{my_progress}")
                    print(my_progress)
                    zero_reason, self.my_subplan = self.LLM.passive_prediction_zero_order(my_progress,self.opponent_subplans)
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_zero:{zero_reason}")
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                    self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                    print("=========被动更新==========")
                    self.plan_logger.info("=========被动更新==========")
                    print(f"{self.agent_names[self.agent_id]}: {self.my_subplan}\n")
                    print(f"{self.agent_names[self.opponent_agent_id]}: {self.opponent_subplans}")
                    self.action_history = []
                    self.action_history_w_mes = []
                    self.opponent_subplans = None

                #===== 只在更新计划的时候走 =====
                if self.my_subplan is None or len(self.action_history) >= self.action_history_max_length:
                    print("=============\n", self.LLM.token_stats)
                    oppo_progress = self.get_oppo_progress()
                    my_progress = self.get_progress_description()
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} oppo_progress:{oppo_progress}")
                    self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_progress:{my_progress}")
                    print(my_progress)
                    print("\n")
                    print(oppo_progress)
                    #这个基本不可能到这


                    if self.opponent_subplans is not None: #说明消息发送了计划
                        zero_reason, self.my_subplan = self.LLM.passive_prediction_zero_order(my_progress,self.opponent_subplans)
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_zero:{zero_reason}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")


                    else: #就主动
                        zero_reason, self.my_subplan = self.LLM.prediction_zero_order(my_progress)
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_zero:{zero_reason}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")
                        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} my_subplan:{self.my_subplan}")



                        first_reason, self.opponent_subplans = self.LLM.prediction_first_order(oppo_progress)
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} predict_first:{first_reason}")
                        self.episode_logger.info(f"\n{self.agent_names[self.agent_id]} oppo_subplan:{self.opponent_subplans}")
                        self.plan_logger.info(f"\n{self.agent_names[self.agent_id]} oppo_subplan:{self.opponent_subplans}")


                        print("=========主动更新==========")
                        self.plan_logger.info("=========主动更新==========")
                        print(f"{self.agent_names[self.agent_id]}: {self.my_subplan}\n")
                        print(f"{self.agent_names[self.opponent_agent_id]}: {self.opponent_subplans}")
                        answer, reason, difference = self.LLM.coordination_aware(my_progress,oppo_progress,self.opponent_subplans,self.my_subplan)
                        #有必要就更新
                        if "YES" in answer.upper() and self.message_time < self.max_message_time:
                            message = self.comm(difference,self.my_subplan)
                            plan =  "send a message: " + message
                            self.comm_counts += 1
                            self.comm_chars += len(message)
                            self.message_time += 1
                            # self.plan_logger.info(
                            #     f"\n{self.agent_names[self.agent_id]}: low-level-plan:{plan}"
                            # )

                    #subplan更新后清空
                    self.action_history = [] #COBEL clean the action history
                    self.action_history_w_mes = []


                # ======subplan规划结束=======
                if plan is None:
                    plan = self.intuitive_planning()

                self.plan_logger.info(
                            f"\n{self.agent_names[self.agent_id]}: low-level-plan:{plan}"
                        )

                
                    
                if "SUBPLAN DONE" in plan: #TODO:have to program a fuzzy match in parse
                    self.my_subplan = None
                    # self.plan = None #其实不需要
                    continue

                if plan is None:  # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"

                self.plan = plan

                if not plan.startswith('send a message:'):
                    self.action_history.append(
                        f"{plan} at step {self.num_frames}"
                    )
                    self.message_time = 0
                self.action_history_w_mes.append(f"{'send a message' if plan.startswith('send a message:') else plan}")
                lm_times += 1
            if self.plan.startswith("go to"):
                action = self.gotoroom()
            elif self.plan.startswith("explore"):
                self.explore_count = 0
                action = self.goexplore()
            elif self.plan.startswith("go grasp"):
                action = self.gograsp()
            elif self.plan.startswith("put"):
                action = self.putin()
            elif self.plan.startswith("transport"):
                action = self.goput()
            #    self.with_character = [self.agent_id]
            elif self.plan.startswith("send a message:"):
                # 发送消息动作
                action = {
                    "type": 6,  # 动作类型6表示发送消息
                    "message": " ".join(
                        self.plan.split(" ")[3:]
                    ),  # 提取消息内容，去掉"send a message:"前缀
                } #
                self.plan = None  # 清除当前计划，准备执行下一个动作,other actions will maintain the plan
            elif self.plan.startswith("wait"):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        info.update({"action": action, "plan": self.plan})
        if self.debug:
            self.logger.info(f"{self.agent_names[self.agent_id]}: {self.plan}")
            self.logger.debug(info)
        self.last_action = action
        return action
    
    def get_tokens(self):
        return self.LLM.token_stats

    def get_com_counts(self):
        return self.comm_counts

    def get_com_chars(self):
        return self.comm_chars

    def get_api_num(self):
        return self.LLM.api
    
    def parse_belief_line(self,belief_type,beliefs):
        print("======开始解析信念============")
        print(beliefs)
        formatted_beliefs = []
        
        #对手拿了什么
        #first
        my_container_first =  None
        oppo_container_first = None
        #zero

        oppo_container_zero = None
        
        print(self.rooms_name)
        for line in beliefs.splitlines():
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


                believer_agent = tokens[2]  # first order 中，第二个 agent 是“相信者”（如 Bob）

                #房间情况
                if "explore" in predicate:
                    if self.parse_room(subject) is None:
                        continue
                    room_str,room_name,room_id = self.parse_room(subject)
                    if room_str not in self.rooms_name:
                        continue
                    if room_str not in self.oppo_rooms_explored.keys():
                        if 'part' in obj:
                            self.oppo_rooms_explored.update({f'{room_str}':'part'})
                            formatted_beliefs.append(f"{room_str}':'part'")
                        if 'all' in obj:
                            self.oppo_rooms_explored.update({f'{room_str}':'all'})
                            formatted_beliefs.append(f"{room_str}':'all'")
                    else:
                        if 'part' in obj:
                            self.oppo_rooms_explored[room_str] = 'part'
                            formatted_beliefs.append(f"{room_str}':'part'")
                        if 'all' in obj:
                            self.oppo_rooms_explored[room_str] = 'all'
                            formatted_beliefs.append(f"{room_str}':'all'")


                #智能体
                for i, agent_name in enumerate(self.agent_names):
                    if agent_name in subject.capitalize():
                        agent_id = i
                        if agent_id == self.opponent_agent_id:
                            if 'hold' in predicate:

                                if self.parse_obj(obj) is None:
                                    continue


                                obj_str, name, id = self.parse_obj(obj)
                                obj_type = 0

                                if name.lower() not in self.goal_objects.keys():
                                    oppo_container_first = int(id)
                                    obj_type = 1

                                
                                hold_dic = {'id': int(id), 'type': obj_type, 'name': name.lower(), 'contained': [None, None, None], 'contained_name': [None, None, None]}
                                for hand_id , hand in enumerate(self.oppo_holding_objects_first):
                                    if hand['id'] is None:
                                        self.oppo_holding_objects_first[hand_id] = hold_dic
                                        break
                                formatted_beliefs.append(f"{hold_dic}")

                            elif 'at' in predicate:
                                if self.parse_room(obj) is None:
                                    continue
                                room_str, name, id = self.parse_room(obj)
                                if room_str not in self.rooms_name:
                                    continue
                                self.oppo_current_room_first = room_str

                                formatted_beliefs.append(f"oppo AT{room_str}")
                        elif agent_id == self.agent_id:
                            if 'hold' in predicate:
                                #判断容器
                                if self.parse_obj(obj) is None:
                                    continue
                                obj_str, name, id = self.parse_obj(obj)
                                obj_type = 0
                                if name.lower() not in self.goal_objects.keys():
                                    my_container_first = int(id)
                                    obj_type = 1
                                else:
                                    obj_type = 0
                                hold_dic = {'id': int(id), 'type': obj_type, 'name': name.lower(), 'contained': [None, None, None], 'contained_name': [None, None, None]}
                                for hand_id , hand in enumerate(self.my_holding_first):
                                    if hand['id'] is None:
                                        self.my_holding_first[hand_id] = hold_dic
                                        break
                                formatted_beliefs.append(f"my_first HOLD {hold_dic}")
                            elif 'at' in predicate:
                                if self.parse_room(obj) is None:
                                    continue
                                room_str, name, id = self.parse_room(obj)
                                if room_str not in self.rooms_name:
                                    continue
                                self.my_current_room_first = room_str
                                formatted_beliefs.append(f"my_first current room:{room_str}")



                #物体判断
                if predicate == 'in' and (subject not in self.agent_names):
                    
                    if self.parse_room(obj) is None or self.parse_obj(subject) is None:
                        continue


                    room_str,room_name,room_id = self.parse_room(obj)
                    obj_str,obj_name, obj_id = self.parse_obj(subject)

                    if room_str not in self.rooms_name:
                        continue
                    if 'bed' in subject:
                        self.my_object_per_room[room_str][2].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str.lower()} IN {room_str}")
                        continue
                    

                    if obj_name.lower() in self.goal_objects.keys():
                        if obj_str.lower() not in self.oppo_object_per_room[room_str][0]:
                            self.oppo_object_per_room[room_str][0].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str.lower()} IN {room_str}")

                    elif 'bed' in obj_name.lower():
                        if obj_str.lower() not in self.oppo_object_per_room[room_str][2]:
                            self.oppo_object_per_room[room_str][2].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str} IN {room_str}")

                    else:
                        if obj_str.lower() not in self.oppo_object_per_room[room_str][1]:
                            self.oppo_object_per_room[room_str][1].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str} IN {room_str}")
            else:
                try:
                    believe_idx = tokens.index('believe')  # 不区分大小写
                except ValueError:
                    continue
                
                # if tokens[believe_idx - 1] != self.agent_names[self.agent_id].lower():
                #     continue  # 不匹配 agent_name

                belief_tokens = tokens[believe_idx + 1:]      # 用原始 tokens 提取内容

                if len(belief_tokens) < 3:
                    continue
                subject = belief_tokens[0]
                predicate = belief_tokens[1]
                obj = belief_tokens[2]
                believer_agent = tokens[0]  # zero order 中的主体（如 Alice）

                #房屋
                if "explore" in predicate:
                    if self.parse_room(subject) is None:
                        continue
                    room_str,room_name,room_id = self.parse_room(subject)
                    if room_str not in self.rooms_name:
                        continue
                    if room_str not in self.my_rooms_explored.keys():
                        if 'part' in obj:
                            self.my_rooms_explored.update({f'{room_str}':'part'})
                            formatted_beliefs.append(f"{room_str}':'part'")
                        if 'all' in obj:
                            self.my_rooms_explored.update({f'{room_str}':'all'})
                            formatted_beliefs.append(f"{room_str}':'all'")
                    else:
                        if 'part' in obj:
                            self.my_rooms_explored[room_str] = 'part'
                            formatted_beliefs.append(f"{room_str}':'part'")
                        if 'all' in obj:
                            self.my_rooms_explored[room_str] = 'all'
                            formatted_beliefs.append(f"{room_str}':'all'")
                #物体
                if 'in' in predicate and (subject not in self.agent_names):

                    if self.parse_room(obj) is None or self.parse_obj(subject) is None:
                        continue
                    

                    room_str,room_name,room_id = self.parse_room(obj)
                    obj_str,obj_name, obj_id = self.parse_obj(subject)

                    if room_str not in self.rooms_name: #同时避免了容器
                        continue
                    if 'bed' in subject:
                        self.my_object_per_room[room_str][2].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str} IN {room_str}")
                        continue
                    
                    
                    if obj_name.lower() in self.goal_objects.keys():
                        if obj_str.lower() not in self.my_object_per_room[room_str][0]:
                            self.my_object_per_room[room_str][0].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str} IN {room_str}") 

                    elif 'bed' in obj_name.lower():
                        if obj_str.lower() not in self.my_object_per_room[room_str][2]:
                            self.my_object_per_room[room_str][2].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str} IN {room_str}")

                    else:
                        if obj_str.lower() not in self.my_object_per_room[room_str][1]:
                            self.my_object_per_room[room_str][1].append(obj_str.lower())
                        formatted_beliefs.append(f"{obj_str} IN {room_str}")
                #智能体        
                for i, agent_name in enumerate(self.agent_names):
                    if agent_name in subject.capitalize():
                        if i == self.opponent_agent_id:
                            if self.parse_obj(obj) is None:
                                    continue
                            if 'hold' in predicate:
                                #判断容器
                                obj_str, name, id = self.parse_obj(obj)
                                if name.lower() not in self.goal_objects.keys():
                                    obj_type = 1
                                    oppo_container_zero = int(id)
                                else:
                                    obj_type = 0
                                hold_dic = {'id': int(id), 'type': obj_type, 'name': name.lower(), 'contained': [None, None, None], 'contained_name': [None, None, None]}
                                for hand_id , hand in enumerate(self.oppo_holding_objects_zero):
                                    if hand['id'] is None:
                                        self.oppo_holding_objects_zero[hand_id] = hold_dic
                                        break
                            elif 'at' in predicate:
                                if self.parse_obj(obj) is None:
                                    continue
                                room_str, name, id = self.parse_obj(obj)
                                if room_str not in self.rooms_name:
                                    continue
                                self.oppo_current_room_zero = room_str
                                formatted_beliefs.append(f"oppo_zero AT {room_str}")
             


        if belief_type == 'first':
            if oppo_container_first != None or my_container_first != None:
                oppo_contain = []
                my_contain = []
                oppo_contain_name = []
                my_contain_name = []
                for line in beliefs.splitlines():
                    line = line.strip()

                    tokens = line.split()
                    if len(tokens) < 3:
                        continue
                    
                    tokens = [t.lower() for t in tokens]
                
                    if tokens.count('believe') < 2:
                        continue
                    first_believe_idx = tokens.index('believe')
                    second_believe_idx = tokens.index('believe', first_believe_idx + 1)
                    belief_tokens = tokens[second_believe_idx + 1:]


                    if len(belief_tokens) < 3:
                        continue
                    

                    subject = belief_tokens[0]
                    predicate = belief_tokens[1]
                    obj = belief_tokens[2]

                    if 'in' in predicate:
                        if self.parse_obj(obj) is None or self.parse_obj(subject) is None:
                            continue

                        con_str,con_name,con_id = self.parse_obj(obj)
                        obj_str,obj_name,obj_id = self.parse_obj(subject)
                        if con_str in self.rooms_name:
                            continue
                        if oppo_container_first == int(con_id):
                            for index, dic in enumerate(self.oppo_holding_objects_first):
                                if dic['id'] == int(con_id):
                                    for idx, obj_loc in enumerate(dic['container']):
                                        if obj_loc is None:
                                            dic['contained'][idx] = obj_id
                                            dic['contained_name'][idx] = obj_name
                                    formatted_beliefs.append(dic)

                        if my_container_first == int(con_id):
                            for index, dic in enumerate(self.my_holding_first):
                                if dic['id'] == int(con_id):
                                    for idx, obj_loc in enumerate(dic['container']):
                                        if obj_loc is None:
                                            dic['contained'][idx] = obj_id
                                            dic['contained_name'][idx] = obj_name
                                    formatted_beliefs.append(dic)



            


        else:
            if oppo_container_zero != None:
                oppo_contain = []
                oppo_contain_name = []
                for line in beliefs.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = line.split()
                    if len(tokens) < 3:
                        continue
                    
                    tokens = [t.lower() for t in tokens]
                    if tokens.count('believe') < 1:
                        continue
                    
                    believe_idx = tokens.index('believe')  # 不区分大小写

                    belief_tokens = tokens[believe_idx + 1:]      # 用原始 tokens 提取内容
                    if len(belief_tokens) < 3:
                        continue
                    subject = belief_tokens[0]
                    predicate = belief_tokens[1]
                    obj = belief_tokens[2]
                    believer_agent = tokens[0]

                    if 'in' in predicate:
                        if self.parse_obj(obj) is None or self.parse_obj(subject) is None:
                            continue

                        con_str,con_name,con_id = self.parse_obj(obj)
                        obj_str,obj_name,obj_id = self.parse_obj(subject)
                        if con_str in self.rooms_name:
                            continue
                        if oppo_container_zero == int(con_id):
                            for index, dic in enumerate(self.oppo_holding_objects_zero):
                                if dic['id'] == int(con_id):
                                    for idx, obj_loc in enumerate(dic['contained']):
                                        if obj_loc is None:
                                            dic['container'][idx] = obj_id
                                            dic['container_name'][idx] = obj_name
                                    formatted_beliefs.append(dic)
        print("----------------解析结果----------------------\n")
        belief_string = ""
        for belief_formatted in formatted_beliefs:
            print(belief_formatted,"\n")
            belief_string += belief_formatted
            belief_string += "\n"
            self.episode_logger.info(f"{belief_type}信念更新:\n{belief_string}") 
        print("==================解析结束============================\n")

    def parse_room(self, text):
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
        name_capitalized = name_raw.capitalize()  # "livingroom" → "Livingroom"

        # 格式化输出字符串
        formatted = f"<{name_capitalized}> ({id_str})"

        return formatted, name_capitalized, id_str
    
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
        formatted = f"<{name_raw}>({id_str})"

        return formatted, name_raw, id
    
    def get_oppo_progress(self):
        s = f"I've taken {self.steps}/3000 steps. "

        sss = {}
        for room, obj_list in self.oppo_object_per_room.items():
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
                    ss = ', '.join([f"{x}" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container {x}"
                else:
                    ss = ', '.join([f"{x}" for x in cons])
                    s_con = f"containers " + ss
            if len(obj_list[2]) > 0:
                s_bed = 'the goal position bed'
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += 'nothing'
            elif s_obj != "" and s_con != "" and s_bed == "":
                sr += s_obj + ', and ' + s_con
            elif s_obj != "" and s_con == "" and s_bed != "":
                sr += s_obj + ', and ' + s_bed
            elif s_obj == "" and s_con != "" and s_bed != "":
                sr += s_con + ', and ' + s_bed
            elif s_obj != "" and s_con != "" and s_bed != "":
                sr += s_obj + ', ' + s_con + ', and ' + s_bed
            else:
                sr += s_obj + s_con + s_bed
            sss[room] = sr

        satisfied_obj_info = [self.object_info[x] for x in self.satisfied if x in self.object_info]

        if len(satisfied_obj_info) == 0:
            s += ""
        else:
            s += f"{'I' if self.single else 'We'}'ve already transported "
            unique_satisfied = []
            for x in satisfied_obj_info:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x['type'] == 0]) == 0:
                s += 'nothing'
            s += ', '.join([f"<{x['name']}> ({x['id']})" for x in unique_satisfied if x['type'] == 0])
            s += ' to the bed. '

        s_hold = ["", ""]
        for i, obj in enumerate(self.oppo_holding_objects_first):
            if obj['type'] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj['type'] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj['contained']):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = 'nothing'
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

        if self.oppo_holding_objects_first[0]["type"] == 0 and self.oppo_holding_objects_first[1]['type'] == 0:
            s += f"I'm holding two target objects <{self.oppo_holding_objects_first[0]['name']}> ({self.oppo_holding_objects_first[0]['id']}) and <{self.oppo_holding_objects_first[1]['name']}> ({self.oppo_holding_objects_first[1]['id']}). "
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "I'm holding nothing. "
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"I'm holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"I'm holding {s_hold[0]}{s_hold[1]}"

        # print(self.current_room, self.obj_per_room)
        if self.oppo_current_room_first not in self.oppo_rooms_explored: pred_room = 'none'
        else: pred_room = self.oppo_rooms_explored[self.oppo_current_room_first]
        if pred_room != 'all' and sss[self.oppo_current_room_first] == 'nothing':
            s += f"I'm in the {self.oppo_current_room_first}, where I've explored {pred_room} of it. "
        else:
            s += f"I'm in the {self.oppo_current_room_first}, where I've explored {pred_room} of it and found {sss[self.oppo_current_room_first]}. "
        ### opponent modeling
        if not self.single:
            s_hold = ["", ""]
            for i, obj in enumerate(self.my_holding_first):
                if obj['type'] == 0:
                    s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
                elif obj['type'] == 1:
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj['contained']):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = 'nothing'
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "
            if self.my_holding_first[0]["type"] == 0 and self.my_holding_first[1]['type'] == 0:
                ss = f"two target objects <{self.my_holding_first[0]['name']}> ({self.my_holding_first[0]['id']}) and <{self.my_holding_first[1]['name']}> ({self.my_holding_first[1]['id']}). "
            if s_hold[0] == "" and s_hold[1] == "":
                ss = "nothing. "
            elif s_hold[0] != "" and s_hold[1] != "":
                ss = f"{s_hold[0][:-2]}, and {s_hold[1]}"
            else:
                ss = f"{s_hold[0]}{s_hold[1]}"

            if self.my_current_room_first is None:
                s += f"I don't know where {self.agent_names[self.agent_id]} is. "
            elif self.my_current_room_first == self.oppo_current_room_first:
                s += f"I also see {self.agent_names[self.agent_id]} here in the {self.oppo_current_room_first}, he is holding {ss}"
            else:
                s += f"Last time I saw {self.agent_names[self.agent_id]} was in the {self.my_current_room_first}, he was holding {ss}"

            for room in self.rooms_name:
                if room == self.oppo_current_room_first:
                    continue
                #s += f"I've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
                if room not in self.oppo_rooms_explored: pred_room = 'none'
                else: pred_room = self.oppo_rooms_explored[room]
                if pred_room != 'all' and sss[room] == 'nothing':
                    s += f"I've explored {pred_room} of the {room}. "
                else:
                    s += f"I've explored {pred_room} of the {room}, and I found {sss[room]} there. "
        return s
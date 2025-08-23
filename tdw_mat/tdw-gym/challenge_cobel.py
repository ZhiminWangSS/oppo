##
import argparse
import os
import json
import gym
import time
import pickle
import logging
import sys


# add this dictionary to python env path:
base_path = os.getcwd()
print(base_path)
current_dir = os.path.dirname(__file__)

sys.path.append(base_path)
sys.path.append(current_dir)
from belief_symbolic_representation import belief_builder #COBEL init the rules bulider
from h_agent import H_agent
# from lm_agent import lm_agent
from lm_agent_cobel import lm_agent_cobel

BeliefBuilder = belief_builder.BeliefBuilder
# 注册测试环境
gym.envs.registration.register(id="transport_challenge_MA", entry_point="tdw_gym:TDW")


class Challenge:
    """
    多智能体运输挑战环境管理类
    用于管理环境、执行评估和记录结果
    """

    def __init__(
        self,
        logger,
        time_logger,
        port,
        data_path,
        output_dir,
        number_of_agents=2,
        max_frames=3000,
        launch_build=True,
        screen_size=512,
        data_prefix="dataset/nips_dataset/",
        gt_mask=True,
        save_img=True,
    ):
        """
        初始化挑战环境

        参数:
            logger: 日志记录器
            port: 环境端口号
            data_path: 数据文件路径
            output_dir: 输出目录
            number_of_agents: 智能体数量（默认2个）
            max_frames: 最大帧数（默认3000）
            launch_build: 是否启动构建（默认True）
            screen_size: 屏幕大小（默认512）
            data_prefix: 数据集前缀路径
            gt_mask: 是否使用真实掩码（默认True）
            save_img: 是否保存图像（默认True）
        """
        self.env = gym.make(
            "transport_challenge_MA",
            port=port,
            number_of_agents=number_of_agents,
            save_dir=output_dir,
            max_frames=max_frames,
            launch_build=launch_build,
            screen_size=screen_size,
            data_prefix=data_prefix,
            gt_mask=gt_mask,
        )
        self.gt_mask = gt_mask
        self.logger = logger
        self.time_logger = time_logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.save_img = save_img
        print(data_path)
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        self.logger.info("done")

        #COBEL
        self.rules_prompt_path = "../belief_symbolic_representation/construct.csv"
        self.rules_output_file  = "rules.txt"
        self.challenge_description = "In this domain, two agents must collaborate to transporting as many target objects as possible to a designated bed location(unknown at beginning)(totice that bed is an important entity class as destination with only attribution: location) using available containers given a shared goal(e.g. like transport 2 apples and one pen to the bed). Each target object corresponds to a task, which can be either complete or incomplete. Agents can autonomously plan subgoals based on the overall objective.Objects, containers, and agents all have a location attribute. Initially, objects and containers are scattered across various rooms. A container can hold up to three objects, and an agent can carry up to two items at a time—these may be objects or containers. The domain consists of multiple rooms, and agents are deployed within this multi-room space, where they can freely move and explore. Each room’s exploration state is categorized as None (unexplored), Part (partially explored), or All (fully explored)."

    #COBEL rules builder
    def rules_builder(self):
        builder = BeliefBuilder(self.rules_prompt_path)
        os.makedirs("./belief_rules",exist_ok=True)
        path = os.path.join("./belief_rules",self.rules_output_file)
        # if the file has content
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        builder.build_complete_belief(self.challenge_description,path)




    def submit(self, agents, logger, eval_episodes):
        """
        执行智能体评估过程

        参数:   
            agents: 智能体列表
            logger: 日志记录器
            eval_episodes: 要评估的回合列表

        返回:
            float: 平均完成率
        """
        ##COBEL init the rules builder
        self.rules_builder()

        total_finish = 0.0
        if eval_episodes[0] == -1:
            eval_episodes = range(len(self.data))
        num_eval_episodes = len(eval_episodes)
        # 无循环部分
        start = time.time()
        results = {}
        total_tokens = [0,0]
        total_com = [0,0]
        for i, episode in enumerate(eval_episodes):
            #COBEL belief info logger
            episode_logger = init_episode_logs(self.output_dir, episode)

            print(f"当前执行的episode为：{episode}")
            start_time = time.time()
            # 检查是否已经评估过该回合
            if os.path.exists(
                os.path.join(self.output_dir, str(episode), "result_episode.json")
            ):
                with open(
                    os.path.join(self.output_dir, str(episode), "result_episode.json"),
                    "r",
                ) as f:
                    result = json.load(f)
                total_finish += result["finish"] / result["total"]
                results[episode] = result
                continue
            # The episode has been evaluated before

            # 创建输出目录
            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
            self.logger.info(
                "Episode {} ({}/{})".format(episode, i + 1, num_eval_episodes)
            )
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")

            # 重置环境
            state, info, env_api = self.env.reset(
                seed=self.data[episode]["seed"],
                options=self.data[episode],
                output_dir=os.path.join(self.output_dir, str(episode)),
            )

            # 重置每个智能体
            for id, agent in enumerate(agents):
                if type(env_api) == list:
                    curr_api = env_api[id]
                else:
                    curr_api = env_api
                if info["goal_description"] is not None:
                    if agent.agent_type == "h_agent":
                        agent.reset(
                            goal_objects=info["goal_description"],
                            output_dir=os.path.join(self.output_dir, str(episode)),
                            env_api=curr_api,
                            agent_color=info["agent_colors"][id],
                            agent_id=id,
                            gt_mask=self.gt_mask,
                            save_img=self.save_img,
                        )
                    elif agent.agent_type == "lm_agent":
                        agent.reset(
                            obs=state[str(id)],
                            goal_objects=info["goal_description"],
                            output_dir=os.path.join(self.output_dir, str(episode)),
                            env_api=curr_api,
                            agent_color=info["agent_colors"][id],
                            agent_id=id,
                            rooms_name=info["rooms_name"],
                            gt_mask=self.gt_mask,
                            save_img=self.save_img,
                        )
                    elif agent.agent_type == "lm_agent_cobel":
                        agent.reset(
                            obs=state[str(id)],
                            goal_objects=info["goal_description"],
                            output_dir=os.path.join(self.output_dir, str(episode)),
                            env_api=curr_api,
                            agent_color=info["agent_colors"][id],
                            agent_id=id,
                            rooms_name=info["rooms_name"],
                            gt_mask=self.gt_mask,
                            save_img=self.save_img,
                            episode_logger=episode_logger
                        )
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir=os.path.join(self.output_dir, str(episode)))
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")

            # 执行评估过程
            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            local_reward = 0.0
            while not done:
                step_num += 1
                actions = {}
                # 保存图片
                if self.save_img:
                    self.env.save_images(
                        os.path.join(self.output_dir, str(episode), "Images")
                    )
                for agent_id, agent in enumerate(agents):
                    
                    # print(f"agent状态：{state[str(agent_id)]}")
                    actions[str(agent_id)] = agent.act(state[str(agent_id)])
                    # 执行大模型推理获得动作
                    print(f"agent_id:{agent_id}\n",agent.get_tokens())
                state, reward, done, info = self.env.step(actions)
                local_reward += reward
                local_finish = self.env.check_goal()
                self.logger.info(
                    f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}"
                )
                if done:
                    break
                # for agent_id, agent in enumerate(agents):
                #     if actions[str(agent_id)] == "send a message":
                #         communication_num += 1
                #         print("Communication action taken by agent:", agent_id)


            # 记录结果
            for agent_id,agent in enumerate(agents):
                print(f"{agent_id}:{agent.get_com_cost()}")
                total_com[agent_id] += agent.get_com_cost()
                total_tokens[agent_id] += agent.get_tokens()
            total_finish += local_finish[0] / local_finish[1]
            result = {
                "finish": local_finish[0],
                "total": local_finish[1],
                "step_num": step_num,
                "comunication_tokens":total_tokens[0]+total_tokens[1],
                "agent_0_com_tokens":total_tokens[0], #character
                "agent_1_com_tokens":total_tokens[1], #character
                "agent_0_com_num":total_com[0], #count TODO
                "agent_1_com_num":total_com[1],
                "tokens_per_step":(total_tokens[0]+total_tokens[1])/step_num
                #"communication num": communication_num
            }

            with open(
                os.path.join(self.output_dir, str(episode), "result_episode.json"), "w"
            ) as f:
                json.dump(result, f)
            results[episode] = result

        # 计算并保存最终结果
        avg_finish = total_finish / num_eval_episodes
        results = {"episode_results": results, "avg_finish": avg_finish}
        with open(os.path.join(self.output_dir, "eval_result.json"), "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"eval done, avg transport rate {avg_finish}")
        self.logger.info("time: {}".format(time.time() - start))
        return avg_finish

    def close(self):
        """
        关闭环境，释放资源
        """
        self.env.close()

#COBEL basic logger for llm 
def init_logs(output_dir, name="simple_example"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler() # 控制台输出
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 新增一个logger用于记录时间信息
    time_logger = logging.getLogger(f"{name}_time")
    time_logger.setLevel(logging.DEBUG)
    time_fh = logging.FileHandler(os.path.join(output_dir, "time.log"))
    time_fh.setLevel(logging.DEBUG)
    time_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    time_fh.setFormatter(time_formatter)
    time_logger.addHandler(time_fh)
    
    

    return logger, time_logger

#COBEL logger only for belief and observation
def init_episode_logs(output_dir, episode):
    """
    初始化每个episode的日志记录器
    """
    episode_dir = os.path.join(output_dir, str(episode))
    os.makedirs(episode_dir, exist_ok=True)
    
    episode_logger = logging.getLogger(f"episode_{episode}")
    episode_logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(os.path.join(episode_dir, f"llm_plan_{episode}.log"))
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    
    episode_logger.addHandler(fh)
    
    return episode_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type=str, default="try")
    parser.add_argument("--run_id", type=str, default="run_0")
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="dataset/dataset_train/")
    parser.add_argument("--port", default=1071, type=int)
    parser.add_argument("--agents", nargs="+", type=str, default=("h_agent",))
    parser.add_argument(
        "--eval_episodes",
        nargs="+",
        default=(-1,),
        type=int,
        help="which episodes to evaluate on",
    )
    parser.add_argument(
        "--max_frames", default=3000, type=int, help="max frames per episode"
    )
    parser.add_argument("--no_launch_build", action="store_true")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_gt_mask", action="store_true")
    # LLM parameters
    parser.add_argument(
        "--source",
        default="openai",
        choices=["hf", "openai", "deepseek"],
        help="openai API or load huggingface models",
    )
    parser.add_argument(
        "--lm_id",
        default="gpt-3.5-turbo",
        help="name for openai engine or huggingface model name/path",
    )
    parser.add_argument(
        "--prompt_template_path",
        default="LLM/prompt_single.csv",
        help="path to prompt template file",
    )
    parser.add_argument("--t", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument(
        "--cot", action="store_true", help="use chain-of-thought prompt"
    )
    parser.add_argument(
        "--echo", action="store_true", help="to include prompt in the outputs"
    )
    parser.add_argument("--screen_size", default=512, type=int)
    parser.add_argument(
        "--no_save_img", action="store_true", help="do not save images", default=False
    )
    args = parser.parse_args()

    args.number_of_agents = len(args.agents)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok=True)
    logger,time_logger = init_logs(args.output_dir)#COBEL normal logger
    

    challenge = Challenge(
        logger,
        time_logger,
        args.port,
        args.data_path,
        args.output_dir,
        args.number_of_agents,
        args.max_frames,
        not args.no_launch_build,
        screen_size=args.screen_size,
        data_prefix=args.data_prefix,
        gt_mask=not args.no_gt_mask,
        save_img=not args.no_save_img,
    )
    agents = []
    for i, agent in enumerate(args.agents):
        if agent == "h_agent":
            agents.append(H_agent(i, logger, args.max_frames, args.output_dir))
        elif agent == "lm_agent":
            agents.append(lm_agent(i, logger, args.max_frames, args, args.output_dir))
        elif agent == "lm_agent_cobel":
            agents.append(lm_agent_cobel(i, logger, args.max_frames, args, args.output_dir))
        else:
            pass
    try:
        print("yes")
        challenge.submit(agents, logger, args.eval_episodes)
        # 提交进入chanllenge遍历执行
    finally:
        challenge.close()


if __name__ == "__main__":
    main()

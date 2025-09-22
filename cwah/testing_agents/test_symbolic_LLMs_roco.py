import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{curr_dir}/..')
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment_roco import UnityEnvironment
from agents.LLM_agent_roco import LLM_agent
from arguments import get_args
from cwah.algos.arena_mp2_roco import ArenaMP
import subprocess

def kill_process_on_port(port):
    try:
        # 执行 lsof 命令获取占用指定端口的进程 PID
        result = subprocess.run(
            ['lsof', '-t', '-i', f':{port}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"端口 {port} 上没有进程被占用或 lsof 执行失败。")
            return
        
        pids = result.stdout.strip().split('\n')
        pids = [pid for pid in pids if pid]  # 过滤空行

        if not pids:
            print(f"没有找到占用端口 {port} 的进程。")
            return

        print(f"找到占用端口 {port} 的进程 PID: {pids}")

        # 使用 kill -9 终止每个进程
        for pid in pids:
            subprocess.run(['kill', '-9', pid])
            print(f"已终止 PID {pid} 的进程。")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    args = get_args()
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    # with open("test_env.json", "w") as f:
    #     json.dump(env_task_set, f, indent=4)

    args.record_dir = f'../test_results/{args.mode}' # set the record_dir right!
    Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    if "image" in args.obs_type:
        os.system("Xvfb :98 & export DISPLAY=:98")
        import time
        time.sleep(3) # ensure Xvfb is open
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        executable_args = {
                        'file_name': args.executable_file,
                        'x_display': '98',
                        'no_graphics': False,
                        'timeout_wait': 5000,
        }
    else:
        executable_args = {
                        'file_name': args.executable_file,
                        'no_graphics': True,
        }

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)
    num_tries = args.num_runs
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]


    def env_fn(env_id):
        return UnityEnvironment(num_agents=2,
                               max_episode_length=args.max_episode_length,
                               port_id=env_id,
                               env_task_set=env_task_set,
                               agent_goals=['LLM', 'LLM'],
                               observation_types=[args.obs_type, args.obs_type],
                               use_editor=args.use_editor,
                               executable_args=executable_args,
                               base_port=args.base_port)

    args_agent1 = {
        'agent_id': 1,
        'char_index': 0,
        'args': args,
    }
    args_agent2 = {
        'agent_id': 2,
        'char_index': 1,
        'args': args,
    }

    agents = [lambda x, y: LLM_agent(**args_agent1), lambda x, y: LLM_agent(**args_agent2)]
    

    # copy the code below to record results
    if args.num_per_task != 10:
        test_episodes = args.test_task
    else:
        test_episodes = episode_ids
    for iter_id in range(num_tries):
        steps_list, failed_tasks = [], []
        total_tokens = {}
        total_comm_counts = {}
        total_comm_chars = {}
        if not os.path.isfile(args.record_dir + '/results.pik'):
            test_results = {}
        else:
            test_results = pickle.load(open(args.record_dir + '/results.pik', 'rb'))

        current_tried = iter_id

        for episode_id in test_episodes:
            kill_process_on_port(args.base_port)
            kill_process_on_port(args.base_port)
            arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents, args.record_dir, args.debug)
            curr_log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(
                env_task_set[episode_id]['task_id'],
                env_task_set[episode_id]['task_name'],
                iter_id)

            if os.path.isfile(curr_log_file_name):
                with open(curr_log_file_name, 'rb') as fd:
                    file_data = pickle.load(fd)
                S[episode_id].append(file_data['finished'])
                L[episode_id].append(max(len(file_data['action'][0]), len(file_data['action'][1])))

                test_results[episode_id] = {'S': S[episode_id],
                                            'L': L[episode_id]}
                continue

            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2

            is_finished = 0
            steps = 250
            # try:
            arena.reset(episode_id)
            success, steps, saved_info = arena.run()


            #COBEL episode count
            episode_0_com_count = arena.agents[0].comm_counts
            episode_1_com_count = arena.agents[1].comm_counts
            episode_0_api = arena.agents[0].get_api_num()
            episode_1_api = arena.agents[1].get_api_num()
            episode_0_token_stats = arena.agents[0].get_token_stats()
            episode_1_token_stats = arena.agents[1].get_token_stats()
            print('-------------------------------------')
            print('success' if success else 'failure')
            print('steps:', steps)
            print('-------------------------------------')
            if not success:
                failed_tasks.append(episode_id)
            else:
                steps_list.append(steps)
            is_finished = 1 if success else 0
            log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(saved_info['task_id'],
                                                                                saved_info['task_name'],
                                                                                current_tried)

            if len(saved_info['obs']) > 0:
                pickle.dump(saved_info, open(log_file_name, 'wb'))
            else:
                with open(log_file_name, 'w+') as f:
                    f.write(json.dumps(saved_info, indent=4))
            # except:
            #     # ipdb.set_trace()
            #     arena.reset_env()

            S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            average_calls_per_discussion = 0

            for call in arena.get_calls():
                average_calls_per_discussion += (call)

            average_calls_per_discussion /= len(arena.get_calls())

            result_dic = {'S': S[episode_id],
                                        'L': L[episode_id],
                                        'symbolic_roco': {
                                            'episode_0_com': episode_0_com_count,
                                            'episode_1_com': episode_1_com_count,
                                            'episode_0_api': episode_0_api,
                                            'episode_1_api': episode_1_api,
                                            'episode_0_tokens': {"prompt_tokens":episode_0_token_stats[0],"completion_tokens":episode_0_token_stats[1]},
                                            'episode_1_tokens': {"prompt_tokens":episode_1_token_stats[0],"completion_tokens":episode_1_token_stats[1]},
                                        }}
            test_results[episode_id] = result_dic
            #==========================
            
            
            # 保存为json
            json_path = os.path.join(args.record_dir, f"{episode_id}_result.json")
            with open(json_path, "w") as f_json:
                json.dump(result_dic, f_json, indent=4)
        kill_process_on_port(args.base_port)
        kill_process_on_port(args.base_port)
        args.base_port += 1
        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        pickle.dump(test_results, open(args.record_dir + '/results.pik', 'wb'))


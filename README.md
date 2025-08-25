## TODO

ongoing：zhimin测试全部大参数模型的cobel效果，还在debug中

shaokang:
- 对intuitive planning, misalignment aware, measurement update三个模块替换为小模型进行测试，测试模型：
  - 大模型先统一deepseekv3 (如果阿里云有这个) 没有qwen3也可以
  - 注意！！！：关闭思考模式（参考qwen官方文档），qwen3系统都是混合思考模型；另外思考模式的token是单独计算在reasoning_token的
  - qwen3 - 4b 首推问题都应该不大，可能输出规范性会比较差
  - qwen3 - 1.7b 这个能work最好
  - qwen3 - 0.6b 如果能work那就非常好，目前看来0.6B用来作misalignnment应该比较合适
  - tips:3090主机有一些跑出来的中间结果在result_cobelv1.1_0824中，可以拿那些belief来测试效果

  - 已有code，写好了generate，见LLM_cobel，建议看commit记录对比新增代码。

- 更新measurement_update，改成我们现在有template的初始化，所以在measurement update中可以只输入observation 和 rules 输出这一步观测的信息（格式化表示），再把这些信息通过正则提取更新进去。


TIPs:
- 3090主机的./bashrc 里面存放了小组的chatanywhere api（含deepseek）和我自己的阿里云api，可以用我的api来跑实验。 使用os.env指令读取即可。
- 注意在主机上跑代码改代码时，用git checkout shaokang切换到自己的分支上coding。

- capo baseline测试
  - 可以用主机跑目前显卡显存根本吃不满，可以同时跑4个tdw
  - 注意结果文件夹的命名result_{算法}{版本}_{日期monnth-day},我修改过了结果保存逻辑，现在同一个配置跑多次会自动在设置的输出文件夹下依次生成run_1 run_2等等。可以参考我的lauch.json:
  "args": [
                "--output_dir","results_cobelv1.1_0824",
                "--lm_id","deepseek-chat",
                "--experiment_name","try",
                "--run_id","",
                "--port","10004",
                "--agents","lm_agent_cobel","lm_agent_cobel",
                "--communication",
                "--prompt_template_path","LLM/prompt_com.csv",
                "--max_tokens","1024",
                "--data_prefix","dataset/dataset_test/",
                "--eval_episodes","0","1","4","9","10",
                "--screen_size","256",
                
            ], #我额外制定了保存第一人称图片的参数 即删去--not_save_img，注意保存图片
  - 注意不同tdw间的端口冲突




- subgoal应该包括好几个low-level actions，目前只让大模型自己选择subgoal有点把握不了这个程度
  - TODO： few-shot引导一下subgoal生成
- token消耗因为measurement update每次要输入输出完整的belief，成本有点高
  - TODO： 因为我们现在有template的初始化，所以在measurement update中可以只输入observation 和 rules 输出这一步观测的信息（格式化表示），再把这些信息通过正则提取更新进去
- 双系统目前还没有实现
  - TODO：快速推理有三块
    - measurement update
    - misalignment aware
    - intuitive planning
  其中最难用小模型做的有两部分
  - 一是把完整观测输入给模型，没有办法带入协作者视角去提取信息，往往导致把自己的视觉看到的（但是没有通过通信告诉协作者）的信息错误放到1st-beliefs中
  - 目前只把message放入了，忽略了 i saw agent这类情况对于一阶信念的作用

  - 二是intuitive planning需要根据subgoal选择一个最好的动作


- 此外，如果性能不好，考虑引入重规划，刚开始的subgoal不太稳定，还可能一直误导后面的intuitive planning。
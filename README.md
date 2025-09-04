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



- bed的位置好像不用知道 直接transport过去就行了

- TODO
  - 完成物体更新到belief里面
  - intuitive plan 传入state即可

======opposubplan======== explore current room and fill containers with objects here.
======mysubplan======== explore Office(3000) and fill containers with objects here
没有container喜欢加cotainer
- 为什么能抓别人手上的


- target_object_state(banana(Unknown)): location(Kitchen(5000))
    - target_object_state(bread(Unknown)): location(Kitchen(5000))
    - target_object_state(apple(Unknown)): location(Kitchen(5000))


update也不稳定 成本也高



differet content十分不稳定



实验表：
- coela
- roco 还没做
- capo
- embodied
- belief的更新不太好


0830记录
两版本实验：
- 输出belief差距
- 只输出0阶的

TODO：
- 第一阶段的规则生成需要改很多prompt 这个可以给禾姐做
- 观察一下 isaw还会不会有物体重复出现的问题（在手上+在屋子里）SOLVED
- 优化subgoal的设计 
  - 房间的探索情况应该被映射为物品可能情况 比如剩一个苹果 剩两个房间未被探索 所以应该去那两个房间探索
    - 这一步可以使用可能情况的列表来表示 prediction时，同时进行 比如得到apple3 in [livingroom1000, kitchen3000] 交给intuitive planning去做
    - 然后如果通过通信知道了在哪个房间，应该先explore and locate object location and then grasp it and transport 这个其实应该是intuitive来做 或者说subplan就定成这种
    - 然后如果是刚开始 subplan可以是抽象一点的 explore current room and grasp objects as many as possible
      - 解析为 explore current room -> 看到几个物品 -> go grasp -> go grasp -> 满了 -> SUBPLAN DONE

- 替换好像还有点问题
- init不太稳定 ongoing SOLVED 后续需要调整prompt减少示例
- V1prompt 没有对init进行增强
- send a message 重复次数太多


实验一些fancy的点：
- 设置严苛的通信条件 对不不同算法的性能 比如一帧50个字母
  - 分三个等级来做 一个500 一个100 一个50（极端情况） 但是没有量纲的概念
  - cwah可以换算成步数
- 主表是成功率对比+通信成本对比（两个维度：一是通信api次数 二是平均字数）
- 模型使用deepseekV3 + 小模型（未定）
- 消融实验
  - 消融符号表征 改成跟coela一样的自然语言文本描述
  - 消除预测更新 仅靠观测更新

Goal:
Transport 6 apples, 2 breads, 1 loaf_bread, 1 burger to the bed.
Room list:
<Office> (1000), <Livingroom> (2000), <Livingroom> (3000), <Office> (4000), <Kitchen> (5000), <Livingroom> (7000), <Bedroom> (8000). 

V1.5 新的prompt(only_zero+new subplan) + mes_th = 2 + action_th = 3 有completion更新（没debug）
  - check completion 更新
  - check difference
  - check subplan （这一版的log比较简洁）
  - ep 9 10 11 19 20 21
V1.4 旧promt（不同+old subplan) + mes_th =2 + action_th = 3 无completion更新（没de bug）

V1.5.1 (run2)
  - 加了推理 （intuitive）
  - prediction指定了是几个动作的组合
  
V1.5.2 
  - intuitive改成理由
  - 重写了logger 更清晰
  - 修改了action history 防止send message占据任务长度
  - debug visible_obj不更新
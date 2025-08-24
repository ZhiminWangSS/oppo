## TODO

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
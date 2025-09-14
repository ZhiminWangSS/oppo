
================================================
Belief Symbolic Language (BSL)

'''
    Syntax:
    ?belief = ?entity PREDICATE [?entity:?confidence]
    ?entity — a placeholder for any agent, object, or location in the environment (e.g., agentA, apple, room3)
    PREDICATE — a relational verb or state descriptor (e.g., IN, HOLD, SEE, BELIEVE)
    ?confidence — one of: certain, high, medium, low

    Zero-order belief:
    ?agent BELIEVE ?belief
    Example: agentA BELIEVE apple IN [kitchen:high]

    First-order belief:
    ?agentA BELIEVE ?agentB BELIEVE ?belief
Example: agentA BELIEVE agentB BELIEVE banana IN [pantry:medium]

    Rules for representing uncertainty:

    For mutually exclusive (conflicting) possibilities, use OR.
    Example:
    ?object IN [room1:high] OR IN [room2:low]
    — The object cannot be in both rooms simultaneously.
    For compatible (non-conflicting) possibilities, use AND.
    Example:
    agentA HOLD apple : certain AND banana : certain
    — The agent can hold multiple objects at once.
    Conflict resolution rule:
    If multiple mutually exclusive beliefs about the same entity-predicate pair exist, and one is upgraded to “certain”, all other conflicting possibilities for that predicate must be removed.
    Example:
    Initially: ball IN [roomA:high] OR IN [roomB:medium]
    After observation: ball IN [roomA:certain]
    Result: ball IN [roomA:certain] (IN [roomB:medium] is discarded)
    Note: The “?” prefix denotes a variable or placeholder to be instantiated with a concrete entity name during runtime or grounding. It is not part of the final instantiated belief statement.'''

==================================================



===================belief world construction=================

Chanllenge Description:
Belief Symbolic Language:

Domain Belief Knowledge:

Belief Format:
tdw:
?object IN ?room/?agent/?

==================================================

==================belief correction and update=====================
Instruction:
You are a precise, logic-driven belief tracking engine operating in a multi-agent environment with partial observability and dialogue. Your input consists of:
1.Visual Description: 
Describes what the self agent directly sees (e.g., objects, agents, locations).
This knowledge is private — only self initially knows it.
Dialogue History:
Each message has a sender and a receiver.
Only the sender and receiver gain knowledge from the message.
Other agents remain unaware unless they later observe or are told.

Your task:
From the input, extract and infer the following types of beliefs held by self, and update previous beliefs them exactly in the format specified below. Remember that confidence level of the beliefs extracted from input are all "certian", because these beleifs are directly observed. 

You must:
1.Distinguish between private and shared knowledge.
2.Correct previous estimated beliefs according to the input.
3.Maintain the beliefs in the format of Belief Symbolic Language.
4.DO NOT generate information not be mentioned both in Visual Description and Dialogue History
Visual Description:
Dialogue History:
Belief Symbolic Language:

Old Beliefs:

Answer:
updated zero order beliefs:
updated first order beliefs:

room state 跟progress一起输出
===========================================================


=======================first  predict with guides=============================
Task Description:I am $AGENT_NAME$. My teammate $OPPO_NAME$ and I want to transport as many target objects as possible to the bed with the help of containers within 3000 steps. I can hold two things at a time, and they can be objects or containers. I can grasp containers and put objects into them to hold more objects at a time. Note that a container can contain three objects, and will be lost once transported to the bed. The room can be explored none/part/all.

Instruction:
Please help me predict the likely locations of uncertain beliefs(beliefs without certain confidence) based solely on an agent’s first order beliefs.

Given the goal, your task is to first reason the likely state uncertain beliefs(beliefs without certain confidence) and assign a belief level(high/medium/low) to update previous beliefs. And then predict the possible next subplans the  $OPPO_NAME$ will take.

DO NOT predict for objects already known to be held or seen (certain state).

Reasoning Guidelines:
Beliefs reasoning:
Ignore beliefs with certain location.
Use the room exploration state to reason the possible locations:
e.g. self BELIEVE <livingroom>(1000) EXPLORED all - self BELIEVE <goal_objects>(id) IN [<livingroom>(1000)]:low, and part - medium and none - high.
You are allowed to assign multiple possible location for one goal objects, use "and" to represent them:
e.g. self BELIEVE <pen>(12123) IN [<livingroom>(1000)]:low and [<kitchen>(2000)]

Subplan reasoning:
The generated subplan must meet following requirements:
1.There are 5 allowed actions you can use to construct the subplan. The subplan should composed of 1-3 actions. 1) ‘go to’: move to a specified room. 2) ’explore’: explore a room for underlying target objects. 3) ‘go grasp’: go to grasp a specified target object. 4) ‘put’: Place an object into a specified container. 5) ’transport’: Transport holding objects or containers to the bed and drop them on the bed.
Here are some examples for you:
Go to <living>(room) and explore it.
Go grasp <banana>(12123) and put it into the <basket>(22123).
Go grasp <banana>(12123) and <apple>(32123) and transport them to the bed.

You need to reason three most possible subplans,use "subplan1: subplan2: subplan3:" to represent.

Important Rules:
No natural language explanation in the beliefs.
Represent objects,container and room strictly in the format <name>(id) like <livingroom>(1000) <wicker_basket>(5388017). If you don't goal obejects' id, use <1> <2> <3> to represent multiple same objects. 
Do not delete any beleifs in previous beliefs, you are allow to add a new belief.



Goals: $GOAL$
Old Beliefs:
===========================================================
Instruction:
You are a strategic inference engine specialized in predicting the likely locations of uncertain beliefs(beliefs without certain confidence) based solely on an agent’s first order beliefs.

Given the goal, your task is to first reason the likely state uncertain beliefs(beliefs without certain confidence, maybe high/low/medium/unknown) and assign a belief level(high/medium/low) to update previous beliefs. And then predict the possible next subplans the agent will take.

DO NOT predict for objects already known to be held or seen (certain state).

Reasoning Guidelines:
Beliefs reasoning:
Ignore beliefs with certain location.
Use the room exploration state to reason the possible locations:
e.g. self BELIEVE <livingroom>(1000) EXPLORED all - self BELIEVE <goal_objects>(id) IN [<livingroom>(1000)]:low, and part - medium and none - high.
You are allowed to assign multiple possible location for one goal objects, use "and" to represent them:
e.g. self BELIEVE <pen>(12123) IN [<livingroom>(1000)]:low and [<kitchen>(2000)]

Subplan reasoning:
The generated subplan must meet following requirements:
1.There are 5 allowed actions you can use to construct the subplan. The subplan should composed of 1-3 actions. 1) ‘go to’: move to a specified room. 2) ’explore’: explore a room for underlying target objects. 3) ‘go grasp’: go to grasp a specified target object. 4) ‘put’: Place an object into a specified container. 5) ’transport’: Transport holding objects or containers to the bed and drop them on the bed.
Here are some examples for you:
Go to <living>(room) and explore it.
Go grasp <banana>(12123) and put it into the <basket>(22123).
Go grasp <banana>(12123) and <apple>(32123) and transport them to the bed.

You need to reason three most possible subplans,use "subplan1: subplan2: subplan3:" to represent.

Important Rules:
No natural language explanation in the beliefs.
Represent objects,container and room strictly in the format <name>(id) like <livingroom>(1000) <wicker_basket>(5388017). If you don't goal obejects' id, use <1> <2> <3> to represent multiple same objects. 
Do not delete any beleifs in previous beliefs, you are allow to add a new belief.

Task Description:I am $AGENT_NAME$. My teammate $OPPO_NAME$ and I want to transport as many target objects as possible to the bed with the help of containers within 3000 steps. I can hold two things at a time, and they can be objects or containers. I can grasp containers and put objects into them to hold more objects at a time. Note that a container can contain three objects, and will be lost once transported to the bed. The room can be explored none/part/all.





Goals: $GOAL$
Old Beliefs:


====================first predict==========================
Task Description:I am $AGENT_NAME$. My teammate $OPPO_NAME$ and I want to transport as many target objects as possible to the bed with the help of containers within 3000 steps. I can hold two things at a time, and they can be objects or containers. I can grasp containers and put objects into them to hold more objects at a time. Note that a container can contain three objects, and will be lost once transported to the bed. The room can be explored none/part/all.

Instruction:
Please help me predict the likely states of uncertain beliefs(beliefs without certain confidence) based on my first order beliefs.

Given the goal, your task is to first reason the likely state uncertain beliefs(beliefs without certain confidence) and assign a belief level(high/medium/low) to update previous beliefs. And then predict the possible next subplans the  $OPPO_NAME$ will take.
DO NOT predict for objects already known to be held or seen (certain state).


The generated subplan must meet following requirements:
1.There are 5 allowed actions you can use to construct the subplan. The subplan should composed of 1-3 actions. 1) ‘go to’: move to a specified room. 2) ’explore’: explore a room for underlying target objects. 3) ‘go grasp’: go to grasp a specified target object. 4) ‘put’: Place an object into a specified container. 5) ’transport’: Transport holding objects or containers to the bed and drop them on the bed.
Here are some examples for you:
Go to <living>(room) and explore it.
Go grasp <banana>(12123) and put it into the <basket>(22123).
Go grasp <banana>(12123) and <apple>(32123) and transport them to the bed.

You need to reason three most possible subplans,use "subplan1: subplan2: subplan3:" to represent.

Important Rules:
No natural language explanation in the beliefs.
Represent objects,container and room strictly in the format <name>(id) like <livingroom>(1000) <wicker_basket>(5388017). If you don't know goal obejects' id, use <1> <2> <3> to represent multiple same objects. 
Do not delete any beliefs in previous beliefs, you are allow to add a new belief in the belief symbolic language format.

Belief Symbolic Format

Answer strictly in this format:
Updated Beliefs:
Subplans: subplan1: subplan2: subplan3:



==================miscoordination aware====================
Beliefs:
Miscoordination Rules:
- subplan
- object
- container
- room explore state



Belief Symbolic Language
?belief = ?entity PREDICATE [?entity]:?confidence
confidence = certain/high/medium/low
zero order beliefs = ?agent BELIEVE ?belief
first order beliefs = ?agentA BELIEVE ?agentB BELIEVE ?belief
Rules: 
If a entity have multiple conflict possible results, use OR to represent.
Such as an object maybe in multiple rooms:
?object IN [room1]:high OR IN [room2]:low
If a entity have multiple compatible possible results, use AND to connect.
Such as an agent can hold two objects at a time:
agentA HOLD apple:certian AND banana:certain
rules
belief type:

?object IN ?room/?agent/?container
?agent IN ?room
?agent HOLD ?object/?container

?container IN ?room/?agent

?bed IN ?room

rules:
use self BELIEVE ?belief_type to represent zero-order beliefs
use self BELIEVE ?agent BELIEVE ?belief_type to represent first-order beliefs, the "?agent" exclude self

Chanllenge Description:
Agents can hold two things at a time, and they can be objects or containers. I can grasp containers and put objects into them to hold more objects at a time. Note that a container can contain three objects, and will be lost once transported to the bed.
=================================measurement update============================
System:
You are a precise, logic-driven belief tracking engine operating in a multi-agent environment with partial observability and dialogue. Your input consists of:
1.Visual Description: 
Describes what the self agent directly sees (e.g., objects, agents, locations).
This knowledge is private — only self initially knows it.
Dialogue History:
Each message has a sender and a receiver.
Only the sender and receiver gain knowledge from the message.
Other agents remain unaware unless they later observe or are told.

Your task:
From the input, extract and infer the following types of beliefs held by self, and update previous beliefs them exactly in the format specified below. Remember that confidence level of the beliefs extracted from input are all "certian", because these beleifs are directly observed. 
You must:
1.Distinguish between private and shared knowledge.
2.Correct previous estimated beliefs according to the input.


OUTPUT FORMAT(Strictly Enforced)
All outputs must follow this schema exactly. Use brackets [ ] for confidence annotations. Do not add explanations in the output block.
zero-order-beliefs
self BELIEVE ?object IN [?room:confidence]
self BELIEVE ?object HOLDED BY [?agent:confidence]
self BELIEVE ?object INSIDE [?container:confidence] HOLDED BY [?agent:confidence]

self BELIEVE ?agent IN [?room:confidence]
self BELIEVE ?agent HOLD [?object:confidence]
self BELIEVE ?agent PLAN [?plan:confidence]

self BELIEVE ?room EXPLORED [?state:confidence]

self BELIEVE ?bed IN [?room:confidence]
first-order-beliefs
self BELIEVE ?agent BELIEVE ?object IN [?room:confidence]
self BELIEVE ?agent BELIEVE ?object HOLDED BY [?agent:confidence]
self BELIEVE ?agent BELIEVE ?object INSIDE [?container:confidence] HOLDED BY [?agent:confidence]

self BELIEVE ?agent BELIEVE ?object IN [?room:confidence]
self BELIEVE ?agent BELIEVE ?object HOLDED BY [?agent:confidence]
self BELIEVE ?agent BELIEVE INSIDE [?container:confidence] HOLDED BY [?agent:confidence]

self BELIEVE ?agent BELIEVE ?agent IN [?room:confidence]
self BELIEVE ?agent BELIEVE ?agent HOLD [?object:confidence]
self BELIEVE ?agent BELIEVE ?agent PLAN [?plan:confidence]

self BELIEVE ?agent BELIEVE ?room EXPLORED [?state:confidence]

self BELIEVE ?agent BELIEVE ?bed IN [?room:confidence]

Update Guidelines:
Assume I'm Alice(self).
If self sees <pen>(12123) in <livingroom>(1000):
old belief: ?self BELIEVE <pen>(12123) IN [<livingroom>(1000)]:high and [<kitchen>(2000)]:medium
new belief: ?self BELIEVE <pen>(12123) IN [<livingroom>(1000):certain]
If self holding a <pen>(12123) and a <plate>(32123) with a <banana>(22123) in it:
old belief: ?self BELIEVE self HOLD [pen:certain]
new belief: ?self BELIEVE self HOLD [<pen>(12123); <banana>(22123) IN <plate>(32123)]:certain
If Bob tells Alice: "I've explored all of the <bedroom>(3000) and found the <pen>(12123)."
self BELIEVE <pen>(12123) IN [<bedroom>(3000)]:certain
self BELIEVE <bedroom>(3000) EXPLORED [all]:certain
self BELIEVE Bob BELEVE <pen>(12123) IN [<bedroom>(3000)]:certain
self BELIEVE Bob BELIEVE <bedroom>(3000) EXPLORED [all]:certain

Important Rules:
No hallucination. Only output beliefs that are supported.
No natural language explanation in the output block.
If a belief is assigned certain, remove all mutually exclusive beliefs.
Represent objects,container and room strictly in the format <name>(id) like <livingroom>(1000) <wicker_basket>(5388017).
If a belief is new, just add it into the beliefs in proper orders.
If beliefs conflict, replace them with new beliefs.
If a belief contains multiple entities, use ; to divide them
Please answer strcitly in this format:
zero order beliefs:
first order beleifs:


User:
I'm Alice. My teammate Bob and I want to transport as many target objects as
possible to the bed with the help of containers. Please help me update my beliefs according to my visual description and dialogue history.
Previous Beliefs: $BELIEFS$
Visual Description: $OBS$
Dialogue History：$DIALOGUE_HISTROY$

zero order beliefs:
first order beleifs:



- belief reasoning
- miscoordination reasoning
=============== predition ================

System:
You are a strategic inference engine specialized in predicting the likely locations of uncertain beliefs(beliefs without certain confidence) based solely on an agent’s current beliefs and belief reasoning rules.

Given the goal, your task is to first reason the likely state uncertain beliefs(beliefs without certain confidence, maybe high/low/medium/unknown)  and assign a belief level(high/medium/low) to update previous beliefs. And then predict the possible next subplans the agent will take.
Do not predict for objects already known to be held or seen (certain state).

Reasoning Guidelines:
Beliefs reasoning:
Ignore beliefs with certain location.
Use the room exploration state to reason the possible locations:
e.g. self BELIEVE <livingroom>(1000) EXPLORED all - self BELIEVE <goal_objects>(id) IN [<livingroom>(1000)]:low, and part - medium and none - high.
You are allowed to assign multiple possible location for one goal objects, use "and" to represent them:
e.g. self BELIEVE <pen>(12123) IN [<livingroom>(1000)]:low and [<kitchen>(2000)]

Subplan reasoning:
The generated subplan must meet following requirements:
1.There are 5 allowed actions you can use to construct the subplan. The subplan should composed of 1-3 actions. 1) ‘go to’: move to a specified room. 2) ’explore’: explore a room for underlying target objects. 3) ‘go grasp’: go to grasp a specified target object. 4) ‘put’: Place an object into a specified container. 5) ’transport’: Transport holding objects or containers to the bed and drop them on the bed.
Here is an example for you:
You need to reason three most possible subplans,use "subplan1: subplan2: subplan3:" to represent.

Important Rules:
No natural language explanation in the beliefs.
Represent objects,container and room strictly in the format <name>(id) like <livingroom>(1000) <wicker_basket>(5388017). If you don't goal obejects' id, use <1> <2> <3> to represent multiple same objects. 
Do not delete any beleifs in previous beliefs, you are allow to add a new belief.

Answer strictly in this format:
Updated beliefs:

Subplan List:
subplan1:
subplan2:
subplan3:

user:
I'm Alice. My teammate Bob and I want to transport as many target objects as
possible to the bed with the help of containers. Please help me predict the possible locations of unseen goal objects. And make several subplans for me.

Previous Beliefs: $BELIEFS$
Goals: $GOAL$

Updated beliefs:

Subplan List:
subplan1:
subplan2:
subplan3:

=============== miscoordination aware ================
You are a collaborative reasoning and conflict detection engine.


最终方法论
==== belief correct ====

==== belief predict ====
explore - obj
history subplan - loc holding
oppo subplan list - oppo location oppo holding
==== miscoordination aware ===
检测是否会出现潜在的冗余规划和冲突规划
- 冲突探索 - subplan
- 重复探索 - exploration state
- 容器利用不好 - container
- 找到所有东西，立刻全力搬运 - objects（不用传物品完成情况 直接读（看看coela是不是这样的） 是的 不用传完成情况）














============ parse plan =================
I am \$AGENT\_NAME\$. My teammate \$OPP\_NAME\$ and I want to transport as many
target objects as possible to the bed with the help of containers within 3000 steps. I can hold
two things at a time, and they can be objects or containers. I can grasp containers and put
objects into them to hold more objects at a time.
Assume that you are an expert decision maker. Given our shared goal, action plan, my
progress, and previous actions, please help me choose the best available action to achieve the
goal as soon as possible. Note that a container can contain three objects, and will be lost once
transported to the bed. I can only put objects into the container I hold after grasping it. All
objects are denoted as <name> (id), such as <table> (712). Actions take several steps to
finish. It may be costly to go to another room or transport to the bed, use these actions
sparingly.
Goal: \$GOAL\$
Meta plan: \$META\_PLAN\$
Dialogue history: \$DIALOGUE\_HISTORY\$
Progress: \$PROGRESS\$
Previous action: \$PREVIOUS\_ACTIONS\$
Action list: \$ACTION\_LIST\$



====== 第一步 更新 Tom推理 ====== #measurement update
System：
Beleif Format
Rules
Instructions

User：
OBS
MES
OLD_Knowledge + OLD Beliefs

OUTPUT New Knowledge + OLD Beliefs

Belief在这套机制里面起到的作用 检测冲突
如果接受到了他的subplan
- 冲突建议
- 回复冲突
    - 认同 不回复 直接改 因为他直接按照原来的做 所以冲突已经解决了
    - 不认同 
        - 不中断 因为我已经做了你的了或者我做不了了 比如已经到了房间/已经拿了东西 
        回复我不行 我准备做xx 请调整你的计划


Alice: explore current room
Bob: explore current room
====== 第二步 （IF SUBPLAN = NONE）正向推理 belief ====== #predict
GOALS
ROOMS
New Knowledge + OLD Beleifs


OUTPUT
New Knowledge + New Beliefs = incompleted objs + New subplans

two-step Belief + subplan 两人同时进行

====== 第三步 检测潜在冲突或者冗余 ======
INPUT
两人的beliefs
My subplan：只有一个
Oppo subplan list

冲突计划：探索同一间屋子；抓取同一个物体
冗余行动：
没有利用好容器，（前提手上至少一个空的，才能建议拿容器，并且没完成的大于3个）；
探索已经探索过的房间。


====== 第四步 如果冲突 通信解决冲突 如果不冲突 执行动作 ====== belief plan
My subplan
Oppo subplan list
通信内容：1.冲突原因：即物品状态信息，e.g.1. 因为我已经探索了a房间/因为我发现了新容器且你手上有空儿/因为你也没有探索过a房间 2. 所以你会重复探索a/不用容器导致冗余/你也去探索a房间了
发出去信息 = 状态+计划建议 如何处理？
- 讨论 进入讨论状态
    - 接受者发现冲突了 -> 调整计划
    - 接受者了解了这个冲突（因为还没规划到这步/预测不太准） -> 解决计划
- 不讨论
    - 接受者发现冲突 -> 该计划 按他的来
    - 接受者未发现冲突 -> 继续做，告诉他我的计划
plan 通信：提出一套不冲突的计划，然后讨论？
obj 通信：
progress

possible belief confict and redundant trigger
AKA miscoordination trigger

就是根据双方belief的差距然后推断可能发生的冲突或冗余。
最开始，探索完当前房间后，可能造成计划的冲突
e.g. Alice explore another room Bob explore another room
发生潜在冲突 -》 触发通信讨论？ Alice：我觉得计划可能冲突，因为。。。。。 我建议我去a 你去b Bob:ok

中间的时候：探索完房间了，可能时间不太一样，Alice比较快，触发讨论。我探索完这个了，你好久没有更新东西给我了，下一步我准备去xxx，你可以去xxx； Bob，实际上，我还没有探索完，


潜在冗余：Alice拿到容器后，又发现了容器，知道Bob现在手上没容器，而且还在找
潜在冗余：Alice探索过了livingroom，建议Bob不要来探索了，因为知道bob还不知道livingroom的情况。

======== predict subplan =========
Instruction: I'm Alice, Bob and I ....
The generated subplan must meet following requirements:
1.There are 5 allowed actions you can use to construct the subplan. The subplan should composed of 1-3 actions. 1) ‘go to’: move to a specified room. 2) ’explore’: explore a room for underlying target objects. 3) ‘go grasp’: go to grasp a specified target object. 4) ‘put’: Place an object into a specified container. 5) ’transport’: Transport holding objects or containers to the bed and drop them on the bed.
Here is an example for you:

2.The subplan should be concise, brief, and reliable.
Goals:
Rooms:
predict beliefs: None
predict oppo_subplan: 


Certain Beliefs: #通过观测和通信
self BELIEVE <apple>(12123) IN [<livingroom>(1000):certain]

self BELIEVE <livingroom>(1000) EXPLORED [all:certain]
#不确定的不放

Beliefs: #通过预测
self BELIEVE <apple>(2) IN [<Kitchen>(2000):low;<Bedroom>:high]


(IF NOT receive ?agent subplans)
Predict First-order Beliefs:
Goals:$GOALS$
Knowledges:$First-order knowledges$
Knowledges = Observed Objs,Containers,Agents and Room state

Predict Zero-order beliefs:
Goals:$GOALS$
Knowledges:$Zero-order knowledges$



- objs
- my_subplan



Output:
zero-order-beliefs:
first-order-beliefs:





rules: 
HOLD at most two items including ?object and ?container
?plan should be a short description about what ?agent plans to do.
Here are examples about plan:
go to livingroom(1000) and explore it, grasp the object in livingroom(1000) as many as possible.
go to kitchen(2000) and explore it to locate apple(12123), and grasp the apple(12123).




BELIEVE ?agent BELIEVE CONFIDENCE certain/high/low

cwah一开始知道有多少容器吗



Motivation
现在的LLM collaboration缺少自主动协调一致性的能力 因此导致依赖大量通信去一致协作——即使他们当下的计划是一致的（capo） 另外一个是coela 或者在协作不一致的时候没有主动协调的能力。
但人类，会主动感知和思考协作的状态 根据历史中体现的他们的已有知识来推测接下来的行为，从而判断彼此的规划是否一致和最优。受此启发，我们提出一个新的框架——cobel 利用模型推理能力构建协作式的信念世界，来驱动智能体拥有主动协作的能力。


解决的问题：LLM没有主动协调智能体协作的能力。 结果导致不一致或者冗余的规划

相比proagent 他无法在





Here is the prediction rules:
1. If a room's exploration state is none, the objects in this room's confidence level is high;
2. If a room's exploration state is part, the objects in this room's confidence level is medium;
3. If a room's exploration state is all, the objects in this room's confidence level is low;
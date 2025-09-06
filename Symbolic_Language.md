基本语法
?object IN ?room
?object HOLDED BY ?agent
?object HOLDED BY ?agent

?agent HOLD ?item
?agent IN ?room
?agent PLAN ?plan

?container IN ?room
?container HOLDED BY ?agent

Belief语法：
zero-order-beliefs
BELIEVE ?object IN ?room_list
BELIEVE ?object HOLDED BY ?agent
BELIEVE ?object INSIDE ?container HOLDED BY ?agent



first-order-beliefs
BELIEVE ?agent BELIEVE ?object IN ?room
BELIEVE ?agent BELIEVE ?object HOLDED BY ?agent
BELIEVE ?agent BELIEVE ?object INSIDE ?container HOLDED BY ?agent 

BELIEVE ?agent BELIEVE ?object CONFIDENCE certain/high/low
BELIEVE ?agent BELIEVE ?container CONFIDENCE certain/high/low




rules: 
HOLD at most two items including ?object and ?container
?plan should be a short description about what ?agent plans to do.
Here are examples about plan:
go to livingroom(1000) and explore it, grasp the object in livingroom(1000) as many as possible.
go to kitchen(2000) and explore it to locate apple(12123), and grasp the apple(12123).




BELIEVE ?agent BELIEVE CONFIDENCE certain/high/low

cwah一开始知道有多少容器吗
### rules

```
zero-order beliefs:  



- object_state(?object)
	- location (?room OR ?container OR ?hand)
  	- complete(known OR unknown)

- container_state(?container)
  - location(?room OR ?hand)
  - objects_inside(?object,?object,?object) #at most three

- agent_state(?agent)
  - location(?room)
  - objects_in_hand(?object,?object) #at most two
  - subgoal(a short description about what ?agent plans to do)
  #推理bob的subgoal 如果是none 阈值很大的话 直接不管 执行下一步 或者 bob 更新了subgoal给我说
            取值：[推理不出来，推理得到当前的subgoal（需要不需要更新）]
            推理的输入：[bob的（+历史）subgoal 和 alice视角的bob状态 以及任务完成情况] = 0阶信念

- room(?room)
  - exploration_state(None OR Part OR All)

- bed
  - location(?room)

first-order beliefs (?agent): #alice belief about bob's internal states
- other_know_object(?agent, ?object)
  - location(known OR unknown)
  - completion(known OR unknown)

- other_know_container(?agent, ?container)
  - location(known OR unknown)
  - objects_inside(?object,?object,?object)

- other_know_agent(?agent)
  - location(known OR unknown)
  - subgoal(known OR unknown)
  - objects_on_hand(?object OR ?container,?object OR ?container) #at most two entities 具体什么物体 还是 数量 物体比较好一点

- other_know_room(?room,?agent)
  - exploration_state(known OR unknown)

- other_know_bed_location(?agent)
  - location(known OR unknown)
```











```

rules:

in_hand(targetobject, room OR hand(agent) OR container)
e.g. in_hand(pen_1, hand(alice), )


- target_object():
	- location [] 

zero-order beliefs:
#实体是 pen_1 

- target_object(pen_1) # 实体类
	- location[in_hand(Alice)] #in_hand 关系/谓词
- target_object(pen_2)
	- location[in_hand]
- container(container1)
  - location[in_hand]                             
  - capacity[]      

- agent(Alice)
  - location[in living_room]
  - carried_items[count: 2]                               
  - subgoal ["go to explore kitchen and transport all things there" at 999 frame]

- agent(Bob)
  - location[in_kitchen]
  - carried_items[count: 0]                               
  - subgoal [unknown]

- room(livingroom)
  - exploration_state[all]
- room(bedroom)
  - exploration_state[part]
- room(kitchen)
  - exploration_state[none]

- bed_location
  - location[in_room] → in bedroom

first order beliefs about bob:
- know_container(Bob)
  - location[yes]                                    
  - num_objects_inside[2]                                  

- know_agent(Bob)
  - location[yes]                              
  - subgoal[yes]                   
  - objects_on_hand[list: []]                            

- know_room(Bob, livingroom) #me is alice
  - exploration_state[yes]                         

- know_bed_location(Bob)
  - location[yes] → in bedroom                           
```


20250817
``` 
zero-order_belief:  
- target_object_state(?object)
  - location(?room or ?hand or ?container)
  - completion(?complete or ?incomplete)

- container_state(?container)
  - location(?room or ?hand)
  - contents[?object, ?object, ?object]  # at most three objects; no containers allowed

- agent_state(?agent)
  - location(?room)
  - subgoal(a short description about what ?agent plans to do)
  - object_in_hand[?item, ?item]  # at most two items; ?item is object or container

- room_state(?room)
  - exploration_state(?None or ?Part or ?All)

- bed_location()
  - location(?room)

first-order_belief(?agent)  # belief of ?agent's internal state about others' knowledge
- other_know_target_object(?object)
  - location(?know or ?unknow)
  - completion(?know or ?unknow)

- other_know_container(?container)
  - location(?know or ?unknow)
  - contents[?object, ?object, ?object]  # list of objects; reflects belief about what other knows is inside

- other_know_agent(?other_agent)
  - location(?know or ?unknow)
  - subgoal(?know or ?unknow)
  - object_in_hand[?item, ?item]  # list of items; ?item is object or container, reflecting belief about other's held items

- other_know_room_state(?room)
  - exploration_state(?know or ?unknow)

- other_know_bed_location()
  - location(?know or ?unknow)
```

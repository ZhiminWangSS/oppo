All observation needed:include visual observation and message;

visual observation by shaokang TODO

measurement update prompt:

- zero-order completed
- first-order completed

### a template of observation

```
Bob 's Observation：
Visual observation: I'm holding a container <plate> (7818192) with nothing in it, and a target object <apple> (3205029). I see <banana> (10210312). I'm in the <Livingroom> (8000). I see Alice holding nothing. 


Message:
Alice: "I will go to kitchen(10001)."
Bob: "Alice, I found 2 apples and 1 burger in Livingroom (8000). I’ll grab the plate first to hold more items." 

```

zero-order

```
Bob 's Visual observation: I'm holding a container <plate> (7818192) and a target object <apple> (3205029). The container <plate> (7818192) contains nothing. I see <banana> (10210312) in the in the <Livingroom> (8000). I'm in the <Livingroom> (8000). I see Alice holding nothing. 


Message:
Alice: "I will go to kitchen(10001)."
Bob: "Alice, I am holding <apple> (3205029) and <plate> (7818192). I am in <Livingroom>(8000). And i found <banana> (10210312) in livingroom,too. I’ll go to <bedroom>(2000)." 


You are Bob. You are cooperating with Alice to transport objects to the bed. Based on your visual observation and message, what information do you know now? Use the rules below to construct your known information.

Rules:
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

Just answer in the rules form, without any analysis and reasoning progress. Remember the full name include the id such as (12312). If you don't no any information, just put "Unknwon" in the ().
```

message construction - first-order

``` 
Message:
Alice: "I will go to kitchen(10001)."
Bob: "Alice, I am holding <apple>(1131231) and burger<123123123>. I am in <Livingroom>(8000). And i found pen in livingroom,too. I’ll go to <bedroom>(2000)." 

Bob and Alice are two agents cooperate to finish tasks. Based on the message,  what bob believe about alice's knowledge after the message received by Alice? Use the rules below to construct what Bob believe Alice know including bob:

Rules:
- know_target_object(?object)
  - location(?room or ?hand or ?container)
  - completion(?completed or ?incompleted)

- know_container(?container) 
  - location(?room or ?hand)
  - contents[?object, ?object, ?object]  # list of objects; reflects belief about what other knows is inside

- know_agent(?agent)
  - location(?room)
  - subgoal(a short description about what ?agent plan to do next)
  - object_in_hand[?item, ?item]  # list of items; ?item is object or container, reflecting belief about other's held items
  
- know_room_state(?room)
  - exploration_state(?all or ?part or ?none)

- know_bed_location()
  - location(?room)

just output in the rules form, without any analysis and reasoning progress. Remember the full name include the id such as (12312). If you don't no any information, just put "Unknwon" in the ().
```



```
Old beliefs:
zero-order_belief: 
- target_object_state(bread1)
  - location(Unkown)
  - completion(Unkown)

- target_object_state(bread2)
  - location(Unkown)
  - completion(Unkown)
  
- target_object_state(apple1)
  - location(Unkown)
  - completion(Unkown)

- target_object_state(apple2)
  - location(Unkown)
  - completion(Unkown)

- target_object_state(banana1)
  - location(Unkown)
  - completion(Unkown)

- target_object_state(banana2)
  - location(Unkown)
  - completion(Unkown)
  
- target_object_state(banana3)
  - location(Unkown)
  - completion(Unkown)
  
- target_object_state(burger1)
  - location(Unkown)
  - completion(Unkown)
  
- target_object_state(burger2)
  - location(Unkown)
  - completion(Unkown)
  
- target_object_state(orange1)
  - location(Unkown)
  - completion(Unkown)
Belief rules：
```







OLD

```
You are **Bob**,  and you are currently completing the task of trasporting target objects to bed in multiple rooms with **Alice**. I will provide you with a set of rules for establishing beliefs and a description of your current observations  that imply some belief information. You need to construct the beliefs in two-stage.
First, construct the zero-order beliefs which represents your own mental knowledge over the environments. You may need to find avaluable information not only from the observation, and may need to analyze the message to extract some information that Alice sents to you. Here are examples show how to construct zero-order beliefs.
1. From message,  Alice said she found <apple(10111)> in <livingroom>(122323), so Bob construct: 
target_object(<apple> (10111))
    location(<livingroom>(122323))
    completion(incomplete)
2. From visual observation, Bob last saw Alice in <livingroom>(122323), so Bob construct:
agent_state(Alice)
location(Livingroom (122323))
subgoal(Unkown)
object_in_hand[]
Second, construct the first-order beliefs which represents your estimates about **Alice**'s mental knowledge. Remember that Bob and Alice are two independent individual,  they exchange information their own mental knowledge through dialogue or see their state in the observation. Here are examples show how to construct first-order beliefs.

1. From dialogue history,  Alice said she found <apple(10111)> in <livingroom>(122323), so Bob construct: 
other_know_target_object(<apple> (10111))
    location(<livingroom>(122323))
    completion(incomplete)

Do not output the process of analysis or reasoning, just output the final belief content. If goal object contains multiple duplicate objects, identify them by the suffix 1, 2, 3 before finding them and knowing the id like <apple> (3205029). The id is 3205029. But replace the suffix with id when you find it and know the id.

Belief rules:
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
- know_target_object(?object)
  - location(?know or ?unknow)
  - completion(?know or ?unknow)

- know_container(?container)
  - location(?know or ?unknow)
  - contents[?object, ?object, ?object]  # list of objects; reflects belief about what other knows is inside

- know_agent(?agent)
  - location(?know or ?unknow)
  - subgoal(?know or ?unknow)
  - object_in_hand[?item, ?item]  # list of items; ?item is object or container, reflecting belief about other's held items
  
- know_room_state(?room)
  - exploration_state(?know or ?unknow)

- know_bed_location()
  - location(?know or ?unknow)
  
Bob 's Observation：
Visual observation: I'm holding a container <plate> (7818192) with nothing in it, and a target object <apple> (3205029). I see <banana> (10210312). I'm in the <Livingroom> (8000). I see Alice holding nothing. 
Message:
Alice: "I will go to kitchen(10001)."
Bob: "Alice, I found 2 apples and 1 burger in Livingroom (8000). I’ll grab the plate first to hold more items." 
```


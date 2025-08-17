tom reasoning - infer collaboraters' subgoal

```
- know_agent(Alice)
  - location(kitchen(10001))
  - subgoal(go to kitchen(10001))
  - object_in_hand[Unknown, Unknown]

- know_room_state(kitchen(10001))
  - exploration_state(Unknown)

- know_room_state(Livingroom(8000))
  - exploration_state(Unknown)

- know_room_state(bedroom(2000))
  - exploration_state(Unknown)

- know_target_object(apple(1131231))
  - location(hand)
  
- know_target_object(burger(123123123))
  - location(hand)

- know_target_object(pen)
  - location(Livingroom(8000))

- know_agent(Bob)
  - location(Livingroom(8000))
  - subgoal(go to bedroom(2000))
  - object_in_hand[apple(1131231), burger(123123123)]

You are Alice, you are cooperating with bob to transport some objects to bed. The goal objects include a pen a apple a burger and a banana.I will give you some knowledge organized in structured format. Please reason over these information and infer what alice will do next(subgoal).First give your reasons and answer me  with the subgoal only.

Reasons：
subgoal：
```

deliberate planning infer my subgoal based on collaborators' state and my 0-zero beliefs

```
You are Bob, you are cooperating with Alice transport some objects to bed. The goal objects include a pen，a apple，a burger and a banana. I will give you some knowledge about Alice state and subgoal, and environments' information. Please reason over these information and decide what to do next(subgoal).First give your reasons and answer me  with the subgoal only.


Alice:
- know_agent(Alice)
  - location(kitchen(10001))
  - subgoal(go to kitchen(10001))
  - object_in_hand[Unknown, Unknown]


Reasons：
subgoal：
```


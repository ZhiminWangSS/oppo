```
zero-order_belief from Alice:  
- target_object_state(pen_1)
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


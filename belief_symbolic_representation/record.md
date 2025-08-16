### Qwen3-235-a32b
``` 
构建零阶belief...
zero-order beliefs:  
- target_object
  - location[in_room, in_container, in_hand]
  - completion[complete, incomplete]

- container
  - location[in_room, in_hand]
  - capacity[max_3_objects]
  - contents[set_of_target_objects] (0 to 3)
  - is_carried[true, false]

- agent
  - location[in_room]
  - carried_items[count: 0 to 2] (can be target_objects or containers)
  - subgoal

- room
  - exploration_state[None, Part, All]

- bed_location
  - location[unknown → in_room] (initially unknown, to be discovered)
  - is_destination[true]

**Reasoning on attributions and restraints:**
- *target_object*: Its location can be in a room, inside a container, or in an agent’s hand. Completion status reflects whether it has been successfully transported to the (initially unknown) bed location.
- *container*: Has a maximum capacity of 3 objects. It can be located in a room or carried by an agent. When carried, it contributes to the agent’s item limit (counts as one item). The contents are constrained by capacity.
- *agent*: Can carry up to two items at a time—each item being either a target object or a container. Location is room-based. Subgoal reflects the agent’s autonomously planned intermediate objective toward the shared goal.
- *room*: Exploration state tracks how much of the room’s contents have been discovered by the agents, affecting information availability and planning.
- *bed_location*: This is a special location that serves as the goal destination for completing tasks. It is initially unknown, implying that discovery through exploration is a necessary precondition for task completion. Once identified, it anchors the completion condition for transporting objects.
检查并精炼belief...
zero-order beliefs:  
- target_object
  - location[in_room, in_container, in_hand]
  - completion[complete, incomplete]

- container
  - location[in_room, in_hand]
  - capacity[max_3_objects]

- agent
  - location[in_room]
  - carried_items[count: 0 to 2]
  - subgoal

- room
  - exploration_state[None, Part, All]

- bed_location
  - location[in_room]
构建一阶belief...
zero-order beliefs:  
- target_object
  - location[in_room, in_container, in_hand]
  - completion[complete, incomplete]

- container
  - location[in_room, in_hand]
  - capacity[max_3_objects]

- agent
  - location[in_room]
  - carried_items[count: 0 to 2]
  - subgoal

- room
  - exploration_state[None, Part, All]

- bed_location
  - location[in_room]

first-order beliefs:
- know_target_object
  - location[yes, no]
  - completion[yes, no]

- know_container
  - location[yes, no]
  - capacity[yes, no]
  - num_objects_inside[0, 1, 2, 3]

- know_agent
  - location[yes, no]
  - carried_items[yes, no]
  - subgoal[yes, no]
  - objects_on_hand[list]

- know_room
  - exploration_state[yes, no]

- know_bed_location
  - location[yes, no]
最终检查belief...
zero-order beliefs:  
- target_object
  - location[in_room, in_container, in_hand]
  - completion[complete, incomplete]

- container
  - location[in_room, in_hand]
  - capacity[max_3_objects]

- agent
  - location[in_room]
  - carried_items[count: 0 to 2]
  - subgoal

- room
  - exploration_state[None, Part, All]

- bed_location
  - location[in_room]

first-order beliefs:
- know_target_object
  - location[yes, no]
  - completion[yes, no]

- know_container
  - location[yes, no]
  - num_objects_inside[0, 1, 2, 3]

- know_agent
  - location[yes, no]
  - subgoal[yes, no]
  - objects_on_hand[list]

- know_room
  - exploration_state[yes, no]

- know_bed_location
  - location[yes, no]
``` 

### Qwen3-30b-a3b-thinking
``` 
构建零阶belief...
zero-order beliefs:
- target_object
  - location[in_room, in_hand]
  - completion[complete, incomplete]
- container
  - location[in_room, in_hand]
  - object_count[0,1,2,3]
- agent
  - location[in_room]
  - subgoal
检查并精炼belief...
zero-order beliefs:
- target_object
  - location[in_room, in_hand]
  - completion[complete, incomplete]
- container
  - location[in_room, in_hand]
  - capacity[max three]
- agent
  - location[in_room]
  - subgoal
  - capacity[max two]
构建一阶belief
first-order beliefs:
- know_target_object
  - location[yes, no]
  - completion[yes, no]
- know_container
  - location[yes, no]
  - capacity[yes, no]
- know_agent
  - location[yes, no]
  - subgoal[yes, no]
  - capacity[yes, no]
最终检查belief...
first-order beliefs:
- know_target_object
  - location[yes, no]
  - completion[yes, no]
- know_container
  - location[yes, no]
  - count[number]
- know_agent
  - location[yes, no]
  - subgoal[yes, no]
  - hand[list]
``` 
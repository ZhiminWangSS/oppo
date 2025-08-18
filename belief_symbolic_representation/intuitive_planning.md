```
I’m Alice. My friend Bob and I want to transport as many target objects as possible to the bed with the help of containers.I can hold two things at a time, and they can be objects or containers. I can grasp containers and put objects into them to hold more objects at a time. Given my state, my previous actions, please help me choose the best available action according to my subgoal and previous actions. If I have done all actions need in the subgoal, just answer SUBGOAL DONE.

My state:
- agent_state(Alice)  
  - location(<Livingroom> (8000))  
  - subgoal(go and explore livingroom)  
  - object_in_hand[] 

Previous actions: go to <livingroom>(4000) at step 3; explore <livingroom>(4000) at step 99
Availuable actions: (You can only choose the action in the list, just answer without any analysis)
A. go to <Livingroom> (1000) 
B. go to <Office> (3000) 
C. explore <livingroom>(4000)
D. SUBGOAL DONE.
Answer:

```


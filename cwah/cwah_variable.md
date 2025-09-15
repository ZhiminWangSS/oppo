progress 入口:
current_room
{'id': 198, 'category': 'Rooms', 'class_name': 'livingroom', 'prefab_name': 'PRE_ROO_Livingroom_02', 'obj_transform': {'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [1.25, 1.25, 2.5], 'size': [8.0, 3.0, 8.0]}, 'properties': [], 'states': []}

grabbed_object [] = [self.id2node[x] for x in self.grabbed_objects]
估计里面是字典


unchecked_container: 
{'livingroom': [{'id': 140, 'category': 'Furniture', 'class_name': 'kitchencabinet', 'prefab_name': 'kitchen_cabinet', 'obj_transform': {'position': [-2.9199996, 2.05, 1.809576], 'rotation': [0.0, -1.0, 0.0, -8.940697e-08], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [-2.75357628, 2.05115032, 1.53582048], 'size': [0.4021535, 0.6996989, 0.567739248]}, 'properties': ['SURFACES', 'CAN_OPEN', 'CONTAINERS'], 'states': ['CLOSED']},], 'kitchen': None, 'bedroom': None, 'bathroom': None}

ungrabbed_objects:
{'livingroom': [], 'kitchen': None, 'bedroom': None, 'bathroom': None} = obj_per_room


goal_location_room = #应该是不包括所在的中心房间 应该是目标物体所在的房间？
['kitchen', 'bedroom', 'bathroom'] OR 'kitchen'


self.goal_location = <dishwasher> (159)

satisfied = [] 里面是字典

opponent_grabbed_objects = []


opponent_last_room = None

room_explored = None



==============agent


=================goal
{'inside_cutleryfork_<dishwasher> (159)': [3, True, 2], 'inside_plate_<dishwasher> (159)': [1, True, 2]}





message_generator_outputs:
"Hi Bob, I'm going to explore the kitchen next for cutleryforks and a plate. Let me know if you find anything or need help."
base_prompt:
I'm Alice. I'm in a hurry to finish the housework with my friend Bob together. Given our shared goal, dialogue history, and my progress and previous actions, please help me choose the best available action to achieve the goal as soon as possible. Note that I can hold two objects at a time and there are no costs for holding objects. All objects are denoted as <name> (id), such as <table> (712).
Goal: Find and put 3 cutleryforks, 1 plate into the <dishwasher> (159).
Progress: I'm holding nothing. I'm in the livingroom, where I found nothing. I don't know where Bob is. The kitchen is unexplored. The bedroom is unexplored. The bathroom is unexplored. 
Dialogue history:
Alice: "Hi, I'll let you know if I find any goal objects and finish any subgoals, and ask for your help when necessary."
Bob: "Thanks! I'll let you know if I find any goal objects and finish any subgoals, and ask for your help when necessary."

Previous actions: [goexplore] <livingroom> (198)
Available actions:
A. [send_message] <"Hi Bob, I'm going to explore the kitchen next for cutleryforks and a plate. Let me know if you find anything or need help.">
B. [goexplore] <kitchen> (56)
C. [goexplore] <bedroom> (294)
D. [goexplore] <bathroom> (11)




=========================unsatisfied=======================
{'inside_cutleryfork_<dishwasher> (159)': 2, 'inside_plate_<dishwasher> (159)': 1}

==================satisfied==========================
[{'id': 385, 'category': 'PRE_PRO_Fork_01', 'class_name': 'cutleryfork', 'prefab_name': 'PRE_PRO_Fork_01', 'obj_transform': {'position': [-8.744065, 0.857233047, -0.7356067], 'rotation': [0.0, 0.4539912, 0.0, 0.8910062], 'scale': [0.99999994, 0.9999996, 0.99999994]}, 'bounding_box': {'center': [-8.719957, 0.859946668, -0.7687891], 'size': [0.174502179, 0.01939815, 0.226643085]}, 'properties': ['GRABBABLE', 'MOVABLE'], 'states': []}]



====================oppo grasped======================
[{'id': 386, 'category': 'PRE_PRO_Fork_01', 'class_name': 'cutleryfork', 'prefab_name': 'PRE_PRO_Fork_01', 'obj_transform': {'position': [-8.175909, 0.9091671, 0.271380424], 'rotation': [-0.673237145, -0.714319468, 0.174941242, -0.07678178], 'scale': [1.0, 0.999999762, 1.0]}, 'bounding_box': {'center': [-8.176578, 0.9476023, 0.256823123], 'size': [0.0437824465, 0.254153669, 0.116975658]}, 'properties': ['GRABBABLE', 'MOVABLE'], 'states': []}]

===================self.current_room============
{'id': 56, 'category': 'Rooms', 'class_name': 'kitchen', 'prefab_name': 'PRE_ROO_Kitchen_02', 'obj_transform': {'position': [-3.75, 0.0, 6.25], 'rotation': [0.0, -0.7071068, 0.0, 0.7071067], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [-6.249999, 1.25, 2.5], 'size': [8.0, 3.0, 8.0]}, 'properties': [], 'states': []}
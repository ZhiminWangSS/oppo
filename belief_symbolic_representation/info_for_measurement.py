import re
from init_info import agent_info

measurement_observation = {}
current_frame = agent_info["obs"]["current_frames"]
current_room = agent_info["current_room"]
holding = ['','']
container = ['','']
oppo_holding = ['','']
oppo_container = ['','']
oppo_pro_holding = None
for id ,item in enumerate(agent_info['obs']['held_objects']):
    if item['id'] is not None:
        holding[id] = item["name"] + "<" + str(item["id"]) + ">"
        if item['contained'] != [None, None, None]:
            container[id] = 'with'
            for obj,index in enumerate(item["contained"]):
                if obj != None:
                    container[id] += item['contained_name'][index] + '<' + obj + '>, '
            container[id] += "in it. "
if holding[0] == '' and holding[1] == '':
    pro_holding = "holding nothing "
else:
    pro_holding = "I'm holding" + holding[0] + container[0] + holding[1] + container[1]
seeing = 'I see '

for item in agent_info["visible_objects"].values():
    if item["type"] != 3:
        seeing += item["name"] + '<' + str(item["id"]) + '>' + " in " + item["position"] + '. '
    else:
        last_agent_position = item['position'][5:]
        last_see_frame = item['position'][:2]
if int(last_see_frame) == current_frame:
    for id ,item in enumerate(agent_info['obs']['oppo_held_objects']):
        if item['id'] is not None:
            oppo_holding[id] = item["name"] + "<" + str(item["id"]) + ">"
            if item['contained'] != [None, None, None]:
                oppo_container[id] = 'with'
                for obj,index in enumerate(item["contained"]):
                    if obj != None:
                        oppo_container[id] += item['contained_name'][index] + '<' + obj + '>, '
                oppo_container[id] += "in it. "
    if oppo_holding[0] == '' and oppo_holding[1] == '':
        oppo_pro_holding = " partner is holding nothing"
    else:
        oppo_pro_holding = "partner is  holding" + oppo_holding[0] + oppo_container[0] + oppo_holding[1] + oppo_container[1]    
observation = "At " + str(current_frame) + ", I'm in " + current_room + ', ' + pro_holding + seeing
observation += oppo_pro_holding if oppo_pro_holding else ''

messages = agent_info['obs']['messages']
print(observation)
measurement_observation['observation'] = observation
measurement_observation['alice'] = messages[0]
measurement_observation['bob'] = messages[1]
print(measurement_observation)


    

    








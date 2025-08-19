import re

def process_agent_list(text, agent_names_list, new_subgoal_content="new subgoal content"):
    """
    遍历agent_names_list，对每个agent_name进行处理：
    - 如果文本中没有agent(agent_name)，则添加模板
    - 如果有，则查找其后第一个subgoal并替换内容
    """
    result_text = text
    
    for agent_name in agent_names_list:
        # 检查是否已存在该agent
        agent_pattern = f'agent\\({re.escape(agent_name)}\\)'
        agent_matches = list(re.finditer(agent_pattern, result_text))
        
        if not agent_matches:
            # 如果不存在，则添加模板
            print(f"Agent {agent_name} 不存在，添加模板")
            template = f'''agent({agent_name})\n   - location(Unknown)\n   - objects_in_hand[Unknown,Unknown] \n   - subgoal("{new_subgoal_content}")
'''
            result_text += template
        else:
            # 如果存在，则替换第一个subgoal（从后往前处理避免位置偏移）
            print(f"Agent {agent_name} 存在，替换subgoal")
            for match in reversed(agent_matches):
                agent_end_pos = match.end()
                
                # 查找该agent后的第一个subgoal
                # 使用正则表达式查找subgoal(...)
                remaining_text = result_text[agent_end_pos:]
                subgoal_match = re.search(r'subgoal\([^)]*\)', remaining_text)
                
                if subgoal_match:
                    # 计算subgoal在原文中的绝对位置
                    subgoal_start = agent_end_pos + subgoal_match.start()
                    subgoal_end = agent_end_pos + subgoal_match.end()
                    
                    # 替换subgoal内容
                    old_subgoal = subgoal_match.group(0)
                    new_subgoal = f'subgoal("{new_subgoal_content}")'
                    
                    result_text = result_text[:subgoal_start] + new_subgoal + result_text[subgoal_end:]
                    print(f"  已替换 {old_subgoal} -> {new_subgoal}")
                else:
                    print(f"  未找到 {agent_name} 后的subgoal")
    
    return result_text

# 测试示例
text = '''
agent(Mary)
- subgoal($SUBGOAL$)
- location(bedroom)
- subgoal:("open door")
- object_in_hand[book, pen]
     
agent(John)
- subgoal($OTHER_GOAL$)
- location(kitchen)
- subgoal:("close window")
'''

agent_list = ['Mary', 'John', 'Alice', 'Bob']

print("原始文本:")
print(text)
print("\n" + "="*50)

result = process_agent_list(text, agent_list, "walk to destination")

print("\n处理后文本:")
print(result)
import os
import json
import pandas as pd
import openai
from typing import Dict, Any, Optional
import re

api_key=os.environ.get("CHATANYWHERE_API_KEY")
base_url=os.environ.get("CHATANYWHERE_URL")

client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
)


class BeliefBuilder:
    def __init__(self, csv_path: str):
       
        self._load_prompts(csv_path)
        self.belief = None
        self.history = []
        self.suggestions = None
        self.previous_content = None

    def _load_prompts(self, csv_path: str) -> Dict[str, str]:

        self.prompts = {}
        try:
          
            df = pd.read_csv(csv_path, encoding="utf-8")
            self.prompts["init"] = df['prompt'][0]
            self.prompts['debate'] = df["prompt"][1]
            self.prompts['refine'] = df['prompt'][2]
        except Exception as e:
            print(e)

    def _call_openai_api(self, prompt: str) -> str:
       
        self.history.append({"role": "user", "content": prompt})
        messages = [{"role": "user", "content": prompt}]

        try:
            response = client.chat.completions.create(
                # model="qwen3-235b-a22b-instruct-2507",
                model="gpt-4o",
                messages=messages,
                temperature=0,
               
                max_tokens=2000,
            )
            output = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": output})
            return output
        except Exception as e:
            print(e)
            return ""

    def init_construction(self, challenge_description: str,belief_language):
        
        prompt = self.prompts.get("init", "")
        prompt = prompt.replace("$Task_description$", challenge_description)
        prompt = prompt.replace("$Belief_language$", belief_language)
        print("=================init prompt===================\n",prompt)
        if not prompt:
            raise ValueError("unfound init prompt")
       
        response = self._call_openai_api(prompt)
        print(response)
        return response

    def discussion(self,challenge_description:str,content:str,belief_language):
        
        prompt = self.prompts.get("debate", "")
        if not prompt:
            raise ValueError("unfound discussion prompt")
    
        prompt = prompt.replace("$Task_description$", challenge_description)
        prompt = prompt.replace('$Alice_content$',content)
        prompt = prompt.replace("$Belief_language$", belief_language)
        response = self._call_openai_api(prompt)
        print("=================discuss prompt===================\n",prompt)
        print(response)
        return response
        

    def refine(self,challenge_des:str,previous_content:str,suggestions:str,belief_language):
        
        prompt = self.prompts.get("refine", "")
        prompt = prompt.replace("$Task_description$", challenge_des)
        prompt = prompt.replace('$previous_content$',previous_content)
        prompt = prompt.replace("$Belief_language$", belief_language)
        prompt = prompt.replace('$suggestions$',suggestions)
        if not prompt:
            raise ValueError("prompt unfound")
    
        response = self._call_openai_api(prompt)
        print("=================refine prompt===================\n",prompt)
        print(response)
        return response

    
    def build_complete_belief(self, challenge_description: str,outputfile:str, belief_language):
        
        
        
        for i in range(3):
            if i == 0:
                construction = self.init_construction(challenge_description,belief_language)
            else:
                construction = self.refine(challenge_description,self.previous_content,self.suggestions,belief_language)

            self.previous_content = construction

            discussion = self.discussion(challenge_description,construction,belief_language)
            self.suggestions = discussion
            match = re.search(r'satisfied:\s*([a-zA-Z]+)', discussion, re.IGNORECASE)
            satisfied = None
            if match:
                satisfied  =  match.group(1).strip()
            print(satisfied)
            if satisfied.lower() == 'yes':
                break
        final_construction = self.previous_content
        match = re.search(r'zero order belief rules:\s*(.*)', final_construction, re.DOTALL | re.IGNORECASE)
        if match:
            final_construction = match.group(1).strip()
        final_construction = "zero order belief rules:\n" + final_construction
        self.save_belief(final_construction,outputfile)

    def save_belief(self, belief, output_path: str) -> None:
        
        with open(output_path, 'w') as f:
            f.write(str(belief))
            f.write("\n")



def main():

    csv_path = r"./belief_symbolic_representation/construct.csv"

    builder = BeliefBuilder(csv_path)

    challenge_cwah = "In this task, multi agents cooperate to finish a housework in a multiple-room household scene. The objects are initially in any room or cabnet. The cabnets can contain objects, and the cabnets can be checked or unchecked. Agents can hold objects to transport them to the target table. The room's exploration state includes explored and unexplored."
    
    challenge_tdw = "In this task, multi agents cooperate to finish a housework in a multiple-room household scene. Agents are set randomly in the room. The objects and containers are initially IN any room. The container can CONTAIN several objects. Agents can HOLD objects to transport them to the bed, once the object arrive the bed the object disappear. Agents can hold container and put objects into them to hold more objects at a time. Agent can get the rooms' exploration state. Agent can know each others' information like which room they are in through communication"
    belief_language = '''
Syntax:
?belief = ?entity PREDICATE [?entity:?confidence] OR ?entity ATTRIBUTE [?state:?confidence]
PREDICATE — a relational verb or state descriptor (e.g., IN, HOLD, BELIEVE, AT, INSIDE)
ATTRIBUTE — a property or characteristic of an entity (e.g., EXPLORATION_STATE, CLEAN_STATE)
?entity — a placeholder for any agent, object, or location in the environment (e.g., agentA, apple, room3)
?state — a specific condition or status of an entity (e.g., part, opened)
?confidence — one of: certain, high, medium, low

Zero-order belief:
?agent BELIEVE ?belief
Example: 
agentA BELIEVE apple IN [kitchen:high]
agentA BELIEVE livingroom EXPLORED [part:high]

First-order belief:
?agentA BELIEVE ?agentB BELIEVE ?belief
Example: agentA BELIEVE agentB BELIEVE banana IN [pantry:medium]

'''
    belief_language_no_conf = """
Syntax:
?belief = ?entity PREDICATE ?entity OR ?entity ATTRIBUTE ?state
?entity - a placeholder for any agent, object, or location in the environment (e.g., agentA, apple, room3)
?state - a placeholder descibe the entity's attributes (e.g. EXPLORED)
ATTRIBUTE - state descriptor (e.g., EXPLORED, CLEANED, CHECKED)
PREDICATE — a relational verb  (e.g., IN, HOLD, BELIEVE, AT, INSIDE)

Zero-order belief format:
?agent BELIEVE ?belief
Example: 
agentA BELIEVE apple IN <kitchen>(2000)
agentA BELIEVE <livingroom>(1000) EXPLORED part

First-order belief format:
?agentA BELIEVE ?agentB BELIEVE ?belief
Example: agentA BELIEVE agentB BELIEVE banana IN <bedroom>(3000)

"""


    os.makedirs("./belief_rules",exist_ok=True)
    outputfile = "./belief_rules/rules_test.txt" 
    builder.build_complete_belief(challenge_tdw,outputfile,belief_language_no_conf)
    process = builder.history
    

if __name__ == "__main__":
    main()

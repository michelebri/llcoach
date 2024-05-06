import os

from action import Action
from utils import parse_actions

# Define the actions as a list of dictionaries


actions_directory = "actions"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(current_script_dir, actions_directory)

if not os.path.exists(full_path):
    os.makedirs(full_path)

# Open the "actions1.txt" file in the domains directory and read its contents
with open("actions1.txt", "r") as file:
    actions_str = file.read()

action_data = parse_actions(actions_str=actions_str)

# Create Action instances
for action_data in actions_data:
    action_instance = Action(
        action_data["action_id"],
        action_data["description"],
        action_data["args"],
        action_data["preconditions"],
        action_data["effects"]
    )
    action_instance.save_to_file(full_path)

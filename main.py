
from RAG import RAG
import os
import datetime
from utils import prepare_actions_str, prepare_agents_str, initialize_actions
from visual_coach import VLM

current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Directories #
planning_domains_dir = os.path.join(current_script_dir, "experiments", "domains")
scenarios_dir = os.path.join(current_script_dir, "experiments", "scenarios")
actions_dir = os.path.join(current_script_dir, "experiments", "actions")
agents_dir = os.path.join(current_script_dir, "experiments", "agents")
frames_dir = os.path.join(current_script_dir, "experiments", "frames")
actions_database_dir = os.path.join(current_script_dir, "action_db")

############
# SETTINGS #
############
ENABLE_PRINT = False
USE_VLM = True
ALLOW_PLAN_REFINEMENT = False
REINITIALIZE_RAG = True
USE_COACH_RECOMMENDATIONS = True
##################
#########
# COACH #
#########
##########
# DOMAIN #
##########
#Load domain from the chosen txt file in the experiments/domains directory
DOMAIN_NAME = "domain.txt"
##########
# AGENTS #
##########
#Load agents from the chosen txt file in the experiments/agents directory
AGENTS_NAME = "agents.txt"
# SCENARIO #
############
#Load scenario from the chosen txt file in the experiments/scenarios directory (full path)
SCENARIO_NAME = "scenario.txt"
################
# ACTIONS FILE #
################
# Load actions from the chosen txt file in the experiments/actions directory
ACTIONS_FILE_NAME = "actions.txt"
########
# GOAL #
########
# Planning goal to be passed to the VLM, the action retrieval agent and the downstream plan selector
planning_goal = "The own team should score a goal in the opponent's goal."
CURRENT_TEAM_COLOR = "red"
CURRENT_FRAME = "action_3.png"
file_name = CURRENT_FRAME.split(".")[0]+ "_" + CURRENT_TEAM_COLOR + "team_scoreGoal_.txt"

# Load the coach frame
frame_path = os.path.join(frames_dir, CURRENT_FRAME)

# Load the domain
with open(os.path.join(planning_domains_dir, DOMAIN_NAME), "r") as file:
    planning_domain = file.read()

# Load the scenario
with open(os.path.join(scenarios_dir, SCENARIO_NAME), "r") as file:
    scenario = file.read()

# Load the agents
with open(os.path.join(agents_dir, AGENTS_NAME), "r") as file:
    agents = file.read()

# If REINITIALIZE_RAG is True, reinitialize the actions database
if REINITIALIZE_RAG:
    initialize_actions(actions_database_dir, os.path.join(actions_dir, ACTIONS_FILE_NAME))


if __name__ == "__main__":

    #######################
    # AGENT 1: VLM Coach  #
    #######################

    rag = RAG(actions_database_dir)

    # Extract
    usable_actions = rag.extract_usable_actions(planning_goal, planning_domain)

    usable_actions_str, action_ids_str = prepare_actions_str(usable_actions, actions_database_dir)
    with open("plan_results/"+ file_name,"w") as savefile:
        coach_recommendations = ""
        if USE_VLM:
            vlm = VLM(usable_actions_str)
            tactic = 'use the robot that are nearest to the opponents goal'
            coach_scenario, coach_recommendations = vlm.get_coaching(frame_path, CURRENT_TEAM_COLOR, planning_goal,tactic)
            savefile.write("OUTPUT VLM \n")
            savefile.write("\n############ COACH SCENARIO ###############\n")
            savefile.write(coach_scenario)
            savefile.write("\n############ COACH RECOMMENDATIONS ###############\n")
            savefile.write(coach_recommendations)
    
        ########################################################
        # AGENT 2: Coach Recommendations  -> High-level Plan   #
        ########################################################

        #1 ) General information about the task

        SYSTEM_PROMPT = """
        You are an AI system that has to generate a high-level plan for the team using the recomandations of a Human coach.
        Taking in input the coach reccomendations, the planning domain, the planning goal, the scenario and the agents,
        generate a high-level plan that achieves the planning goal in the shortest number of steps.
        In the plan absolutly you cannot write action about the opponent team players.
        
        """
        #2) Information about the planning domain
        SYSTEM_PROMPT += "\n\nPLANNING DOMAIN:\n\n"+planning_domain

        #3) Information about the actions retrieved from the vector database
        SYSTEM_PROMPT += "\nACTIONS THAT CAN BE PERFORMED:\n"+usable_actions_str

        #4) Information about the ouptut
        SYSTEM_PROMPT += """\n
            The output plan must have the form of newline-separated lists of actions, 
            each action is represented by its ID, the agent performing that action, followed by its arguments:
                ACTION_ID AGENT_ID ARGUMENTS
            where ACTION_ID is one of the following:
            """
        #5) Information about the agents

        SYSTEM_PROMPT += action_ids_str

        SYSTEM_PROMPT += agents

        SYSTEM_PROMPT += """\n
        AGENT_ID is one of the following:
        """
        SYSTEM_PROMPT += prepare_agents_str(agents)
            
        #6) Additional information
        SYSTEM_PROMPT +="""\n\n\n\n
        ARGUMENTS is a dictionary of the form {ARGUMENT_NAME: ARGUMENT_VALUE}.
        ALWAYS COMPLY with the TYPE OF ARGUMENTS of the actions.
        The ARGUMENTS are case-sensitive. 
        The ACTION_ID is case-sensitive. 
        PRECONDITIONS determine the conditions that must be met for the action to be executed.
        In particular, a ROBOT should be in a certain WAYPOINT to perform an action.
        AGENT_ID is the agent performing the action.
        AGENT_ID can ONLY perform actions that are allowed.
        The high-level plan MUST ACHIEVE the COACH RECOMMENDATIONS.
        The SCENARIO describes the current configuration of elements on the field:
        """
        if ENABLE_PRINT:
            print(SYSTEM_PROMPT)

        #/7) Query the LLM with SCENARIO and COACH RECOMMENDATIONS
        LLM_QUERY = ""
        if USE_VLM:
            LLM_QUERY += "\nSCENARIO\n" + coach_scenario
        else:
            LLM_QUERY += "\nSCENARIO\n"  + scenario

        if USE_COACH_RECOMMENDATIONS:
            if USE_VLM:
                LLM_QUERY += "\nCOACH RECOMMENDATIONS\n" + coach_recommendations
            else:
                LLM_QUERY += "\nCOACH RECOMMENDATIONS\n" + planning_goal

        plan_refinement = rag.query_llm(SYSTEM_PROMPT, LLM_QUERY, temperature=0.9, max_tokens=500)

        ##################
        # AGENT 2 OUTPUT #
        ##################
        if ENABLE_PRINT: 
            print("##################\n# AGENT 2: enumerate high-level plans #\n##################")
            print(plan_refinement)
        savefile.write("\n##########HIGH LEVEL PLANS###############")
        savefile.write(plan_refinement)




        ########################################################
        # AGENT 3 : Parallelize the plan                       #
        ########################################################
        PLAN_PARALLELIZER_SYSTEM_PROMPT = """
        You are an agent tasked with finding out which actions in a multi-agent plan can be executed at the same time. 
        If several actions can be performed by different agents at the same time, you should put them in a JOIN \{\} block. For example:
            If ACTION1 precedes ACTION2 and ACTION2 can be executed at the same time as ACTION3, you should output:
                ACTION1 AGENT1 ARGS
                JOIN {ACTION2 AGENT1 ARGS, ACTION3 AGENT2 ARGS}
        otherwise, return the original actions as in the original plan.
        
        EXTREMELY IMPORTANT: respect the order of the actions in the plan.
        
        EXTREMELY IMPORTANT: no JOIN blocks with a single action. NEVER:
            JOIN{ACTION1 AGENT1 ARGS}

        EXTREMELY IMPORTANT: the same agent cannot perform two actions at the same time. NEVER:
            JOIN{ACTION1 AGENT1 ARGS, ACTION2 AGENT1 ARGS}

        You should only write the resulting plan, no more, no less. Actions and JOIN blocks should be separated by newlines.
        DO NOT CHANGE the structure of the plan, only add JOIN blocks where necessary.
        """

        
        if ALLOW_PLAN_REFINEMENT:
            PLAN_PARALLELIZER_SYSTEM_PROMPT += "\nYou are allowed to modify the resulting plan but keep the same structure and encoding."
        else:
            PLAN_PARALLELIZER_SYSTEM_PROMPT += "\nYou are NOT allowed to modify the resulting plan but keep the same structure and encoding."

        # Query the LLM with the high-level plan
        
        PLAN_PARALLELIZER_QUERY = plan_refinement

        if USE_COACH_RECOMMENDATIONS:
            if USE_VLM:
                PLAN_PARALLELIZER_QUERY += "\nCOACH RECOMMENDATIONS\n" + coach_recommendations
            else:
                pass

        PLAN_PARALLELIZER_SYSTEM_PROMPT += "\nAnswer with just the plan and nothing else."

        # SELECT THE PARALLELIZED PLAN #

        parallelized_plan = rag.query_llm(PLAN_PARALLELIZER_SYSTEM_PROMPT, PLAN_PARALLELIZER_QUERY, temperature=0.1, max_tokens=200, top_p=1.0)

        ##################
        # AGENT 3 OUTPUT #
        ##################
        if ENABLE_PRINT: 
            print("\n\n######################\n# AGENT 4: parallelize plan #\n######################\n\n")
            print(parallelized_plan)
        
        savefile.write("\n##########PARALLELIZED PLAN###############\n")
        savefile.write(parallelized_plan)

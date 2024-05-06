from openai import OpenAI
import os
import base64

class VLM:

    def __init__(self,useful_actions):
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.txt")
                # Verify if the file "key.txt exists" else raise an exception
        if not os.path.exists(key_file):
            raise Exception("OpenAI key file not found")

        # Load the openai key from a local file called "key.txt"
        with open(key_file, "r") as file:
            openai_key = file.read().strip()

            self.client = OpenAI(api_key=openai_key)
        self.useful_actions = useful_actions

#TODO: Import the domain, actions and agents description from the main script 
# (they are loaded from the corresponding files)
    def coach_response(self,image_path,color_team,planning_goal,tactic):

        with open(image_path, "rb") as im_file:
            encoded_image = base64.b64encode(im_file.read()).decode("utf-8")
            agent = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
        "text":"Assess the current soccer match depicted in the image and describe the action. \n\
            As a coach assisting the team with" + color_team + "dots, provide a precise summary using the format below.\n\
            each player's position should be related to designated waypoints on the field.\n\
                    SCENARIO:\n\
                        <ROLE_OWN_TEAM> is at <WAYPOINT>\n\
                        <ROLE_OPPONENT_TEAM> is at <WAYPOINT>\n\
                        ...\n\
                        BALL is at <WAYPOINT>\n\
                    ROLES:\n\
\n\
                        STRIKER: Aims to score goals and has the ball.\n\
                        GOALIE: Defends the goal.\n\
                        JOLLY: Supports the team dynamically helping to score a goal.\n\
                        Specify if the player is the OWN_TEAM or OPPONENT_TEAM.\n\
\n\
                    WAYPOINTS:\n\
\n\
                        CENTER_CIRCLE: Middle of the field.\n\
                        OUR_GOAL: Our team's goal area.\n\
                        MIDFIELD_LINE: Divides the field in half.\n\
                        OPPONENT_GOAL: Opponent's goal area.\n\
                        DEFENSIVE_THIRD: Our defensive zone.\n\
                        OFFENSIVE_THIRD: Our attacking zone near the opponent's goal.\n\
                        CORNER_FLAG: Corner of the field.\n\
                        PENALTY_BOX: Area for penalties near the goal.\n\
\n\
                    After describing the SCENARIO, provide coaching advice.\n\
                 Your suggestions should utilize" + self.useful_actions + " and recommend player movements\n\
                 to achieve the objective of "+ planning_goal +". Your attitude is perform the following tactic " +tactic+"\n\
                For the suggestion remember this tips : ' when you outline the plan that each action you write changes\n\
        the situation so the next steps should be written so that the previous action changes the game situation.\n\
         If a robot pass the ball the robot in the next step cannot kick or pass.\n\
         The other robot need to wait to receive the ball.\n\
        A robot cannot receive a pass if is not at the WAYPOINT DESTINATION. \n\
        If you want to re pass the ball you have to write a new moving action for the previous robot.\n\
        EXTREMELY IMPORTANT: If you write a pass action you have to write a moving action for the SENDER AND RECEIVER in order to\n\
        have the robot in waypoint starting and destination.'\n\
                    COACH ADVICE:\n\
                    [give the actions to be performed avoiding conditionals but describing the precise steps to be performed in sequence.]\n\
\n\
                    Ensure the SCENARIO and COACH ADVICE sections are distinct, using the '######' token as a separator."},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                        },
                    ],
                    }
                ],
                max_tokens=400,
                temperature=0.1,
                top_p=0.8,
            )

            response = (agent.choices[0].message.content)
            print(response)

            return response


    def get_coaching(self, image_path, color_team, planning_goal,tactic):
        vlm_response = self.coach_response(image_path, color_team, planning_goal,tactic)

        #Split the response at the ###### token
        response_parts = vlm_response.split("######")
        return response_parts[0], response_parts[1]




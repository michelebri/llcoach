import os

class Action:
    def __init__(self, action_id, description, args, preconditions, effects, file_path=None):
        self.action_id = action_id
        self.description = description
        self.args = args
        self.preconditions = preconditions
        self.effects = effects
        self.file_path = file_path

    def save_to_file(self, directory):
        filename = os.path.join(directory, f"{self.action_id}.txt")
        self.file_path = filename
        with open(filename, "w") as file:
            file.write("ACTION\n")
            file.write(f"    ID: {self.action_id}\n")
            file.write(f"    DESCRIPTION: {self.description}\n")
            file.write("    ARGS:\n")
            for arg, arg_type in self.args.items():
                file.write(f"        {arg}: {arg_type}\n")
            file.write("    PRECONDITIONS:\n")
            for precondition in self.preconditions:
                file.write(f"        {precondition}\n")
            file.write("    EFFECTS:\n")
            for outcome in self.effects:
                file.write(f"        {outcome}\n")

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r") as file:
            lines = file.readlines()
        action_id = lines[1].strip().split(":")[1].strip()
        description = lines[2].strip().split(":")[1].strip()
        args = {}
        preconditions = []
        effects = []
        effect = False
        for line in lines[4:]:
            line = line.strip()
            if line.startswith("ARGS"):
                arg_parts = line.split()
                arg_name = arg_parts[1].strip(":")
                arg_type = arg_parts[2]
                args[arg_name] = arg_type
            elif line.startswith("PRECONDITIONS"):
                preconditions.append(line.split(":")[1].strip())
            elif line.startswith("EFFECTS"):
                effect = True
            elif effect:
                result = line
                effects.append(result)
        return Action(action_id, description, args, preconditions, effects, file_path=filename)

    def __repr__(self):
        return f"""
        ACTION
            ID: {self.action_id}
            DESCRIPTION: {self.description}
            ARGS:
                {self.args}
            PRECONDITIONS:
                {self.preconditions}
            EFFECTS:
                {self.effects}
        """
    
    # Prints as it is printed in the file
    def __str__(self):
        return f"""
        ACTION
            ID: {self.action_id}
            DESCRIPTION: {self.description}
            ARGS:
                {self.args}
            PRECONDITIONS:
                {self.preconditions}
            EFFECTS:
                {self.effects}
        """

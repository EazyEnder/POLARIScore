from config import EXPORT_FOLDER
import os
from datetime import datetime

class Logger():
    def __init__(self, level=0, auto_save=5):
        self.messages = []
        self.level = level
        self.log_file = None
        self.auto_save = auto_save
        self.global_color = "32m"
        self.print_borders = True

        self._init_gc = self.global_color

    def reset(self):
        self.global_color = self._init_gc

    def print(self, message, color="0m", type=None, level=0):
        if self.level < level:
            return None
        if type is None:
            string = message
        else:
            string = f"(\033[{color}{type.upper()}\033[0m) {message}"
        print(string)
        self.messages.append(string)
        if self.auto_save > 0 and len(self.messages) % self.auto_save == 0:
            self.save()
        return string
    
    def border(self, message="", color=None, level=2):
        if not(self.print_borders):
            return ""
        if color is None:
            color = self.global_color
        WIDTH = 30
        message = f" {message} "
        dashes = '-' * ((WIDTH - len(message)) // 2)
        if len(dashes) * 2 + len(message) < WIDTH:
            border_line = f"{dashes}{message}{dashes}-"
        else:
            border_line = f"{dashes}{message}{dashes}"
        return self.print(f"\033[{color}{border_line}\033[0m", level=level)
    
    def warn(self, message):
        return self.print(message, type="warn", color="33m", level=1)
    def error(self, message):
        message = "\033[31m"+message+"\033[0m"
        return self.print(message, type="error", color="31m", level=0)
    def log(self, message):
        return self.print(message, type="info", color=self.global_color, level=2)

    def save(self):
        export_path = os.path.join(EXPORT_FOLDER,"logs")
        if not(os.path.exists(export_path)):
            os.mkdir(export_path)
        self.log_file = os.path.join(export_path, f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt")
        with open(self.log_file, "w") as file:
            for message in self.messages:
                file.write(message + "\n")
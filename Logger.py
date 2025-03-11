from config import EXPORT_FOLDER
import os
from datetime import datetime

class Logger():
    def __init__(self, level=0, auto_save=5):
        self.messages = []
        self.level = level
        self.log_file = None
        self.auto_save = auto_save

    def print(self, message, type="warn", level=0):
        if self.level < level:
            return None
        string = f"({type.upper()}) {message}"
        print(string)
        self.messages.append(string)
        if self.auto_save > 0 and len(self.messages) % self.auto_save == 0:
            self.save()
        return string

    def warn(self, message):
        return self.print(message, type="warn", level=1)
    def error(self, message):
        return self.print(message, type="error", level=0)
    def log(self, message):
        return self.print(message, type="info", level=2)

    def save(self):
        export_path = os.path.join(EXPORT_FOLDER,"logs")
        if not(os.path.exists(export_path)):
            os.mkdir(export_path)
        self.log_file = os.path.join(export_path, f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt")
        with open(self.log_file, "w") as file:
            for message in self.messages:
                file.write(message + "\n")
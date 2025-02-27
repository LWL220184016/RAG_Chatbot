import datetime

class Database_Handler:
    def __init__(
            self, 
        ):

        self.dataID = 0

    def get_newest_chat_name():
        this_month = datetime.datetime.now().strftime("%Y-%m")
        return "chat_" + this_month

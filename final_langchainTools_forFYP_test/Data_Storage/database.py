import datetime

class Database_Handler:
    def __init__(
            self, 
        ):

        self.dataID = 0

    def get_newest_chat_name() -> str:
        """
        傳回最新的對話歷史資料和集的名稱 (chat_YYYY_MM)
            - 例如: "chat_2022-01"
        """

        this_month = datetime.datetime.now().strftime("%Y-%m")
        return "chat_" + this_month

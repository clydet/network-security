import sys
from networksecurity.logging.logger import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_detail:sys):
        self.error_message = error_message
        _, _, exc_tb = error_detail

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error: {self.error_message} on line {self.lineno} in {self.file_name}"

import sys #sys module is used to access system-specific parameters and functions

from SRC.logger import logging #importing logging module from src package

def error_message_detail(error,error_detail:sys): #function to get detailed error message
    _,_,exc_tb=error_detail.exc_info() #extracting exception information
    file_name=exc_tb.tb_frame.f_code.co_filename #getting the filename where the error occurred
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format( 
     file_name,exc_tb.tb_lineno,str(error)) #formatting the error message

    return error_message

    

class CustomException(Exception): #custom exception class inheriting from built-in Exception class
    def __init__(self,error_message,error_detail:sys): #initializer method to set up the exception
        super().__init__(error_message) #calling the initializer of the base Exception class
        self.error_message=error_message_detail(error_message,error_detail=error_detail) #getting detailed error message
    
    def __str__(self): #string representation of the exception
        return self.error_message #returning the detailed error message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)
        logging.info("Logging has started.")    
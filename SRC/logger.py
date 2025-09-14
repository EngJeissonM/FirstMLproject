import logging #this is the library
import os #os is used for creating the folder
from datetime import datetime #to create the log file with current date and time

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #log file name with current date and time
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #here we are creating the folder named logs
os.makedirs(logs_path,exist_ok=True) #if the folder is already there it will not give any error

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) #this is the path where the log file will be created


logging.basicConfig( #here we are going to save each log.
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)
print("sksdldfds jjjjjjjjjjjjjjjjjjj")
if __name__=="__name__":
    logging.info("logging has started")

from utils import train_models
import time
starttime=time.time()
interval=86400 

while True:
    train_models()
    time.sleep(interval - ((time.time() - starttime) % interval))
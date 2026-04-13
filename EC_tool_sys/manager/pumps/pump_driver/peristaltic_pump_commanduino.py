import threading
import time
import logging
from commanduino import CommandManager

class CommanduinoPeristalticPump:

    def __init__(self,
                 config: dict,
                 logger: logging.Logger = None):
        """
        Initialize the communication port and peristaltic pump
        Args:
            pulse_pin: number of stepper motor pulse pin
            direction_pin: number of stepper motor direction pin
            lock: threading.Lock, the lock of QUANQIUBIANLIANG
            logger: log
        """
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
        self.cmd_manager = CommandManager.from_config(config["commanduino"])
        self.pumps = config["pumps"]
        self.logger = logger
        self.lock = threading.Lock()

    def run(self, pump_name, run_time = 110):
        """
        Run the peristaltic pump
        Returns: None
        """
        self.lock.acquire()
        self.cmd_manager.devices[self.pumps[pump_name]].set_pwm_value(255)
        time.sleep(run_time)
        self.cmd_manager.devices[self.pumps[pump_name]].set_pwm_value(0)
        self.lock.release()


import threading
import json
from telemetrix import telemetrix
import time
import logging

_QUANQIUBIANLIANG = 0

def add_one():
    global _QUANQIUBIANLIANG
    _QUANQIUBIANLIANG = _QUANQIUBIANLIANG + 1

class ArduinoPeristalticPump:

    def __init__(self,
                 board: telemetrix.Telemetrix,
                 pulse_pin: int,
                 direction_pin: int,
                 lock: threading.Lock,
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

        self.board = board
        self.logger = logger
        self.pulse_pin = pulse_pin
        self.direction_pin = direction_pin
        self.peristaltic_pump = board.set_pin_mode_stepper(interface=1, pin1=self.pulse_pin, pin2=self.direction_pin)
        self.lock = lock

    def call_back(self, data):
        """
        Feedback whether the peristaltic pump is in place
        Args:
            data: a data list:[report_type, motor_id, raw_time_stamp],the report_type = 19

        Returns: None

        """
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[2]))
        self.logger.info(f'peristaltic pump completed at: {date}')

        self.lock.acquire()
        add_one()
        self.lock.release()

    def set_parameters(self, speed: int = 1000, acceleration: int = 500, position: int = 150000):
        """
        Setting of peristaltic pump parameters, including speed, acceleration and position
        Args:
            speed: the maximum permitted speed of peristaltic pump (0-1000)
            acceleration: the acceleration/deceleration rate of peristaltic pump
            position: the value of the target position relative to the current position.

        Returns: None

        """
        self.board.stepper_set_current_position(self.peristaltic_pump, 0)
        self.board.stepper_set_max_speed(self.peristaltic_pump, speed)
        self.board.stepper_set_acceleration(self.peristaltic_pump, acceleration)
        self.board.stepper_move(self.peristaltic_pump, position)
        time.sleep(0.2)

    def run(self):
        """
        Run the peristaltic pump
        Returns: None

        """
        self.board.stepper_run(self.peristaltic_pump, completion_callback=self.call_back)
        time.sleep(0.2)

    def close(self):
        """
        Close the peristaltic pump
        Returns: None

        """
        self.board.shutdown()

class ArduinoPeristalticPumpSet:
    def __init__(self, config: dict = None, logger: logging.Logger = None):

        self.board = telemetrix.Telemetrix(config["port_num"])
        self.pumps = {}
        self.lock = threading.Lock()
        self.logger = logger

        # set up the individual pumps
        for i in list(config.keys()):
            if i != "port_num":
                self.pumps[i] = ArduinoPeristalticPump(
                    board = self.board,
                    pulse_pin = config[i]["pulse_pin"],
                    direction_pin = config[i]["direction_pin"],
                    lock = self.lock,
                    logger = self.logger)

    def run(self):
        global _QUANQIUBIANLIANG
        for motor in list(self.pumps.values()):
            motor.set_parameters()
        for motor in list(self.pumps.values()):
            motor.run()

        while _QUANQIUBIANLIANG != len(list(self.pumps.values())):
            time.sleep(0.2)

        self.lock.acquire()
        _QUANQIUBIANLIANG = 0
        self.lock.release()

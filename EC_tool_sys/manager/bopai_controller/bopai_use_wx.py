from .base import GASAxis
import logging
import threading
import time


class Bopai(GASAxis):

    def __init__(self, motion_prm: dict, logger: logging.Logger = None):
        """
        Initialize the Bopai controller.
        
        Args:
            motion_prm (dict): Motion parameters for the controller.
            logger (logging.Logger): Logger for logging information.
        """
        super().__init__(motion_prm, logger)
        self.lock = threading.Lock()

    def stir_on(self, cell_name, rotate_speed: int = 1000):
        """
        Turn on the stirrer for a cell.
        
        Args:
            cell_name (str): Name of the cell.
            rotate_speed (int): Rotation speed of the stirrer.
        """
        self.lock.acquire()
        self.jog(axis_index = cell_name, vel = rotate_speed)
        self.lock.release()

    def stir_off(self, cell_name, rotate_speed: int = 0):
        """
        Turn off the stirrer for a cell.
        
        Args:
            cell_name (str): Name of the cell.
            rotate_speed (int): Rotation speed of the stirrer.
        """
        self.lock.acquire()
        self.jog(axis_index = cell_name, vel = rotate_speed)
        self.lock.release()

    def empty_the_cell(self, output, time_sleep = 20):
        """
        Empty the cell.
        
        Args:
            output (str): Output channel for emptying the cell.
            time_sleep (int): Time to wait after triggering the output.
        """
        self.output_trigger(op_index = output, onoff = True)
        time.sleep(time_sleep)
        self.output_trigger(op_index = output, onoff = False)
    def cat_cell2(self, output = "cat_cell"):
        """
        Empty the cell.
        
        Args:
            output (str): Output channel for emptying the cell.
            time_sleep (int): Time to wait after triggering the output.
        """
        self.output_trigger(op_index = output, onoff = True)

    def cat_cell1(self, output = "cat_cell"):
        """
        Empty the cell.
        
        Args:
            output (str): Output channel for emptying the cell.
            time_sleep (int): Time to wait after triggering the output.
        """
        self.output_trigger(op_index = output, onoff = False)
        
    
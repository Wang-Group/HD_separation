import time
from .motion_controller import motion_wx_use
from pymodbus.client.tcp import ModbusTcpClient
from pymodbus.bit_read_message import ReadCoilsResponse
import logging
from telemetrix import telemetrix
import threading


class RoboticPlatformMassHD:
    def __init__(self, rail_param, arm_ip_address, arm_port, logger: logging.Logger, relay_config: dict = None):
        self.client = ModbusTcpClient(arm_ip_address, port = arm_port)
        self.client.connect()
        self.slide_rail = motion_wx_use.MotionController(param = rail_param, logger = logger)
        self.slide_rail.set_motion_params()

        if relay_config is not None:
            self.relay = telemetrix.Telemetrix(relay_config["relay_port_num"])
            self.relay_pin = relay_config["relay_pin"]

    def robot_operation(self, coil_address: int):
        """
        MG400 robot code
        Args:
            coil_address: coil register address (0-113).Each coil register address represents an operation
        """
        self.client.write_coil(coil_address, True)
        time.sleep(4)
        while True:
            result: ReadCoilsResponse = self.client.read_coils(address=0, count=8)
            if result.bits[0] == 0:
                break
        time.sleep(0.5)

    def storage_to_electrodeposition(self, lock: threading.Lock = None, batch_num = 4, preparation_coil_address = 2):
        """
        MG400 robot transfers the carbon paper from storage part to the electrodeposition part
        Args:
            preparation_coil_address: the platform has four electrolytic cells to prepare the catalyst
            by electrodeposition,and each cell corresponds to a coil address(2-5)
        """
        lock.acquire()
        for i in range(batch_num):
            self.slide_rail.home()  # slide rail moves to home
            print(f"{i} home")      
            self.robot_operation(1)  # MG400 robot transfers the carbon paper to the rotary gas
            print(f"{i} operation 1")
            self.slide_rail.single_move_abs(620)  # slide rail moves to the electrodeposition part
            print(f"{i} move to 620")
            self.robot_operation(preparation_coil_address)  # MG400 robot transfers the carbon paper to electrodeposition
            print(f"{i} operation {preparation_coil_address}")
            preparation_coil_address += 1
        self.slide_rail.home()
        lock.release()
        
    def electrodeposition_to_clean(self, catalyst_coil_address: 6):
        """
        MG400 robot transfers the catalyst from the electrodeposition part to the cleaning part
        Args:
            catalyst_coil_address: MG400 robot removes the catalyst from the electrodeposition cell.(6-9)
            
        """
        self.slide_rail.single_move_abs(620)  # slide rail moves to the electrodeposition part
        self.robot_operation(catalyst_coil_address)  # MG400 robot fetches the catalyst from the electrodeposition part
        self.slide_rail.single_move_abs(670)    # MG400 robot moves the catalyst to the wash 
        

    def clean_to_dry(self, dry_coil_address: 11):
        """
        MG400 robot transfers the catalyst from the cleaning part to the drying part
        Args:
        dry_coil_address: MG400 robot places the catalyst in the drying area, which as 24 positions corresponding to 9 coils.(11-19)

        """
        self.slide_rail.single_move_abs(100)  # slide rail moves to the drying part and electrolytic cell(HER)
        self.robot_operation(10)  # MG400 robot and rotary gas rotate the catalyst
        self.robot_operation(dry_coil_address)  # MG400 robot transfers the catalyst to the drying part

    def led_to_HD_test_cell(self, lock: threading.Lock, dry_coil_address = 70):
        """
        MG400 robot transfers the catalyst from the drying area to the electrolysis cell for catalyst screening
        Args:
            dry_coil_address:MG400 robot takes the catalyst from 24 drying areas for catalytic performance test,
            corresponding to 24 coils.(70-93)
            lock: the threading.lock of the robotic arm
        """
        lock.acquire()
        self.slide_rail.single_move_abs(100)    # slide rail moves to the drying part and electrolytic cell(HER)
        self.robot_operation(dry_coil_address)    # MG400 robot removes the catalyst from the drying part
        self.robot_operation(97)     # MG400 robot and rotary gas rotate the catalyst
        self.slide_rail.home()
        self.robot_operation(111)     # MG400 robot moves the catalyst to electrolytic cell(HER)
        lock.release()

    def HD_cell_to_waste(self, lock: threading.Lock):
        """
        MG400 robot transfers the filtered catalyst to the waste
        """
        lock.acquire()
        self.slide_rail.home()
        self.robot_operation(112)         # MG400 robot moves the catalyst from the HD_cell
        self.slide_rail.single_move_abs(100)    # slide rail moves to the drying part and HER cell
        self.robot_operation(113)         # MG400 robot moves the catalyst to the waste
        self.slide_rail.home()
        lock.release()

    def open_the_leds_to_dry(self):
        """
        MG400 robot opens the LED lights to dry the catalyst
        """
        self.robot_operation(101)

    def close_the_leds(self):
        """
        MG400 robot closes the LED lights
        """
        self.robot_operation(102)

    def led_to_HD_test_cell1(self, lock: threading.Lock, dry_coil_address = 70):
        """
        MG400 robot transfers the catalyst from the drying area to the electrolysis cell for catalyst screening
        Args:
            dry_coil_address:MG400 robot takes the catalyst from 24 drying areas for catalytic performance test,
            corresponding to 24 coils.(70-93)
            lock: the threading.lock of the robotic arm
        """
        lock.acquire()
        self.slide_rail.single_move_abs(100)    # slide rail moves to the drying part and electrolytic cell(HER)
        self.robot_operation(dry_coil_address)    # MG400 robot removes the catalyst from the drying part
        self.robot_operation(97)     # MG400 robot and rotary gas rotate the catalyst
        self.slide_rail.home()
        self.robot_operation(114)     # MG400 robot moves the catalyst to electrolytic cell1(HER)
        lock.release()

    def led_to_HD_test_cell2(self, lock: threading.Lock, dry_coil_address = 70):
        """
        MG400 robot transfers the catalyst from the drying area to the electrolysis cell for catalyst screening
        Args:
            dry_coil_address:MG400 robot takes the catalyst from 24 drying areas for catalytic performance test,
            corresponding to 24 coils.(70-93)
            lock: the threading.lock of the robotic arm
        """
        lock.acquire()
        self.slide_rail.single_move_abs(100)    # slide rail moves to the drying part and electrolytic cell(HER)
        self.robot_operation(dry_coil_address)    # MG400 robot removes the catalyst from the drying part
        self.robot_operation(97)     # MG400 robot and rotary gas rotate the catalyst
        self.slide_rail.home()
        self.robot_operation(115)     # MG400 robot moves the catalyst to electrolytic cell2(HER)
        lock.release()

    def HD_cell1_to_waste(self, lock: threading.Lock):
        """
        MG400 robot transfers the filtered catalyst to the waste
        """
        lock.acquire()
        self.slide_rail.home()
        self.robot_operation(111)         # MG400 robot moves the catalyst from the HD_cell
        self.slide_rail.single_move_abs(100)    # slide rail moves to the drying part and HER cell
        self.robot_operation(113)         # MG400 robot moves the catalyst to the waste
        self.slide_rail.home()
        lock.release()

    def HD_cell2_to_waste(self, lock: threading.Lock):
        """
        MG400 robot transfers the filtered catalyst to the waste
        """
        lock.acquire()
        self.slide_rail.home()
        self.robot_operation(112)         # MG400 robot moves the catalyst from the HD_cell
        self.slide_rail.single_move_abs(100)    # slide rail moves to the drying part and HER cell
        self.robot_operation(113)         # MG400 robot moves the catalyst to the waste
        self.slide_rail.home()
        lock.release()

    def HD_cell1_to_cell2(self, lock: threading.Lock):
        """
        MG400 robot transfers the filtered catalyst to the waste
        """
        lock.acquire()
        self.robot_operation(116)         # MG400 robot moves the catalyst from the HD_cell1 to HD_cell2
        lock.release()

    

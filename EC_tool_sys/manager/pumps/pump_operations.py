import serial
import logging
from .pump_driver import SyringePump, SwitchValve, PeristalticPump, ArduinoPeristalticPumpSet, CommanduinoPeristalticPump
import time

SLOW_SPEED = 400
MID_SPEED = 600
HIGH_SPEED = 800

class PumpOperations:
    def __init__(self,
                 port_num1: str,
                 port_num2: str,
                 port_num3: str,
                 channels: dict,
                 commands: dict,
                 logger: logging.Logger = None,
                 syringe_slave = None,
                 switch_slave = None,
                 syringe_slave2 = None,
                #  peristaltic_valve_slave = None,
                #  arduino_peristaltic_config: dict = None,
                #  arduino_peristaltic_config_D2O: dict = None,
                #  commanduino_pump_config: dict = None
                ):
        # initialize the serial communication 
        if syringe_slave is not None:
            ser1 = serial.Serial(port_num1, 9600, timeout = 0.5)
            self.syringe_pump = SyringePump(ser = ser1,
                                            slave = syringe_slave,
                                            logger = logger,
                                            commands = commands)
            self.syringe_pump.forced_reset()

            
        if switch_slave is not None:
            ser2 = serial.Serial(port_num2, 9600, timeout = 0.5)
            self.switch_valve = SwitchValve(ser = ser2,
                                            slave = switch_slave,
                                            logger = logger,
                                            commands = commands)
            self.switch_valve.restoration()

        if syringe_slave2 is not None:
            ser3 = serial.Serial(port_num3, 9600, timeout = 0.5)
            self.syringe_pump2 = SyringePump(ser = ser3,
                                            slave = syringe_slave2,
                                            logger = logger,
                                            commands = commands)
            self.syringe_pump2.forced_reset()

        # if peristaltic_valve_slave is not None:
        #     pass 
        
        # elif (arduino_peristaltic_config is not None) and (arduino_peristaltic_config_D2O is not None):
        #     self.peristaltic_pump = ArduinoPeristalticPumpSet(config = arduino_peristaltic_config,
        #                                                       logger = logger)
        #     self.peristaltic_pump_D2O = ArduinoPeristalticPumpSet(config = arduino_peristaltic_config_D2O,
        #                                                           logger = logger)
        # elif (commanduino_pump_config is not None):
        #     self.peristaltic_pump_set = CommanduinoPeristalticPump(config = commanduino_pump_config, logger = logger)
        # else:
        #     raise ValueError("peristaltic pumps are not defined!")
        
        self.channels = channels
        self.logger = logger

    def dispense_liquid(self, volume, inlet_channel, outlet_channel, in_speed=HIGH_SPEED, out_speed=HIGH_SPEED):
        """
        Transfer liquid from inlet_channel to outlet_channel.
        
        :param volume: The volume of liquid to transfer.
        :param inlet_channel: The channel from which to draw the liquid.
        :param outlet_channel: The channel to which to dispense the liquid.
        :param in_speed: The speed at which to draw the liquid.
        :param out_speed: The speed at which to dispense the liquid.
        :return: None
        """
        while volume > 0:
            if inlet_channel in list(self.channels["switch_valve"].keys()):
                self.syringe_pump.valve_switch(param = self.channels["syringe_pump"],
                                               channel = "valve")
                self.switch_valve.valve_switch(param = self.channels["switch_valve"],
                                               channel = inlet_channel)
            elif inlet_channel in list(self.channels["syringe_pump"].keys()):
                self.syringe_pump.valve_switch(param = self.channels["syringe_pump"],
                                               channel = inlet_channel)
                
            elif inlet_channel in list(self.channels["syringe_pump2"].keys()):
                self.syringe_pump2.valve_switch(param = self.channels["syringe_pump2"],
                                               channel = inlet_channel)
            
            else:
                raise ValueError(f"{inlet_channel} is not defined in the current setup. Check the connection.")

            if volume > 4:
                if (inlet_channel in list(self.channels["switch_valve"].keys())) or (inlet_channel in list(self.channels["syringe_pump"].keys())):
                    self.syringe_pump.set_dynamic_speed(in_speed)
                    self.syringe_pump.move_to_absolute_position(4)
                    time.sleep(1)
           
              

                    if outlet_channel in list(self.channels["switch_valve"].keys()):
                        self.syringe_pump.valve_switch(param = self.channels["syringe_pump"],
                                                    channel = "valve")
                        self.switch_valve.valve_switch(param = self.channels["switch_valve"],
                                                    channel = outlet_channel)
                    else:
                        self.syringe_pump.valve_switch(param = self.channels["syringe_pump"],
                                                    channel = outlet_channel)
                        
                    self.syringe_pump.set_dynamic_speed(out_speed)
                    self.syringe_pump.move_to_absolute_position(0)
                    volume -= 4
                else:
                    if volume > 9:
                        self.syringe_pump2.set_dynamic_speed(in_speed)
                        self.syringe_pump2.move_to_absolute_position(absolute_volume = 9, measuring_range = 10)
                        time.sleep(1)

                        self.syringe_pump2.valve_switch(param = self.channels["syringe_pump2"],
                                                        channel = outlet_channel)

                        self.syringe_pump2.set_dynamic_speed(out_speed)
                        self.syringe_pump2.move_to_absolute_position(0)
                        volume -= 9
                    else:
                        self.syringe_pump2.set_dynamic_speed(in_speed)
                        self.syringe_pump2.move_to_absolute_position(absolute_volume = volume, measuring_range = 10)
                        time.sleep(1)
                        self.syringe_pump2.valve_switch(param = self.channels["syringe_pump2"],
                                                        channel = outlet_channel)

                        self.syringe_pump2.set_dynamic_speed(out_speed)
                        self.syringe_pump2.move_to_absolute_position(0)
                        volume = 0

            else:
                if (inlet_channel in list(self.channels["switch_valve"].keys())) or (inlet_channel in list(self.channels["syringe_pump"].keys())):
                    self.syringe_pump.set_dynamic_speed(in_speed)
                    self.syringe_pump.move_to_absolute_position(volume)
                    time.sleep(1)

                    if outlet_channel in list(self.channels["switch_valve"].keys()):
                        self.syringe_pump.valve_switch(param = self.channels["syringe_pump"],
                                                    channel = "valve")
                        self.switch_valve.valve_switch(param = self.channels["switch_valve"],
                                                    channel = outlet_channel)
                    else:
                        self.syringe_pump.valve_switch(param = self.channels["syringe_pump"],
                                                       channel = outlet_channel)
                        
                    self.syringe_pump.set_dynamic_speed(out_speed)
                    self.syringe_pump.move_to_absolute_position(0)
                    volume = 0

                else:
                    self.syringe_pump2.set_dynamic_speed(in_speed)
                    self.syringe_pump2.move_to_absolute_position(absolute_volume = volume, measuring_range = 10)
                    time.sleep(1)
                    self.syringe_pump2.valve_switch(param = self.channels["syringe_pump2"],
                                                    channel = outlet_channel)

                    self.syringe_pump2.set_dynamic_speed(out_speed)
                    self.syringe_pump2.move_to_absolute_position(0)
                    volume = 0

    def preflush(self, volume, inlet_channel, outlet_channel, in_speed, out_speed):
        """
        Preflush the system by dispensing liquid.
        
        :param volume: The volume of liquid to preflush.
        :param inlet_channel: The channel from which to draw the liquid.
        :param outlet_channel: The channel to which to dispense the liquid.
        :param in_speed: The speed at which to draw the liquid.
        :param out_speed: The speed at which to dispense the liquid.
        :return: None
        """
        self.dispense_liquid(volume, inlet_channel, outlet_channel, in_speed, out_speed)

    def air_flush(self, cell_channel):
        """
        Perform an air flush on the specified cell channel.
        
        :param cell_channel: The channel to flush with air.
        :return: None
        """
        if (cell_channel in list(self.channels["switch_valve"].keys())) or (cell_channel in list(self.channels["syringe_pump"].keys())):
            self.dispense_liquid(volume = 8,
                                inlet_channel="water",
                                outlet_channel="waste",
                                in_speed = MID_SPEED,
                                out_speed = HIGH_SPEED)
        else:
            self.dispense_liquid(volume = 18,
                                inlet_channel="H2O",
                                outlet_channel="waste_2",
                                in_speed = MID_SPEED,
                                out_speed = HIGH_SPEED)
            
        for i in range(2):
            if (cell_channel in list(self.channels["switch_valve"].keys())) or (cell_channel in list(self.channels["syringe_pump"].keys())):
                self.dispense_liquid(volume = 2,
                                    inlet_channel="air",
                                    outlet_channel="waste",
                                    in_speed = MID_SPEED,
                                    out_speed = HIGH_SPEED)
            else:
                self.dispense_liquid(volume = 2,
                                    inlet_channel="air_2",
                                    outlet_channel="waste_2",
                                    in_speed = MID_SPEED,
                                    out_speed = HIGH_SPEED)
        for i in range(3):
            if (cell_channel in list(self.channels["switch_valve"].keys())) or (cell_channel in list(self.channels["syringe_pump"].keys())):

                self.dispense_liquid(volume = 2,
                                        inlet_channel = "air",
                                        outlet_channel = cell_channel,
                                        in_speed = MID_SPEED,
                                        out_speed = HIGH_SPEED)
            else:
                self.dispense_liquid(volume = 2,
                                    inlet_channel = "air_2",
                                    outlet_channel = cell_channel,
                                    in_speed = MID_SPEED,
                                    out_speed = HIGH_SPEED)

    def pipette_liquids(self, volume, channel, cell_channel):
        """
        Pipette liquids into the specified cell channel.
        
        :param volume: The volume of liquid to pipette.
        :param channel: The channel from which to draw the liquid.
        :param cell_channel: The cell channel to which to dispense the liquid.
        :return: None
        """
        if volume < 1e-8:
            pass 
        else:
            if (cell_channel in list(self.channels["switch_valve"].keys())) or (cell_channel in list(self.channels["syringe_pump"].keys())):
                self.preflush(volume = 2.0,
                              inlet_channel = channel,
                              outlet_channel = "waste",
                              in_speed = MID_SPEED,
                              out_speed = HIGH_SPEED)

                self.dispense_liquid(volume,
                                     channel,
                                     cell_channel,
                                     in_speed = MID_SPEED,
                                     out_speed = HIGH_SPEED)

                self.air_flush(cell_channel)
            else:
                self.preflush(volume = 9.0,
                              inlet_channel = channel,
                              outlet_channel = "waste_2",
                              in_speed = MID_SPEED,
                              out_speed = HIGH_SPEED)

                self.dispense_liquid(volume = volume,
                                     inlet_channel = channel,
                                     outlet_channel = cell_channel,
                                     in_speed = MID_SPEED,
                                     out_speed = HIGH_SPEED)

                self.air_flush(cell_channel)


    def close(self):
        """
        Close the devices.
        
        :return: None
        """
        self.syringe_pump.close()
        self.switch_valve.close()
        self.syringe_pump2.close()
        # self.peristaltic_pump.close()

from pywinauto.application import Application
import time
from pywinauto import keyboard
from pywinauto import mouse
import os
import logging
import threading
from datetime import datetime
from pywinauto import Desktop
import re
import socket
import os 
import glob
import shutil

time_out = 10

class MASS_wx:

    def __init__(self, app_path: str, logger: logging.Logger):
        # initialize the object
        self.app_path = app_path
        self.logger = logger
        self.mass = Application(backend='uia').connect(path=app_path)
        # Select the mass window
        self.mass_win = self.find_window_by_regex()
        self.mass_win.set_focus()
        self.mass_win.maximize()
        time.sleep(2)
        # self.create_new_file()

        self.if_initial_window = True

    def find_window_by_regex(self, regex_pattern = r".*MASsoft 10 Professional.*"):
        desktop = Desktop(backend="uia") 
        windows = desktop.windows()
    
        for window in windows:
            if re.search(regex_pattern, window.window_text()):
                return window

    def win_set_focus(self):
        """
        Set focus the application window
        """
        count = 0
        while count < 3:
            try:
                self.mass_win.set_focus()
                time.sleep(2)
                if self.mass_win.exists():
                    break
            except:
                count += 1
        if count == 3:
            self.logger.warning('fail to focus the mass window')

    def create_new_file(self):
        """
        Create the new file and set the parameters of the experiment
        """
        # create new file
        mouse.click(coords=(40, 70))
        time.sleep(1)

        mouse.click(coords=(1575, 117))     # select the 'maximize' button
        time.sleep(1)

        # add two scans
        mouse.click(coords=(713, 304))    # select scan1
        time.sleep(1)
        mouse.click(coords=(785, 72))  # add two scans
        time.sleep(1)
        mouse.click(coords=(785, 72))
        time.sleep(1)

        # edit the scan of mass
        self.set_mass_parameter(coordinate_mass=(713, 304), coordinate_mode=(540, 300), mass='2')
        self.set_mass_parameter(coordinate_mass=(721, 369), coordinate_mode=(542, 367), mass='3')
        self.set_mass_parameter(coordinate_mass=(715, 428), coordinate_mode=(542, 430), mass='4')

        # select all the scan
        mouse.right_click(coords=(690, 300))
        time.sleep(1)
        mouse.click(coords=(838, 581))  # select-all
        time.sleep(1)

    def set_mass_parameter(self, coordinate_mass, coordinate_mode, mass: str):
        """
        Set the mass parameters
        """
        mouse.double_click(coords=coordinate_mass)    # select 'scan' column
        time.sleep(1)
        mouse.click(coords=(870, 450))   # select MID mode
        time.sleep(1)
        mouse.double_click(coords=(840, 513))    # select the 'start Value' column
        time.sleep(1)
        keyboard.send_keys(mass)      # set the mass value
        time.sleep(1)
        mouse.click(coords=(1084, 778))   # select the 'OK' button
        time.sleep(1)

        mouse.double_click(coords=coordinate_mode)   # select 'mode' column
        time.sleep(1)
        mouse.click(coords=(1009, 386))     # select the drop-down menu
        time.sleep(1)
        mouse.click(coords=(898, 417))    # select SEM mode
        time.sleep(1)
        mouse.click(coords=(1040, 769))    # select "ok" button
        time.sleep(2)

    def start_mass(self, lock: threading.Lock):
        """
        Start the experiment
        """
        lock.acquire()
        if self.if_initial_window:
            # run
            self.win_set_focus()
            time.sleep(2)
            # keyboard.send_keys('^A')
            # time.sleep(1)
            # mouse.click(coords=(648, 71))   # select the 'graphic view' button
            # time.sleep(3)
            mouse.click(coords=(388,72))   # select the 'run' button
            start_time = datetime.now()
            self.if_initial_window = False
        else:
            self.win_set_focus()
            time.sleep(2)
            mouse.click(coords=(684, 73))  # execute the new experiment
            time.sleep(3)
            if self.mass_win['Data acquisition'].exists():
                mouse.click(coords=(912, 670))
        lock.release()

        return start_time

    def save_path(self, save_path: str):
        """
        Create the folder for saving.
        """
        base_folder = os.path.abspath(save_path)
        folder_path = base_folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return os.path.abspath(folder_path)
        
    def save_file_old(self, lock: threading.Lock, new_name: str, save_path: str):
        """
        Stop the test and save the file.
        """
        lock.acquire()
        # export data
        save = self.save_path(save_path)
        self.win_set_focus()
        time.sleep(2)
        # mouse.click(coords=(766, 70))  # click the 'stop' button
        # time.sleep(1)
        mouse.click(coords=(31, 36))  # click the 'file' button
        time.sleep(1)
        mouse.click(coords=(330, 261))  # click the 'export' button
        time.sleep(1)
        mouse.click(coords=(522, 260))  # click the 'data as' button
        time.sleep(1)
        export_win = self.mass_win.child_window(title="Export View Data To", control_type="Window")
        export_win.child_window(auto_id="41477", control_type="Pane").click_input()
        keyboard.send_keys('^x')     # copy the file name of mass
        keyboard.send_keys(save)     # select the file folder path
        keyboard.send_keys("{ENTER}")
        time.sleep(0.5)
        (export_win.child_window(title="文件名:", auto_id="FileNameControlHost", control_type="ComboBox").
                child_window(title="文件名:", auto_id="1001", control_type="Edit")).set_edit_text(f' _ {new_name}')
        time.sleep(0.5)
        keyboard.send_keys('^v')    # save data as 'file num _ experiment name'
        export_win.child_window(title="保存(S)", auto_id="1", control_type="Button").click()

        lock.release()

    def save_file(self, lock: threading.Lock):
        """
        Stop the test and save the file.
        """
        lock.acquire()
        self.win_set_focus()
        mouse.click(coords=(31, 36))  # click the 'file' button
        time.sleep(1)
        mouse.click(coords=(330, 261))  # click the 'export' button
        time.sleep(1)
        mouse.click(coords=(522, 260))  # click the 'data as' button
        time.sleep(1)
        keyboard.send_keys("{ENTER}")
        time.sleep(0.1)
        keyboard.send_keys("{SPACE}")


        lock.release()
        
    def close(self):
        """
        Close the instrument.
        """
        self.mass.kill()

class MASS:
    def __init__(self, 
                 HOST: str = "192.168.192.28",
                 PORT: int = 5026, 
                 template_folder: str = r"C:\\Users\Admin\Documents\\Hiden Analytical\\MASsoft10\\",
                 template_file: str = "template.exp") -> None:
                 
        """
        Args:
            HOST: the ip address for the socket communication
            PORT: the port for the socket
            template_folder: the folder where the template is 
            template_file: the file name of the template
        """

        # initialize the params
        self.HOST = HOST
        self.PORT = PORT
        self.template_folder = template_folder
        self.template_file = template_file
        # establish the communication
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST,PORT))
        self.s.recv(1024)

    def send_and_receive(self, msg):
        """
        Send msg to the mass software
        """
        self.s.send(msg.encode())
        time.sleep(3)
        data = self.s.recv(1024)
        res_data = data.decode().split()
        print(res_data)
        if int(res_data[0]) == 0:
            raise ValueError("error in socket communication for MS!")
        return res_data
    
    def open_file(self, file_path: str):
        """
        Open a file for the MASoft10 spec
        """
        msg = f"-f{file_path} -d{time_out}"+"\r\n"
        return self.send_and_receive(msg)
    
    def start_mass(self):
        """
        start to run the mass spec file
        """
        msg = "-xGo -Odt -d{time_out}" + "\r\n"
        self.send_and_receive(msg)

    def export_file(self, index):
        """
        Export a file to a csv file
        """

        # get all the csv files before
        self.all_exps_before = self.find_files_ext(self.template_folder)

        # export the csv file
        msg = f"-xExport -v1 -d{time_out}" +"\r\n"
        self.send_and_receive(msg)

        # get the new csv files
        all_exps_aft = self.find_files_ext(self.template_folder)
        # we should only get 1 csv files
        new_files = [i for i in all_exps_aft if i not in self.all_exps_before]
        # wait until the export is finished 
        while len(new_files) == 0:
            all_exps_aft = self.find_files_ext(self.template_folder)
            # we should only get 1 csv file
            new_files = [i for i in all_exps_aft if i not in self.all_exps_before]
            time.sleep(0.1)
        # check the new files that is generated 
        if len(new_files)>1:
            raise ValueError("multiple exp files are created!")
        else:
            self.new_file_name = new_files[0]
        # copy the csv file to the data
        shutil.copy2(self.new_file_name, f"./tmp/mass/{index}.csv") 
        time.sleep(0.5)
        return f"./tmp/mass/{index}.csv"
    
    def close_file(self):
        """
        Close a mass_spec file
        """
        msg = f"-xClose -d{time_out}" +"\r\n"
        return self.send_and_receive(msg)

    def start_mass_spec(self):
        """
        start to run a mass spec file
        """
        # get the path of the template 
        file_path = self.template_folder + self.template_file
        # open the file and start mass 
        self.open_file(file_path)
        self.start_mass()
    
    def find_files_ext(self, folder: str = None, ext: str = "*.csv"):
        """
        Find all the files with certain extensions under a folder
        Args:
            folder: str, the folder location
            ext: str, the extension, e.g, *.csv, *.exp.
        Returns:
            the paths to the files with the specific extension
        """
        
        all_files = [file for path, subdir, files in os.walk(folder) for file in glob.glob(os.path.join(path, ext))]
        
        return all_files
        

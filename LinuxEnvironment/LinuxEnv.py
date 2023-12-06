import time
#from python_vnc_client.vnc import Vnc
from utils.local_vnc import Vnc
import matplotlib.pyplot as plt
import numpy as np 
import cv2

import subprocess

import paramiko




key_dict = {
    "backspace": 0xFF08,
    "tab": 0xFF09,
    "return": 0xFF0D,
    "escape": 0xFF1B,
    "insert": 0xFF63,
    "delete": 0xFFFF,
    "home": 0xFF50,
    "end": 0xFF57,
    "page_up": 0xFF55,
    "page_down": 0xFF56,
    "left": 0xFF51,
    "up": 0xFF52,
    "right": 0xFF53,
    "down": 0xFF54,
    "f1": 0xFFBE,
    "f2": 0xFFBF,
    "f3": 0xFFC0,
    "f4": 0xFFC1,
    "f5": 0xFFC2,
    "f6": 0xFFC3,
    "f7": 0xFFC4,
    "f8": 0xFFC5,
    "f9": 0xFFC6,
    "f10": 0xFFC7,
    "f11": 0xFFC8,
    "f12": 0xFFC9,
    "shift_left": 0xFFE1,
    "shift_right": 0xFFE2,
    "control_left": 0xFFE3,
    "control_right": 0xFFE4,
    "meta_left": 0xFFE7,
    "meta_right": 0xFFE8,
    "alt_left": 0xFFE9,
    "alt_right": 0xFFEA,
}



class VirtualMachineEnv:
    def __init__(self, vm_name, vnc_username, vnc_host, vnc_port, vnc_password):
        self.vm_name = vm_name
        self.vnc_username = vnc_username
        self.vnc_host = vnc_host
        self.vnc_port = vnc_port
        self.vnc_password = vnc_password
        self.vnc = None 

    def _check_vm_status(self):
        """
        Check if the VM is running.
        """
        try:
            result = subprocess.run(['VBoxManage', 'showvminfo', self.vm_name], capture_output=True, text=True)
            if 'running (since' in result.stdout:
                return True
            else:
                return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
        
    def _start_vm(self):
        """
        Check if the VM is already running, if not, start it.
        """
        # check if VM is running 
        if not self._check_vm_status():
            # start VM
            subprocess.run(["VBoxManage", "startvm", self.vm_name, "--type", "headless"], check=True)
            print(f"Starting VM: {self.vm_name}")
            time.sleep(10)

            input('manually start vnc server on VM') # TODO automate that 


    def _stop_vm(self):
        """
        Check if the VM is running, if so, stop it.
        """
        # check if VM is running 
        if self._check_vm_status():
            # stop VM
            subprocess.run(["VBoxManage", "controlvm", self.vm_name, "poweroff"], check=True)
            print(f"Stopping VM: {self.vm_name}")
            time.sleep(5)




    def reset(self):
        # start VM is necessary
        self._start_vm()

        # establish vnc connection 
        if self.vnc is None:
            print('Establishing VNC connection')
            self.vnc = Vnc(self.vnc_host, self.vnc_port, self.vnc_password)
            self.vnc.connect()
        else:
            print('VNC connection already established')

        # capture screen
        print('Capturing screen')
        image = self.vnc.capture_screen(True)
        # convert to array
        self.observation = np.array(image)
        #input(self.observation)
        #plt.imshow(self.observation)
        #plt.show()
        return self.observation


    def close(self):
        self.vnc.close()
        self.vnc = None 
    
    def step(self, action):
        # send action to vnc, get screen capture, return 
        # action format: (type, key/cursor)
        if action[0] == "key":
            key_action = action[1]
            if key_action in key_dict:
                key_action = key_dict[key_action]
            self.vnc.key_down(key_action)
            time.sleep(0.1)
            self.vnc.key_up(key_action)

        elif action[0] == "mouse":
            # format ('mouse', (x, y), 'left')
            self.vnc.mouse_event(
                event=action[2],
                position=action[1],
            )
        else:
            print(f'Invalid action type: ({action[0]})')
            #raise Exception("Invalid action type")
        
        # capture screen
        print('Capturing screen')
        image = self.vnc.capture_screen(True)
        # convert to array
        self.observation = np.array(image)
        return self.observation, 0, False, None
    

    def render(self):
        # Render a downsampled version of the current observation.
        cv2.imshow(
            "Virtual Desktop",
            cv2.resize(self.observation.astype(np.uint8), (960, 600), interpolation=cv2.INTER_AREA),
        )
        cv2.waitKey(1)
        time.sleep(0.1)





if __name__ == "__main__":
    """
    Test that everything is working using random actions.
    """
    # initialize the environment
    env = VirtualMachineEnv(
        vm_name="server2",
        vnc_username="leon",
        vnc_host="127.0.0.1",
        vnc_port=5999,
        vnc_password="password",
    )

    # reset the environment 
    obs = env.reset()

    for _ in range(250):
        # get a random mouse movement action
        # format ('mouse', (x, y), 'left')
        rndm_x = np.random.randint(0, 1920)
        rndm_y = np.random.randint(0, 1200)
        action = ("mouse", (rndm_x, rndm_y), "Left")

        # take a step in the environment
        obs, _, _, _ = env.step(action)

        env.render()

        

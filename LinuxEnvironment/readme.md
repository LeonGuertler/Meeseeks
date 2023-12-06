# The Environment
This simplistic desktop environment can run headlessly in the background and allows the user to interact with it via a openAI gym style object. The observations are screen captures and the actions are mouse and keyboard inputs.

Keep in mind that this is not the final version of the environment, and bugs are still prevelant.



## TODO
1. Add a script to automatically set up the ubuntu VM (Essentially right now I am simply running a headless ubuntu server with an install display manager (slim) and debian desktop (installed via tasksel). All relevant ports are forwarded (specifically 2522 -> 22 and 5999 -> 5901))
2. Add the missing keys/actions to the environment
3. make the environment more robust
4. Add a proper reset functions that deletes all created files etc
5. Add specific tasks and scp command to check task success
6. One of the tasks should be for the machine to download itself onto the new server and give itself some task



## Set up VM
I'll make this much easier in the future, but for now, you can simply use VirtualBox to set up an ubuntu server, log into the server and install tasksel (sudo apt install tasksel), install slim (sudo apt install slim), use tasksel to install gnome desktop (sudo tasksel), install tigervnc (sudo install tigervnc-standalone-server), and finally, run tigervnc (vncserver -localhost no)
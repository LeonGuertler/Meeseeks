import numpy as np 
import matplotlib.pyplot as plt 

import os, random, cv2, time 



class ActionEnv:
    """
    The basic outline of this environment is as follows:
    - The Scenario Generator generates a randomly colored canvas of given size
        (the size being similar to that of a screen). Subsequently it places a 
        random number of objects at random locations on the canvas.
    - One of the objects is the target object for being clicked. The coordinates
        of that object and the instruction to click it are returned.
    - On top of the canvas, a mouse is simulated using a png and x/y coordinates.
        This simulated mouse is used to determine task success
    """

    action_space = {
        "move_left": 0,
        "move_right": 1,
        "move_up": 2,
        "move_down": 3,
        "left_click": 4,
        "right_click": 5
    }
    
    def __init__(self):
        # initializet the scenario generator
        self.scenario_generator = ScenarioGenerator()

        # set the max allowed steps
        self.max_steps = 1000

    def reset(self):
        """
        Reset the environment. This will generate a new canvas with objects,
        and re-initialize a mouse at a random location.
        """
        # reset the step counter 
        self.step_counter = 0

        # generate a new scenario
        self.canvas, self.target_bounding_box, self.task_description, self.target_click_type = self.scenario_generator.generate_scenario()

        # for ease of calculation, extract the center point of the target bounding box
        self.target_center = np.mean(self.target_bounding_box, axis=0)

        # initialize the mouse cursor
        self.mouse = Mouse(
            canvas_shape=self.canvas.shape
        )

        # insert the mouse cursor into the canvas at a random position 
        self.mouse.get_random_position()
        self.observation = self.mouse.get_observation(self.canvas)

        # calculate the target distance for future reward calculation
        self.target_distance = self._get_distance()

        # return the observation & task
        return self.observation, self.task_description
    

    def step(self, action):
        """
        Given an action, move the mouse accordingly and return the new observation, reward and done flag.
        """
        # check if mouse moves
        if action in [0, 1, 2, 3]:
            # move the mouse 
            self.mouse.move_mouse(action)
            done = False
        
        else:
            # click the mouse. Clicking finishes the episode
            self.mouse.click_mouse(action)
            done = True

        # increment step count 
        self.step_counter += 1

        # check if step count is reached
        if self.step_counter >= self.max_steps:
            done = True

        # determine the rewards
        reward = self._get_reward()
        info = self._evaluate() if done else None

        # get the new observation
        self.observation = self.mouse.get_observation(self.canvas)

        return self.observation, reward, done, info
    
    def render(self):
        """
        Use cv2 to render the canvas.
        """
        cv2.imshow("ActionEnv", self.observation.transpose(1, 2, 0))
        cv2.waitKey(1)
        time.sleep(0.1)


    def _evaluate(self):
        """
        In addition to the intermediate rewards, this provides a binary reward at the end of the episode.
        It checks whether the mouse is on the bounding box and the correct button was clicked.
        """
        # first check if the correct button is pressed
        mouse_left, mouse_right = self.mouse.get_clicks()
        if (self.target_click_type == 'left' and mouse_left) or (self.target_click_type == 'right' and mouse_right):
            # the correct button is pressed
            # now check if the mouse is on the bounding box
            mouse_x, mouse_y = self.mouse.get_position()
            if mouse_x > self.target_bounding_box[0, 1] and mouse_x < self.target_bounding_box[1, 1] and mouse_y > self.target_bounding_box[0, 0] and mouse_y < self.target_bounding_box[2, 0]:
                return 1, self._get_distance()

        # if this code block is reached, the wrong button was pressed or the mouse was not on the bounding box
        return 0, self._get_distance()


    def _get_distance(self):
        """
        Get the current distance between the mouse and the center of the target icon.
        """
        mouse_x, mouse_y = self.mouse.get_position()
        return np.sqrt((mouse_x - self.target_center[1])**2 + (mouse_y - self.target_center[0])**2)
        

    def _get_reward(self):
        """
        The current reward depends on an improved distance to the target icon.
        """
        new_target_distance = self._get_distance()
        reward = self.target_distance - new_target_distance
        self.target_distance = new_target_distance
        return reward





class Mouse:
    def __init__(self, canvas_shape):
        self.x = None 
        self.y = None 
        self.left_click = False
        self.right_click = False


        self.file_path = os.path.join("data", "mouse-cursor.png")

        # load the mouse cursor as a numpy array and initialize a boolean mask of the same shape
        self.cursor = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
        self.cursor = cv2.resize(self.cursor, dsize=(12, 20), interpolation=cv2.INTER_CUBIC)
        # resize to 16x16
        # extract the alpha channel as a binary mask
        self.cursor_mask = self.cursor[:, :, 3] > 0
        # add new axis at the end
        self.cursor_mask = self.cursor_mask[:, :, np.newaxis]
        # remove the alpha channel from the cursor
        self.cursor = self.cursor[:, :, :3]

        # reshape both to channel first
        self.cursor = self.cursor.transpose(2, 0, 1)
        self.cursor_mask = self.cursor_mask.transpose(2, 0, 1)

        self.canvas_shape = canvas_shape[1:]

    def get_random_position(self):
        self.x = np.random.randint(0, self.canvas_shape[1] - self.cursor.shape[2])
        self.y = np.random.randint(0, self.canvas_shape[0] - self.cursor.shape[1])


    def move_mouse(self, action):
        if action == 0:
            # move left
            self.x = max(0, self.x - 10)
        elif action == 1:
            # move right
            self.x = min(self.canvas_shape[1] - self.cursor.shape[2], self.x + 10)
        elif action == 2:
            # move up
            self.y = max(0, self.y - 10)
        elif action == 3:
            # move down
            self.y = min(self.canvas_shape[0] - self.cursor.shape[1], self.y + 10)


    def click_mouse(self, action):
        if action == 4:
            self.left_click = True
        elif action == 5:
            self.right_click = True


    def get_position(self):
        return self.x, self.y

    def get_clicks(self):
        return self.left_click, self.right_click


    def get_observation(self, canvas):
        local_canvas = canvas.copy()
        local_canvas[:, self.y:self.y+self.cursor.shape[1], self.x:self.x+self.cursor.shape[2]] = self.cursor*self.cursor_mask + local_canvas[:, self.y:self.y+self.cursor.shape[1], self.x:self.x+self.cursor.shape[2]] * (~self.cursor_mask)
        return local_canvas




class Icons:
    def __init__(self):
        dataset_path = os.path.join(
            "data", "Icons-50"
        )
        # load the dataset labels with numpy 
        icons = np.load(os.path.join(dataset_path, "Icons-50.npy"), allow_pickle=True).item()

        # create dict of dict with idx as key and class and image as values
        self.icons = {
            str(idx): {"class": class_, "image": image_} for idx, (class_, image_) in enumerate(zip(icons['subtype'], icons['image']))
        }


    def get_rndm_icon(self):
        rndm_idx = np.random.randint(0, len(self.icons))
        return self.icons[str(rndm_idx)]

class Backgrounds:
    """
    The purpose of this class is to generate random monochrome backgrounds of a specified size
    """
    def __init__(self):
        # create a list of random colors represented in int8 RGB
        self.colours = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    def generate_background(self, size=(3, 768, 1366)):
        return np.ones(size, dtype=np.uint8) * self.colours[np.random.randint(0, 1000)].reshape((3, 1, 1))



class ScenarioGenerator:
    def __init__(self):
        self.icons = Icons()
        self.backgrounds = Backgrounds()

    def generate_scenario(self, num_background_icons=None):
        # determine number of noise items
        if num_background_icons is None:
            num_background_icons = np.random.randint(1, 15)

        # generate background
        canvas = self.backgrounds.generate_background()

        # place the background icons randomly on the canvas
        for _ in range(num_background_icons):
            icon = self.icons.get_rndm_icon()['image']
            canvas, _ = self._place_icon_on_canvas(canvas, icon)

        # place the target icon randomly on the canvas
        target_icon = self.icons.get_rndm_icon()
        canvas, target_bounding_box = self._place_icon_on_canvas(canvas, target_icon['image'])



        # convert the target class into a natural language task description
        task_description, click_type = self._generate_task_description(target_icon['class'])


        if False:
            print(target_bounding_box)
            print(task_description)
            plt.imshow(canvas.transpose(1, 2, 0))
            plt.show()

        return canvas, target_bounding_box, task_description, click_type


    def _place_icon_on_canvas(self, canvas, icon):
        # get icon size
        icon_size = icon.shape[1:]

        # get canvas size
        canvas_size = canvas.shape[1:]

        # get random position on canvas
        x = np.random.randint(0, canvas_size[0] - icon_size[0])
        y = np.random.randint(0, canvas_size[1] - icon_size[1])

        # place icon on canvas
        canvas[:, x:x+icon_size[0], y:y+icon_size[1]] = icon

        icon_bounding_box = np.array([
            [x, y],
            [x+icon_size[0], y],
            [x+icon_size[0], y+icon_size[1]],
            [x, y+icon_size[1]]
        ])

        return canvas, icon_bounding_box
    
    def _generate_task_description(self, target_class):
        # Base phrases that can be used to construct the instruction
        base_phrases = [
            ("left click on the {icon}", 'left'),
            ("right click on the {icon}", 'right'),
            ("Please select the {icon} by clicking it", 'left'),
            ("Go ahead and left click on the {icon}", 'left'),
            ("Go ahead and right click on the {icon}", 'right'),
            ("Use the mouse to select the {icon}", 'left'),
            ("Find and left click on the {icon}", 'left'),
            ("Find and right click on the {icon}", 'right'),
            ("Choose the {icon} with a right click", 'right'),
            ("Interact with the {icon} by left clicking it", 'left'),
            ("Interact with the {icon} by right clicking it", 'right'),
            ("Point and left click on the {icon}", 'left'), 
            ("Point and right click on the {icon}", 'right'),
            ("Activate the {icon} by left clicking", 'left'),
            ("Direct your left click towards the {icon}", 'left'),
            ("Direct your right click towards the {icon}", 'right'),
            ("Move the mouse to {icon} and left click", 'left'),
            ("Move the mouse to {icon} and right click", 'right'),
            ("Click on the {icon} to continue", 'left'),
            ("Click on the {icon} to proceed", 'left'),
            ('Select the {icon} to continue', 'left'),
        ]

        # generate the instruction
        instruction_base, click_type = random.choice(base_phrases)
        return instruction_base.format(
            icon=target_class.lower().replace("_", " ")
        ), click_type


# debug
if __name__ == "__main__":
    env = ActionEnv()
    env.reset()

    # test the env with random actions
    action_list = [0, 1, 2, 3] #, 4, 5]

    done = False

    while not done:
        action = np.random.choice(action_list)
        observation, reward, done, episode_reward = env.step(action)
        env.render()
        print(reward, done, episode_reward)

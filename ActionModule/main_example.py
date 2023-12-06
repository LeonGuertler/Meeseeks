from example import models, action_environment

# load env 
env = action_environment.ActionEnv()

# load agent
agent = models.Agent()


# train the agent
agent.train(env)

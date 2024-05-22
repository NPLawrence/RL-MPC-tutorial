"""
A Gym wrapper that serves as the bridge between SB3 and do-mpc
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import do_mpc

class DoMPCEnv(gym.Env):
    """
    Gym environment that uses do-mpc for carrying out simulations
    """
    def __init__(self, simulator:do_mpc.simulator.Simulator, 
                 bounds:dict={'x_high' : np.array([10.0, 10.0]), 'x_low' : np.array([-10.0, -10.0]),
                                'u_high' : np.array([10.0]), 'u_low' : np.array([-10.0])}, 
                    num_steps=100,
                    clip_reset=None,
                    same_state=False,
                    reward=lambda x,a:-np.linalg.norm(x)**2):        
        super().__init__()

        self.simulator = simulator
        self.num_steps = num_steps
        self.clip_reset = clip_reset
        self.same_state = same_state

        self.action_space = spaces.Box(low=bounds['u_low'], high=bounds['u_high'], dtype=np.float32)
        self.observation_space = spaces.Box(low=bounds['x_low'],high=bounds['x_high'], dtype=np.float32)

        self.reward_fun = reward

    def step(self, action):

        self.action = action
        self.t += 1
        self.state = self._simulator(action)
        # do_mpc.tools.printProgressBar(self.t, self.num_steps, prefix='Closed-loop simulation:', length=50)

        info = self._get_info()
        reward, terminated, truncated = info["reward"], info["distance"]<0.01, self.t == self.num_steps

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = self._reset_state()
        self.action = self.action_space.sample()
        self.simulator.x0 = x
        self.state = self._fix_state(x)
        self.t = 0
        info = self._get_info()

        return self.state, info

    def render(self):

        return do_mpc.graphics.default_plot(self.simulator.data)
        
    def close(self):
        ...

    def _reset_state(self):

        if self.same_state:
            state = np.ones(self.observation_space.shape, dtype=self.observation_space.dtype)
            return state / np.linalg.norm(state)
        elif self.clip_reset is not None:
            state = np.clip(self.observation_space.sample(), -self.clip_reset, self.clip_reset)
            return state / np.linalg.norm(state)    
        else:
            return self.observation_space.sample()

        # return self.observation_space.sample()
    
    def _get_info(self):

        reward = self.reward_fun(self.state, self.action)

        return {"time": self.t, "distance": np.linalg.norm(self.state).item(), "reward": reward}
    
    def _simulator(self, action):
        # functionally the same as simulator.make_step() but processes the output and dtype
        a = self._fix_action(action)
        x = self.simulator.make_step(a)
        return self._fix_state(x)
    
    def _fix_state(self, x):
        return np.float32(np.reshape(x, self.observation_space.shape))

    def _fix_action(self, action):
        a = np.reshape(action, self.action_space.shape[::-1])

        return a.reshape(a.shape + (1,))
    

'''if __name__ == '__main__':
    print("Checking batch_reactor")
    # (Tip: to pass the checker it can be useful loosen the constraints since it just uses random actions
    #    and can easily take the system out of the desired constraints)
    from systems.examples.batch_reactor.template_model import template_model
    from systems.examples.batch_reactor.template_mpc import template_mpc
    from systems.examples.batch_reactor.template_simulator import template_simulator

    from stable_baselines3 import PPO
    from stable_baselines3.common import results_plotter
    from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
    from stable_baselines3.common.monitor import Monitor
    import matplotlib.pyplot as plt

    model = template_model()
    mpc = template_mpc(model)
    simulator = template_simulator(model)
    max_x = np.array([[3.7], [3.0], [3.0], [3.0]]).flatten() # writing like this to emphasize do-mpc sizing convention
    min_x = np.array([[0.0], [-0.01], [0.0], [0.0]]).flatten()
    bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : np.array([0.0]), 'u_high' : np.array([0.2])}
    env = DoMPCEnv(simulator, bounds=bounds)
    # check_env(env)

    log_dir = "../models/"
    timesteps = 1e5
    env = Monitor(env, log_dir)
    RLmodel = PPO("MlpPolicy", env, verbose=1)
    RLmodel.learn(total_timesteps=timesteps)
    # %%
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO oscillating_masses_discrete")
    plt.show()'''



if __name__ == '__main__':
    print("Checking oscillating_masses_discrete")
    from systems.examples.oscillating_masses_discrete.template_model import template_model
    from systems.examples.oscillating_masses_discrete.template_mpc import template_mpc
    from systems.examples.oscillating_masses_discrete.template_simulator import template_simulator

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common import results_plotter
    from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
    from stable_baselines3.common.monitor import Monitor
    import matplotlib.pyplot as plt
    import torch

    model = template_model()
    mpc = template_mpc(model)
    simulator = template_simulator(model)
    max_x = np.array([4.0, 10.0, 4.0, 10.0], dtype=np.float32)
    min_x = -max_x
    bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : np.array([-0.5]), 'u_high' : np.array([0.5])}
    def reward(x,a):
        return -np.linalg.norm(x[0])**2
    env = DoMPCEnv(simulator, bounds=bounds, num_steps=200, reward=reward)
    check_env(env)

    # log_dir = "../models/"
    # timesteps = 5e4
    # env = Monitor(env, log_dir)
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                  net_arch=[64, 64])
    # RLmodel = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    # RLmodel.learn(total_timesteps=timesteps)
    # plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "SAC oscillating_masses_discrete")
    # plt.show()


"""
A Gym wrapper that serves as the bridge between SB3 and do-mpc
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import do_mpc

class DoMPCEnvGoal(gym.Env):
    """
    Gym environment that uses do-mpc for carrying out simulations
    """
    def __init__(self, simulator:do_mpc.simulator.Simulator, 
                 bounds:dict={'x_high' : np.array([10.0, 10.0]), 'x_low' : np.array([-10.0, -10.0]),
                                'u_high' : np.array([10.0]), 'u_low' : np.array([-10.0])}, 
                    num_steps=100,
                    clip_reset=None,
                    same_state=False,
                    tol=0.1,
                    render_mode=None):        
        super().__init__()

        self.simulator = simulator
        self.num_steps = num_steps
        self.clip_reset = clip_reset
        self.same_state = same_state
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=bounds['u_low'], high=bounds['u_high'], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation" : spaces.Box(low=bounds['x_low'],high=bounds['x_high'], dtype=np.float32),
            "desired_goal" : spaces.Box(low=bounds['x_low'],high=bounds['x_high'], dtype=np.float32),
            "achieved_goal" : spaces.Box(low=bounds['x_low'],high=bounds['x_high'], dtype=np.float32),
        })

        self.goal = np.zeros(self.observation_space["desired_goal"].shape, dtype=self.observation_space["desired_goal"].dtype)
        self.tol = tol

    def step(self, action):

        self.action = action
        self.t += 1
        self.state = self._simulator(action)
        # do_mpc.tools.printProgressBar(self.t, self.num_steps, prefix='Closed-loop simulation:', length=50)
    
        obs = self._state_to_dict()

        info = self._get_info()
        reward, terminated, truncated = info["reward"], info["distance"]<0.01, info["TimeLimit.truncated"]

        return obs, reward, terminated, truncated, info
    

    def compute_reward(self, achieved_goal, desired_goal, info):
        axis = 0
        if achieved_goal.ndim == 2:
            axis = 1
        return -1*(np.linalg.norm(achieved_goal - desired_goal, np.inf, axis=axis) > self.tol)
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = self._reset_state()
        self.action = self.action_space.sample()
        self.simulator.x0 = x
        self.state = self._fix_state(x)
        obs = self._state_to_dict()
        self.t = 0
        info = self._get_info()

        return obs, info


    def _state_to_dict(self):

        return {"observation" : self.state,
         "desired_goal" : self.goal,
         "achieved_goal" : self.state}
        
    def _reset_state(self):

        if self.same_state:
            state = np.ones(self.observation_space["observation"].shape, dtype=self.observation_space.observation_space["observation"].dtype)
            return state / np.linalg.norm(state)
        elif self.clip_reset is not None:
            state = np.clip(self.observation_space["observation"].sample(), -self.clip_reset, self.clip_reset)
            # return state
            return state / np.linalg.norm(state)    
        else:
            return self.observation_space["observation"].sample()

    
    def _get_info(self):

        # reward = self.reward_fun(self.state, self.action)
        reward = self.compute_reward(self.state, self.goal, {})

        return {"time": self.t, "distance": np.linalg.norm(self.state - self.goal).item(), "reward": reward, "TimeLimit.truncated": self.t == self.num_steps}
    
    def _simulator(self, action):
        # functionally the same as simulator.make_step() but processes the output and dtype
        a = self._fix_action(action)
        x = self.simulator.make_step(a)
        return self._fix_state(x)
    
    def _fix_state(self, x):
        return np.float32(np.reshape(x, self.observation_space["observation"].shape))

    def _fix_action(self, action):
        a = np.reshape(action, self.action_space.shape[::-1])

        return a.reshape(a.shape + (1,))

    def renderMPC(self, mpc, num_steps, path=""):

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
        sim_graphics = do_mpc.graphics.Graphics(self.simulator.data)

        self._run_closed_loop(mpc, self.simulator, num_steps=num_steps)

        fig = do_mpc.graphics.default_plot(self.simulator.data)
        plt.savefig(path)

        return fig[0]


    def _run_closed_loop(self, controller, simulator, num_steps=10, x0=None):

        controller.reset_history()
        simulator.reset_history()
        self.reset()
        x0 = self.state.copy()

        controller.x0 = x0
        simulator.x0 = x0
        controller.set_initial_guess()

        for _ in range(num_steps):
            u0 = controller.make_step(x0)
            x0 = simulator.make_step(u0)
            # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

        return
        
    def close(self):
        ...



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

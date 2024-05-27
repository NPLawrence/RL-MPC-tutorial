"""
Use a standard MPC module with a learnable value function added as a terminal cost.
Includes hindsight relabeling.
Uses do-mpc to optimize in the action variable during rollouts, hence improving the policy.
Keeps the policy network intact but only for training the critic, not interacting with the environment.
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.wrappers import TransformObservation # for augmenting the goal state
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.vec_env import DummyVecEnv

from torch.utils.tensorboard import SummaryWriter


# 
from envs.DoMPCEnvGoal import DoMPCEnvGoal
from casadi import *
import do_mpc
import onnx
from do_mpc.data import save_results, load_results
import matplotlib.pyplot as plt


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "testing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "oscillating_masses"
    """the environment id of the task"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    buffer_size: int = int(5e3)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    n_sampled_goal: int = 4
    """use hindsight experience relabeling. set to 0 to get baseline performance (no HER)."""
    goal_selection_strategy: str = "future"
    """"HER goal selection strategy: "episode", "final", or "future" """
    reward_fn: str = "default"
    """specifies which reward function to use"""
    noise_clip: float = 0.1
    """noise clip parameter of the MPC exploration"""
    is_baseline: bool = False
    """run experiment with just MPC, no added value function"""
    process_noise: float = 0.01
    """how much noise to add to prediction model parameters"""
    n_horizon: int = 1
    """prediction horizon for MPC module"""
    linear_solver: str = "MA27"
    """linear solver for IPOPT"""



def make_env(env_id, seed, idx, capture_video, run_name):
    if env_id == "oscillating_masses":
        from envs.systems.examples.oscillating_masses_discrete.template_model import template_model
        from envs.systems.examples.oscillating_masses_discrete.template_mpc import template_mpc
        from envs.systems.examples.oscillating_masses_discrete.template_simulator import template_simulator

    def thunk():
        model = template_model()
        mpc = template_mpc(model)
        simulator = template_simulator(model)
        max_x = np.array([4.0, 10.0, 4.0, 10.0], dtype=np.float32)
        min_x = -max_x
        bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : np.array([-0.5], dtype=np.float32), 'u_high' : np.array([0.5], dtype=np.float32)}

        env = DoMPCEnvGoal(simulator, bounds=bounds, num_steps=60, tol=0.1, clip_reset=4.0, same_state=False)        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env
    
    return thunk
    
def make_model(env_id, A=None):

    if env_id == "oscillating_masses":
        from envs.systems.examples.oscillating_masses_discrete.template_model import template_model
    
    model = template_model(A = A)
    return model


def make_mpc(model, value_onnx, x0 = None, u0 = None, n_horizon = 1, silence_solver = False, is_baseline = False, solver="MA57", gamma = 1.0):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = n_horizon
    mpc.settings.t_step = 0.5
    mpc.settings.store_full_solution = False

    if silence_solver:
        mpc.settings.supress_ipopt_output()
    # see https://coin-or.github.io/Ipopt/OPTIONS.html and https://arxiv.org/pdf/1909.08104
    mpc.nlpsol_opts = {'ipopt.linear_solver': solver} # MA57, MA27, spral, HSL_MA86, mumps

    lterm = model.aux['cost'] 

    if is_baseline:
        mterm = model.aux['cost']
    else:
        terminal_converter = do_mpc.sysid.ONNXConversion(value_onnx)
        def terminal_casadi(x):
            terminal_converter.convert(x = x.T, goal=np.zeros(x.T.shape))
            return terminal_converter['terminal_cost']
        mterm = -terminal_casadi(model.x['x'])
        # mterm=SX.zeros(1,1)

    mpc.set_objective(lterm=lterm, mterm=mterm)
    
    mpc.set_rterm(u=1e-2)

    max_x = np.array([[4.0], [10.0], [4.0], [10.0]])

    mpc.bounds['lower','_x','x'] = -max_x
    mpc.bounds['upper','_x','x'] =  max_x

    mpc.bounds['lower','_u','u'] = -0.5
    mpc.bounds['upper','_u','u'] =  0.5
    if x0 is not None:
        mpc.x0 = x0
    if u0 is not None:
        mpc.u0 = u0

    mpc.setup()

    return mpc

def convert_value(cost, env):

    obs_size = sum([get_flattened_obs_dim(env.observation_space["observation"])])
    goal_size = sum([get_flattened_obs_dim(env.observation_space["desired_goal"])])

    x = torch.rand(obs_size).unsqueeze(0)
    goal = torch.rand(goal_size).unsqueeze(0)

    torch.onnx.export(cost, (x,goal), "./models/cost.onnx",export_params=True,input_names=["x","goal"], output_names=["terminal_cost"])
    value_onnx = onnx.load("./models/cost.onnx")

    # onnx.checker.check_model(critic_onnx)

    return value_onnx


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        if isinstance(env.observation_space, gym.spaces.Dict):
            input_size = sum([get_flattened_obs_dim(env.observation_space["observation"]), get_flattened_obs_dim(env.observation_space["desired_goal"]), np.prod(env.action_space.shape)])
        else:
            input_size = np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape)

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        # self.actor = Actor(env)

    def forward(self, x, a):
        x = self._combine(x)
        x = self.q_value(x, a)
        return x
    
    def q_value(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    def _combine(self, x):

        x = torch.cat([x["observation"], x["desired_goal"]], 1)

        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        if isinstance(env.observation_space, gym.spaces.Dict):
            input_size = sum([get_flattened_obs_dim(env.observation_space["observation"]), get_flattened_obs_dim(env.observation_space["desired_goal"])])
        else:
            input_size = np.array(env.observation_space.shape).prod()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_mean = nn.Linear(32, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(32, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )


    def forward(self, x):
        x = self._combine(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def _combine(self, x):

        x = torch.cat([x["observation"], x["desired_goal"]], 1)

        return x


    def _explore_noise(self, x):

        x = {"observation": torch.Tensor(x["observation"]), "desired_goal": torch.Tensor(x["desired_goal"])}
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean.zero_(), std)
        noise = normal.rsample() 

        return noise

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_deterministic_action(self, x):
        # also assume x is combined (for passing to do-mpc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean


class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.qf = SoftQNetwork(env).to(device)
        self.actor = Actor(env).to(device)
        self.goal = torch.zeros(env.observation_space["desired_goal"].shape).unsqueeze(0)

    def forward(self, x, goal):
        x = torch.cat([x, goal], 1)
        a = self.actor.get_deterministic_action(x)
        return self.qf.q_value(x,a)



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    envs = DummyVecEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.action_space.high[0])

    A = [[ 0.763,  0.460,  0.115,  0.020],
                  [-0.899,  0.763,  0.420,  0.115],
                  [ 0.115,  0.020,  0.763,  0.460],
                  [ 0.420,  0.115, -0.899,  0.763]]

    A += args.process_noise*np.random.randn(4,4) # artificial mismatch
    A = A.astype('f')

    B = [[0.014],
                  [0.063],
                  [0.221],
                  [0.367]]

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    vf = ValueNetwork(envs).to(device)
    vf.qf.load_state_dict(qf1.state_dict())
    vf.actor.load_state_dict(actor.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # env.observation_space.dtype = np.float32

    ## set n_sampled_goal=0 to get baseline performance (no HER)
    rb = HerReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        envs,
        device,
        n_sampled_goal=args.n_sampled_goal,
        goal_selection_strategy=args.goal_selection_strategy
    )

    ## INITIALIZING mpc
    value_onnx = convert_value(vf, envs)
    model = make_model(args.env_id, A = A)
    estimator = do_mpc.estimator.StateFeedback(model)
    mpc = make_mpc(model, value_onnx, n_horizon=args.n_horizon, silence_solver=True, is_baseline=args.is_baseline, solver=args.linear_solver, gamma=args.gamma)
    mpc.set_initial_guess()    

    # TRY NOT TO MODIFY: start the game
    envs.seed(seed=args.seed)
    obs = envs.reset()
    ep_number = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        ## IMPLEMENT mpc actions around here

        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])

        else:

            ## ~super inefficient~ MPC stuff
            x0 = estimator.make_step(obs["observation"])
            actions = mpc.make_step(x0)

            # A quick way of incorporating the SAC-based exploration
            #   (doesn't make a huge difference, at least for this simple example)
            with torch.no_grad():
                if global_step % args.policy_frequency == 0: # optional: add some delay between disturbances (exploration)
                    noise = actor._explore_noise(obs).numpy()
                    noise = np.float32(noise.clip(-args.noise_clip, args.noise_clip))
                    actions += noise
                    actions = np.float32(actions.clip(envs.action_space.low, envs.action_space.high))


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        if global_step == args.learning_starts:
            start_time = time.time() ## moving this here to get a more accurate read of the mpc speed
            dones = np.ones(dones.shape) ## HER requires a completed episode before training starts, so this is just a one-sample episode

        if dones: 
            ep_number += 1

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos[0]:
            for info in infos:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("misc/episode_number", ep_number)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm)
            q_optimizer.step()
            
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 200 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int((global_step - args.learning_starts) / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int((global_step - args.learning_starts) / (time.time() - start_time)), global_step - args.learning_starts)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        ## UPDATE mpc
        if dones: # update MPC at end of episode

            # plot a rollout
            if ep_number % 15 == 0:
                episode_path =  f"runs/{run_name}/fig-episode-" + f"{ep_number}.pdf"
                ep_data = envs.env_method("renderMPC", mpc, 60, path=episode_path,indices=[0])
                if args.track:
                    wandb.log({"figs": wandb.Image(ep_data[0])})
                envs.reset()

            if not args.is_baseline:
                for param, value_param in zip(qf1.parameters(), vf.qf.parameters()):
                    value_param.data.copy_(param.data)
                for param, value_param in zip(actor.parameters(), vf.actor.parameters()):
                    value_param.data.copy_(param.data)

                value_onnx = convert_value(vf, envs)
                estimator = do_mpc.estimator.StateFeedback(model)
                mpc = make_mpc(model, value_onnx, n_horizon=args.n_horizon, silence_solver=True, is_baseline=args.is_baseline, solver=args.linear_solver, gamma=args.gamma)
                mpc.set_initial_guess()
                # estimator.set_initial_guess()


    episode_path =  f"runs/{run_name}/fig-final.pdf"
    ep_data = envs.env_method("renderMPC", mpc, 60, path=episode_path,indices=[0])
    if args.track:
        wandb.log({"figs": wandb.Image(ep_data[0])})
        wandb.finish()
    # plt.show()
    envs.close()
    writer.close()
    


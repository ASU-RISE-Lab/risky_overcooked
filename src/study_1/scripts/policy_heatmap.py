import numpy as np
import torch
import random
from itertools import count
import matplotlib.pyplot as plt

from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_py.mdp.actions import Action
from risky_overcooked_rl.utils.visualization import TrajectoryHeatmap

import study_1 as Algorithm
import pickle


class PolicyHeatmap():
    def __init__(self,layout,p_slip,human_type, robot_type = 'Oracle',
                 n_trials=1,rationality=10,horizon=400,time_cost=0.0,
                 overwrite_dict=None):
        self.layout = layout
        self.p_slip = p_slip
        self.n_trials = n_trials
        self.horizon = horizon

        # Parse Config ---------------------------------------------------------
        config = Algorithm.get_default_config()
        config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
        config['env']["LAYOUT"] = layout
        config['env']['p_slip'] = p_slip
        config['env']['HORIZON'] = horizon
        config['env']['time_cost'] = time_cost
        config['agents']['model']["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

        if overwrite_dict is not None:
            for key, val in overwrite_dict.items():
                Algorithm.set_config_value(config, key, val)


        # set up env ---------------------------------------------------------
        overwrite = {}
        if p_slip != 'default' and p_slip != 'def':
            overwrite['p_slip'] = p_slip
        overwrite['neglect_boarders'] = config['env']['neglect_boarders']
        mdp = OvercookedGridworld.from_layout_name(layout, **overwrite)
        # mdp = OvercookedGridworld.from_layout_name(config['env']['LAYOUT'])
        # mdp.p_slip = config['env']['p_slip']
        obs_shape = mdp.get_lossless_encoding_vector_shape()
        n_actions = 36
        self.env = OvercookedEnv.from_mdp(mdp, horizon=config['env']['HORIZON'], time_cost=config['env']['time_cost'])

        # load policies ---------------------------------------------------------

        # if policy_type.lower() == 'Rational'.lower():
        #     policy_fname =   f'{layout}_pslip0{int(p_slip * 10)}__rational'
        # elif policy_type.lower() == 'Averse'.lower():
        #     policy_fname = f'{layout}_pslip0{int(p_slip * 10)}__b00_lam225_etap088_etan10_deltap061_deltan069'
        # else: raise ValueError(f'Unknown Policy Type {policy_type}')
        #
        # self.policy = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, policy_fname)
        # self.policy.rationality = rationality
        # self.policy.model.eval()
        self.human_type = human_type
        self.robot_type = robot_type

        # policy_fnames = {
        #     'Averse': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__b00_lam225_etap088_etan10_deltap061_deltan069',
        #     'Rational': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__rational',
        #     'Seeking': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__b00_lam044_etap10_etan088_deltap061_deltan069'
        # }
        policy_fnames = {
            'Averse': f'{layout}_pslip0{int(p_slip * 10)}__b00_lam225_etap088_etan10_deltap061_deltan069',
            'Rational': f'{layout}_pslip0{int(p_slip * 10)}__rational',
            'Seeking': f'{layout}_pslip0{int(p_slip * 10)}__b00_lam044_etap10_etan088_deltap061_deltan069'
        }
        self.policy_fnames = policy_fnames

        config['agents']['save_dir'] = config['save']['save_dir']

        loaded_policies = []
        if self.robot_type=="Oracle":
            loaded_policies =[self.human_type]
        elif self.robot_type == "RS-ToM":
            loaded_policies = list(self.policy_fnames.keys())
        elif self.robot_type in self.policy_fnames.keys():
            loaded_policies = [self.robot_type]
        else:
            raise ValueError(f'Invalid robot condition: {self.robot_type}')

        self.policies = {}
        for p in loaded_policies:
            self.policies[p] = SelfPlay_QRE_OSA.from_file(obs_shape, n_actions, config['agents'],policy_fnames[p],
                                                          save_dir=Algorithm.get_absolute_save_dir())
            self.policies[p].rationality = rationality
            self.policies[p].model.eval()



        # Testing Params ---------------------------------------------------------
        self.human_policies = ['Seeking', 'Averse']
        self.robot_conditions = ['Oracle', 'RS-ToM', 'Rational']



        # Plotting params
        self.colors = {'Oracle': tuple([50 / 255 for _ in range(3)]), 'RS-ToM': (255 / 255, 154 / 255, 0),
                       'Rational': (255 / 255, 90 / 255, 0),
                       'Seeking': (128/255, 0, 0), 'Averse': (255 / 255, 154 / 255, 0)}


        self.traj_heatmap = TrajectoryHeatmap(self.env)
        self.state_history = []

    def run(self,seed=True):
        if seed:
            torch.seed()
            random.seed()
            np.random.seed()


        total_trials = len(self.robot_conditions) * len(self.human_policies) * self.n_trials
        robot = self.robot_type
        human = self.human_type
        # for i,robot in enumerate(self.robot_conditions):

        # Form robot condition ---------------------------------------------------------
        if robot == 'Oracle':
            names = [human]
            models = [self.policies[human]]
        elif robot == 'RS-ToM':
            names = list(self.policies.keys())
            models = list(self.policies.values())
        elif robot in ['Averse','Rational','Seeking']:
            names = ['Rational']
            models = [self.policies[robot]]

        else: raise ValueError(f'Invalid robot condition: {robot}')



        # Form belief updater ---------------------------------------------------------
        belief_updater = BayesianBeliefUpdate(models, models, names=names,
                                              title=f'Belief | {human} Partner')

        print(f'Simulating trials...')
        # prev_trials = (i + j) * len(self.human_policies)
        for _ in range(self.n_trials):
            belief_updater.reset_prior()
            self.state_history += self.simulate_trial(self.policies[human], belief_updater)
            # prog = round((prev_trials + _ + 1) / total_trials * 100, 2)
            # print(f'\r{prog}% complete | ',end='')
        # print('\n')

    def plot(self,axs = None,
             items = ('onion','dish','soup'),
             titles=('Onions','Dishes','Soups'),
             clean=True):
        print(f'Plotting...')
        plt.ioff()
        if axs is None:
            fig, axs = plt.subplots(1, 3)
        elif not isinstance(axs,list):
            raise ValueError(f'axs must be a tuple/list')
        assert len(axs) == len(items)

        for i, ax in enumerate(axs):
            ax.set_xticks([])
            ax.set_yticks([])
            if titles is not None: ax.set_title(titles[i])

        # plt.ioff(); self.traj_heatmap.blocking = False
        self.traj_heatmap.que_trajectory(self.state_history)
        # self.traj_heatmap.preview()

        self.traj_heatmap.img = self.traj_heatmap.draw_backgrounds(axs=axs)

        masks = self.traj_heatmap.calc_masks()
        for i,item in enumerate(items):
            mask = masks[item]
            if clean:
                mask[:,0]=0; mask[:,-1]=0
                mask[0, :] = 0; mask[-1, :] = 0
            self.traj_heatmap.draw_heatmap(axs[i], mask)
            # self.traj_heatmap.draw_heatmap2(axs[i], mask)

    def simulate_trial(self,partner_policy, belief):
        iego, ipartner = 0, 1
        device = partner_policy.device

        state_history = []
        obs_history = []
        action_history = []
        cum_reward = 0

        rollout_info = {
            'onion_risked': np.zeros([1, 2]),
            'onion_pickup': np.zeros([1, 2]),
            'onion_drop': np.zeros([1, 2]),
            'dish_risked': np.zeros([1, 2]),
            'dish_pickup': np.zeros([1, 2]),
            'dish_drop': np.zeros([1, 2]),
            'soup_pickup': np.zeros([1, 2]),
            'soup_delivery': np.zeros([1, 2]),

            'soup_risked': np.zeros([1, 2]),
            'onion_slip': np.zeros([1, 2]),
            'dish_slip': np.zeros([1, 2]),
            'soup_slip': np.zeros([1, 2]),
            'onion_handoff': np.zeros([1, 2]),
            'dish_handoff': np.zeros([1, 2]),
            'soup_handoff': np.zeros([1, 2]),

        }

        self.env.reset()
        state_history.append(self.env.state.deepcopy())

        for t in count():
            obs = self.env.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=device).unsqueeze(0)

            # CHOOSE ACTIONS ---------------------------------------------------------

            # Choose Partner (Human) Action
            _, partner_iJA, partner_pA = partner_policy.choose_joint_action(obs, epsilon=0)
            partner_iA = partner_iJA % 6

            # Choose Ego Action
            ego_policy = belief.best_response
            _, ego_iJA, ego_pA = ego_policy.choose_joint_action(obs, epsilon=0)
            ego_iA = ego_iJA // 6

            # Calc Joint Action
            action_idxs = (ego_iA, partner_iA)
            joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
            joint_action = (Action.ALL_ACTIONS[ego_iA], Action.INDEX_TO_ACTION[partner_iA])

            # UPDATE BELIEF ---------------------------------------------------------
            belief.update_belief(obs, joint_action_idx)

            # STEP ---------------------------------------------------------
            next_state, reward, done, info = self.env.step(joint_action)
            state_history.append(next_state.deepcopy())

            # LOG ---------------------------------------------------------
            obs_history.append(obs)
            action_history.append(joint_action_idx)
            cum_reward += reward
            for key in rollout_info.keys():
                rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

            if done:  break


        return state_history


    @property
    def fname(self):
        pslip = ''.join(f'{int(self.p_slip * 10)}'.split('.'))
        fname = f'simdata_{self.layout}_pslip0{pslip}_horizon{self.horizon}_ntrials{self.n_trials}'
        return fname
    def save_data(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)
    def load_data(self):
        with open(self.fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.data = loaded_dict

if __name__ == "__main__":
    items = ('onion','dish')
    titles =None # ('Onions','Dishes')
    seed= False
    n_trials = 5

    for _ in range(1):
        fig,axs = plt.subplots(2,4, figsize=(10.5, 4.8))

        ##################################
        # sim = Simulator('risky_coordination_ring',0.4)
        hm = PolicyHeatmap('risky_coordination_ring',0.4,human_type='Averse',n_trials=n_trials)
        hm.run(seed=seed)
        hm.plot(axs=list(axs[0,0:2]),items=items,titles=titles)

        # hm = PolicyHeatmap('risky_coordination_ring',0.4,human_type='Averse')
        hm = PolicyHeatmap('risky_coordination_ring',0.4,human_type='Seeking',n_trials=n_trials)
        hm.run(seed=seed)
        hm.plot(axs=list(axs[1, 0:2]), items=items,titles=None)

        #######################################
        hm = PolicyHeatmap('risky_multipath', 0.15,human_type='Averse',n_trials=n_trials)
        hm.run(seed=seed)
        hm.plot(axs=list(axs[0,2:4]),items=items,titles=titles)

        # hm = PolicyHeatmap('risky_coordination_ring',0.4,human_type='Averse')
        hm = PolicyHeatmap('risky_multipath', 0.15, human_type='Seeking',n_trials=n_trials)
        hm.run(seed=seed)
        hm.plot(axs=list(axs[1, 2:4]), items=items,titles=None)

        ######################################
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.075,hspace=0.00,left=0,bottom=0,right=1,top=1)
    plt.show()
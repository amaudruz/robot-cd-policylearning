from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.sail import SAIL
from crowd_nav.policy.sail_traj import SAILTRAJ
from crowd_nav.policy.sail_traj import SAILTRAJ
from crowd_nav.policy.sail_traj_mult import SAILTRAJMULT


policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['sail'] = SAIL
policy_factory['sail_traj'] = SAILTRAJ
policy_factory['sail_traj_simple'] = SAILTRAJ
policy_factory['sail_traj_mult'] = SAILTRAJMULT


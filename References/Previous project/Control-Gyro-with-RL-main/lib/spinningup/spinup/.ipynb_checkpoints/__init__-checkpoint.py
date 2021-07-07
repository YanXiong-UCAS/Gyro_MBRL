# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
from spinup.algos.tf1.sac.sac import sac as sac_tf1
from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1

from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ddpg.ddpg_no_test import ddpg_no_test as ddpg_no_test_pytorch
from spinup.algos.pytorch.ddpg.ddpg_pause import ddpg_pause as ddpg_pause_pytorch
from spinup.algos.pytorch.ddpg.ddpg_her import ddpg_mher as ddpg_mher_pytorch
from spinup.algos.pytorch.ddpg.ddpg_her import ddpg_vher as ddpg_vher_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
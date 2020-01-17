from rlpyt.utils.collections import AttrDict


class RobotTrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._cur_discount = 1
        self.env_info = dict()

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount
        for key, value in env_info._asdict().items():
            if key in self.env_info:
                self.env_info[key].append(value)
            else:
                self.env_info[key] = [value]

    def terminate(self, observation):
        for key, value_list in self.env_info.items():
            self.__setattr__(key, sum(value_list) / len(value_list))
            self.__setattr__(key + '_traj_sum', sum(value_list))
        del self.env_info  # has to be deleted because rlpyt assumes this object doesn't contain dicts
        return self

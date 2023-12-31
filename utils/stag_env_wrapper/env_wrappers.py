# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if np.array(done).all():
                # if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            ob = env.render()
            remote.send(ob)  # rgb_array
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))

        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        # bigimg = tile_images(imgs)
        # if mode == 'human':
        #     self.get_viewer().imshow(bigimg)    #
        #     return self.get_viewer().isopen

    #     elif mode == 'rgb_array':
    #         return bigimg
    #     else:
    #         raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs_sc: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_images(self):
        # self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        # imgs = _flatten_list(imgs)
        return imgs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

# class DummyVecEnv(VecEnv):
#     """
#     VecEnv that does runs multiple environments sequentially, that is,
#     the step and reset commands are send to one environment at a time.
#     Useful when debugging and when num_env == 1 (in the latter case,
#     avoids communication overhead)
#     """
#     def __init__(self, env_fns):
#         """
#         Arguments:
#
#         env_fns: iterable of callables      functions that build environments
#         """
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
#         obs_space = env.observation_space
#         self.keys, shapes, dtypes = obs_space_info(obs_space)
#
#         self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
#         self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
#         self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
#         self.buf_infos = [{} for _ in range(self.num_envs)]
#         self.actions = None
#         self.spec = self.envs[0].spec
#
#     def step_async(self, actions):
#         listify = True
#         try:
#             if len(actions) == self.num_envs:
#                 listify = False
#         except TypeError:
#             pass
#
#         if not listify:
#             self.actions = actions
#         else:
#             assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
#             self.actions = [actions]
#
#     def step_wait(self):
#         for e in range(self.num_envs):
#             action = self.actions[e]
#             # if isinstance(self.envs_sc[e].action_space, spaces.Discrete):
#             #    action = int(action)
#
#             obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
#             if self.buf_dones[e]:
#                 obs = self.envs[e].reset()
#             self._save_obs(e, obs)
#         return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
#                 self.buf_infos.copy())
#
#     def reset(self):
#         for e in range(self.num_envs):
#             obs = self.envs[e].reset()
#             self._save_obs(e, obs)
#         return self._obs_from_buf()
#
#     def _save_obs(self, e, obs):
#         for k in self.keys:
#             if k is None:
#                 self.buf_obs[k][e] = obs
#             else:
#                 self.buf_obs[k][e] = obs[k]
#
#     def _obs_from_buf(self):
#         return dict_to_obs(copy_obs_dict(self.buf_obs))
#
#     def get_images(self):
#         return [env.render(mode='rgb_array') for env in self.envs]
#
#     def render(self, mode='human'):
#         if self.num_envs == 1:
#             return self.envs[0].render(mode=mode)
#         else:
#             return super().render(mode=mode)

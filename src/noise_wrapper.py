import numpy as np
import gymnasium as gym

class NoisyObservationWrapper(gym.ObservationWrapper):
    """
    A Gymnasium observation wrapper that adds noise to the observations.

    Attributes:
    noise_percentage (float): The percentage of noise to add to the observations.
    noisy_indices (list or None): The indices of the observation array to which noise should be added. 
                                  If None, noise is applied to the entire observation.
    """
    def __init__(self, env, noise_percentage=0.1, noisy_indices=None):
        """
        Initialize the NoisyObservationWrapper.

        Parameters:
        env (gym.Env): The environment to wrap.
        noise_percentage (float): The percentage of noise to add to the observations. Default is 0.1 (10% noise).
        noisy_indices (list or None): The indices of the observation array to which noise should be added. 
                                      If None, noise is applied to the entire observation.
        """
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_percentage = noise_percentage
        self.noisy_indices = noisy_indices

    def observation(self, observation):
        """
        Modify the observation by adding noise to it.

        Parameters:
        observation (numpy.ndarray): The original observation from the environment.

        Returns:
        numpy.ndarray: The modified observation with added noise.
        """
        if self.noisy_indices is None:
            noise = np.random.uniform(-self.noise_percentage, self.noise_percentage, size=observation.shape)
            return observation * (1 + noise)
        else:
            noise = np.zeros_like(observation)
            noise[self.noisy_indices] = np.random.uniform(-self.noise_percentage, self.noise_percentage, size=len(self.noisy_indices))
            noisy_observation = observation * (1 + noise)
            return noisy_observation
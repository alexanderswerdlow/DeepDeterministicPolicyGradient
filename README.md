# Deep Deterministic Policy Gradient

Application of a Deep Deterministic Policy Gradient (DDPG) method introduced in [1] to control a rocket with noisy thrust and an off-center mass. The Mujoco enviorment is loosely based off of [2].

[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

[2] [gym_rotor](https://github.com/inkyusa/gym_rotor)

## Dependencies

Before running the project, you’ll need the following installed:
* [Open AI Gym](https://github.com/openai/gym)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [mujoco-py](https://github.com/openai/mujoco-py)
* [Mujoco](https://mujoco.org/download)
Once the repository is cloned, run `pip install -e . && pip install scipy` in the root directory to install the "rocket" package which contains the custom rocket enviorment for gym and scipy. This is necessary for the simulations to run.

You will also need to run the following on Ubuntu: `sudo apt-get install libglew-dev libx11-dev patchelf`

For best performance, it is recommended that you use an Nvidia GPU with CUDA installed. This should be possible on both Linux and Windows but only Linux was tested. You should be able to run the code without a GPU, however this is untested.

Specifically, this project has been tested with Ubuntu 20.04 w/CUDA 11.3 drivers and an RTX 2060. The python enviorment was a vanilla system install with version 3.8.10 with `gym=0.19.0`, `torch=1.10.0+cu113`, `mujoco-py=2.1.2.14` and Mujoco v2.10.

To run the enviorment, you must also change the absolute path on line 64 of rocket/envs/rocket_v2.py.

*Additionally, [markdown-math-gh-compiler](https://github.com/jeremy-rifkin/markdown-math-gh-compiler) was used to embed latex formulae in this README.

## DDPG

The goal is to develop a policy that returns an action in some continuous space. As the name entails, this policy is deterministic, thus creating a mapping from state to action: <img alt="\pi^*(s)=a" src="https://render.githubusercontent.com/render/math?math=%5Cpi%5E%2a%28s%29%3Da" style="transform: translateY(20%);" />

We do this with an off-policy approach with an actor-critic structure. We create an actor <img alt="\pi_\theta(s)" src="https://render.githubusercontent.com/render/math?math=%5Cpi_%5Ctheta%28s%29" style="transform: translateY(20%);" /> that represents our policy and a critic <img alt="Q_\theta(s, a)" src="https://render.githubusercontent.com/render/math?math=Q_%5Ctheta%28s%2C%20a%29" style="transform: translateY(20%);" /> that learns the Q-function.

I chose an off-policy approach since I sought to apply this to control problems where we have no known starting expert trajectories and we may need significant exploration which is more difficult with on-policy approaches where the accuracy of <img alt="Q_\theta(s,a)" src="https://render.githubusercontent.com/render/math?math=Q_%5Ctheta%28s%2Ca%29" style="transform: translateY(20%);" /> is dependent on the current policy.

This off-policy approach uses "experience replay" by adding new transitions: <img alt="(s, a, r, s')" src="https://render.githubusercontent.com/render/math?math=%28s%2C%20a%2C%20r%2C%20s%27%29" style="transform: translateY(20%);" /> to a running dataset and recomputing the policy and Q-function gradient by selecting some batch from this dataset.

We use the following Q-value function definition seen in lecture:

<img alt="Q_\theta(s,a)\approx Q^*(s,a)=E[r(s,a) + \gamma\max_{a'} {Q^*(s', a')]}" src="https://render.githubusercontent.com/render/math?math=Q_%5Ctheta%28s%2Ca%29%5Capprox%20Q%5E%2a%28s%2Ca%29%3DE%5Br%28s%2Ca%29%20%2B%20%5Cgamma%5Cmax_%7Ba%27%7D%20%7BQ%5E%2a%28s%27%2C%20a%27%29%5D%7D" style="transform: translateY(20%);" /> where <img alt="s' \sim P(s'|s,a)" src="https://render.githubusercontent.com/render/math?math=s%27%20%5Csim%20P%28s%27%7Cs%2Ca%29" style="transform: translateY(20%);" />

In order to determine how our <img alt="Q_\theta(s,a)" src="https://render.githubusercontent.com/render/math?math=Q_%5Ctheta%28s%2Ca%29" style="transform: translateY(20%);" /> compares with the optimal <img alt="Q^*(s,a)" src="https://render.githubusercontent.com/render/math?math=Q%5E%2a%28s%2Ca%29" style="transform: translateY(20%);" />, we find the difference between <img alt="Q_\theta(s,a)" src="https://render.githubusercontent.com/render/math?math=Q_%5Ctheta%28s%2Ca%29" style="transform: translateY(20%);" /> and <img alt="r(s,a) + \gamma\max_{a'} {Q^*(s', a')}" src="https://render.githubusercontent.com/render/math?math=r%28s%2Ca%29%20%2B%20%5Cgamma%5Cmax_%7Ba%27%7D%20%7BQ%5E%2a%28s%27%2C%20a%27%29%7D" style="transform: translateY(20%);" />. However, since we don't know <img alt="a'" src="https://render.githubusercontent.com/render/math?math=a%27" style="transform: translateY(20%);" /> apriori, we use our actor <img alt="\pi_\theta(s)" src="https://render.githubusercontent.com/render/math?math=%5Cpi_%5Ctheta%28s%29" style="transform: translateY(20%);" /> to approximate this value. We construct our loss function as the expectation of MSE since we wish to compute the loss only over some random sample from our dataset:

<img alt="L(\theta)=E_{d\sim D}\bigg[\big(Q_\theta(s,a)-(r(s,a)+ \gamma Q_\theta(s', \pi_\theta(s')) \big)^2\bigg]" src="https://render.githubusercontent.com/render/math?math=L%28%5Ctheta%29%3DE_%7Bd%5Csim%20D%7D%5Cbigg%5B%5Cbig%28Q_%5Ctheta%28s%2Ca%29-%28r%28s%2Ca%29%2B%20%5Cgamma%20Q_%5Ctheta%28s%27%2C%20%5Cpi_%5Ctheta%28s%27%29%29%20%5Cbig%29%5E2%5Cbigg%5D" style="transform: translateY(20%);" /> where <img alt="d=(s,a,r,s')\in D" src="https://render.githubusercontent.com/render/math?math=d%3D%28s%2Ca%2Cr%2Cs%27%29%5Cin%20D" style="transform: translateY(20%);" />

We previously saw that we could find an optimal policy <img alt="\pi^*(s)" src="https://render.githubusercontent.com/render/math?math=%5Cpi%5E%2a%28s%29" style="transform: translateY(20%);" /> from the optimal Q-function <img alt="Q^*(s,a)" src="https://render.githubusercontent.com/render/math?math=Q%5E%2a%28s%2Ca%29" style="transform: translateY(20%);" /> and here our goal is the same. Thus we can define the gradient of our objective as

<img alt="\nabla L(\theta_\pi)=E_{d\sim D}\bigg[\nabla{\theta_\pi} Q_\theta(s,\pi_\theta(s))\bigg]=E_{d\sim D}\bigg[\nabla{\pi_\theta(s)} Q_\theta(s,\pi_\theta(s)) \nabla{\theta_\pi} \pi_{\theta_\pi}(s)\bigg]" src="https://render.githubusercontent.com/render/math?math=%5Cnabla%20L%28%5Ctheta_%5Cpi%29%3DE_%7Bd%5Csim%20D%7D%5Cbigg%5B%5Cnabla%7B%5Ctheta_%5Cpi%7D%20Q_%5Ctheta%28s%2C%5Cpi_%5Ctheta%28s%29%29%5Cbigg%5D%3DE_%7Bd%5Csim%20D%7D%5Cbigg%5B%5Cnabla%7B%5Cpi_%5Ctheta%28s%29%7D%20Q_%5Ctheta%28s%2C%5Cpi_%5Ctheta%28s%29%29%20%5Cnabla%7B%5Ctheta_%5Cpi%7D%20%5Cpi_%7B%5Ctheta_%5Cpi%7D%28s%29%5Cbigg%5D" style="transform: translateY(20%);" />

where <img alt="\theta_\pi" src="https://render.githubusercontent.com/render/math?math=%5Ctheta_%5Cpi" style="transform: translateY(20%);" /> denotes the function parameters of the actor specifically.

As discussed in lecture, stability of learning can be improved by computing the bellman value error using an actor and critic that lag behind the current actor and critic. Thus after each new transition we add to our dataset, we compute the aforementioned gradients, update our network parameters with some learning rate, and perform <img alt="\theta_{target} \leftarrow \tau \theta+(1-\tau) \theta_{target}" src="https://render.githubusercontent.com/render/math?math=%5Ctheta_%7Btarget%7D%20%5Cleftarrow%20%5Ctau%20%5Ctheta%2B%281-%5Ctau%29%20%5Ctheta_%7Btarget%7D" style="transform: translateY(20%);" /> where <img alt="\theta" src="https://render.githubusercontent.com/render/math?math=%5Ctheta" style="transform: translateY(20%);" /> parameterizes both the actor and critic.

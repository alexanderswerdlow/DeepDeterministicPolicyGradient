# Deep Deterministic Policy Gradient

Control of a rocket w/noisy thrust and an off-center mass using DDPG [1]. Specifically, the policy can control only the acceleration of the mass which is perpendicular to the direction of the rocket. See `videos/generated_policy.mkv` for a demo, and a closeup of the model in `videos/rocket_screenshot_1.png`. 

## Dependencies
Before running the project, you’ll need the following installed:

- [Open AI Gym](https://github.com/openai/gym)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [mujoco-py](https://github.com/openai/mujoco-py)
- [Mujoco](https://mujoco.org/download)

Once the repository is cloned, run `pip install -e .` in the root directory to install the "rocket" package which contains the custom rocket enviorment for gym. This is necessary for the simulations to run.


For best performance, it is recommended that you use an Nvidia GPU with CUDA installed. This should be possible on both Linux and Windows but only Linux was tested. You should be able to run the code without a GPU, however this is untested.

Specifically, this project has been tested with Ubuntu 20.04 w/CUDA 11.3 drivers and an RTX 2060. The python enviorment was a vanilla system install with version 3.8.10 with `gym=0.19.0`, `torch=1.10.0+cu113`, `mujoco-py=2.1.2.14` and Mujoco v2.10.

Misc: The Mujoco enviorment is loosely based off of [2].

## DDPG

The goal is to develop a policy that returns an action in some continuous space. As the name entails, this policy is deterministic, thus creating a mapping from state to action: $\pi^*(s)=a$

We do this with an off-policy approach with an actor-critic structure. We create an actor $\pi_\theta(s)$ that represents our policy and a critic $Q_\theta(s, a)$ that learns the Q-function.



I chose an off-policy approach since I sought to apply this to control problems where we have no known starting expert trajectories and we may need significant exploration which is more difficult with on-policy approaches where the accuracy of $Q_\theta(s,a)$ is dependent on the current policy.



This off-policy approach uses "experience replay" by adding new transitions: $(s, a, r, s')$ to a running dataset and recomputing the policy and Q-function gradient by selecting some batch from this dataset.



We use the following Q-value function definition seen in lecture: 

$`Q_\theta(s,a)\approx Q^{*} (s,a) = E[r(s,a) + \gamma \max{a'}{Q^*(s', a')]} `$ where $s' \sim P(s'|s,a)$

In order to determine how our $Q_\theta(s,a)$ compares with the optimal $Q^*(s,a)$, we find the difference between $Q_\theta(s,a)$ and $r(s,a) + \gamma\max_{a'} {Q^*(s', a')}$. However, since we don't know $a'$ apriori, we use our actor $\pi_\theta(s)$ to approximate this value. We construct our loss function as the expectation of MSE since we wish to compute the loss only over some random sample from our dataset:

$L(\theta)=E_{d\sim D}\bigg[\big(Q_\theta(s,a)-(r(s,a)+ \gamma Q_\theta(s', \pi_\theta(s')) \big)^2\bigg]$ where $d=(s,a,r,s')\in D$



We can find an optimal policy $\pi^*(s)$ from the optimal Q-function $Q^*(s,a)$ and here our goal is the same. Thus we can define the gradient of our objective as 

$\nabla L(\theta_\pi)=E_{d\sim D}\bigg[\nabla{\theta_\pi} Q_\theta(s,\pi_\theta(s))\bigg]=E_{d\sim D}\bigg[\nabla{\pi_\theta(s)} Q_\theta(s,\pi_\theta(s)) \nabla{\theta_\pi} \pi_{\theta_\pi}(s)\bigg]$

where $\theta_\pi$ denotes the function parameters of the actor specifically. 

Stability of learning can be improved by computing the bellman value error using an actor and critic that lag behind the current actor and critic. Thus after each new transition we add to our dataset, we compute the aforementioned gradients, update our network parameters with some learning rate, and perform $\theta_{target} \leftarrow \tau \theta+(1-\tau) \theta_{target}$ where $\theta$ parameterizes both the actor and critic.


[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

[2] [gym_rotor](https://github.com/inkyusa/gym_rotor)

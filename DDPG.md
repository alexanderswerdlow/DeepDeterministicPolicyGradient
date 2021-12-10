# Deep Deterministic Policy Gradient



The goal is to develop a policy that returns an action in some continuous space. As the name entails, this policy is deterministic, thus creating a mapping from state to action: $\pi^*(s)=a$



We do this with an off-policy approach with an actor-critic structure. We create an actor $\pi_\theta(s)$ that represents our policy and a critic $Q_\theta(s, a)$ that learns the Q-function.



I chose an off-policy approach since I sought to apply this to control problems where we have no known starting expert trajectories and we may need significant exploration which is more difficult with on-policy approaches where the accuracy of $Q_\theta(s,a)$ is dependent on the current policy.



This off-policy approach uses "experience replay" by adding new transitions: $(s, a, r, s')$ to a running dataset and recomputing the policy and Q-function gradient by selecting some batch from this dataset.



We use the following Q-value function definition seen in lecture: 

$Q_\theta(s,a)\approx Q^*(s,a)=E[r(s,a) + \gamma\max_{a'} {Q^*(s', a')]}$ where $s' \sim P(s'|s,a)$

In order to determine how our $Q_\theta(s,a)$ compares with the optimal $Q^*(s,a)$, we find the difference between $Q_\theta(s,a)$ and $r(s,a) + \gamma\max_{a'} {Q^*(s', a')}$. However, since we don't know $a'$ apriori, we use our actor $\pi_\theta(s)$ to approximate this value. We construct our loss function as the expectation of MSE since we wish to compute the loss only over some random sample from our dataset:

$L(\theta)=E_{d\sim D}\bigg[\big(Q_\theta(s,a)-(r(s,a)+ \gamma Q_\theta(s', \pi_\theta(s')) \big)^2\bigg]$ where $d=(s,a,r,s')\in D$



We previously saw that we could find an optimal policy $\pi^*(s)$ from the optimal Q-function $Q^*(s,a)$ and here our goal is the same. Thus we can define the gradient of our objective as 

$\nabla L(\theta_\pi)=E_{d\sim D}\bigg[\nabla{\theta_\pi} Q_\theta(s,\pi_\theta(s))\bigg]=E_{d\sim D}\bigg[\nabla{\pi_\theta(s)} Q_\theta(s,\pi_\theta(s)) \nabla{\theta_\pi} \pi_{\theta_\pi}(s)\bigg]$

where $\theta_\pi$ denotes the function parameters of the actor specifically. 



As discussed in lecture, stability of learning can be improved by computing the bellman value error using an actor and critic that lag behind the current actor and critic. Thus after each new transition we add to our dataset, we compute the aforementioned gradients, update our network parameters with some learning rate, and perform $\theta_{target} \leftarrow \tau \theta+(1-\tau) \theta_{target}$ where $\theta$ parameterizes both the actor and critic.

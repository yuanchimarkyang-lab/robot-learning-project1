# Robot Learning on LIBERO — Behavior Cloning
## 1. Motivation
This mini-project explores how different state-of-the-art algorithms performs on robot-learning tasks.
The objective is to increase my understanding of the algorithms in a practical setting, to gain hands-on experiences in training/fine-tuning, and to obtain insights by close-loop evaluation. 

Currently, this project includes three different algorithms 
- ACT (Action Chunking with Transformers)
- Diffusion Policy, 
- SmolVLA (Small VLA) 

and focuses one specific task: `put bowl on the plate` in `LIBERO-Goal`.

This is an on-going project; different algorithms, different tasks, and different aspects of robot learning might be inclueded in the future.

## 2. Experiment Setup
| Setting | Value |
| -------- | ------- |
| Environment | LIBERO-Goal |
| TaskPut | the bowl on the plate |
| Training data | Same 49 demonstrations |
| Control frequency | 20 Hz |
| Max rollout length | 300 steps |
| Final evaluation | 50 official LIBERO initial states |
| Models | ACT, Diffusion Policy, SmolVLA-LoRA |
| Batch Size | 8 |
| Framework | LeRobot |
| Hardware	| RTX 5060 Laptop GPU, 8 GB |


## 3. Learning Curves
### ACT (Action Chunking with Transformers)
![Learning Curve](figures/act_LC.png)

The training loss decreases smoothly as the trianing steps increases.
Unlike the training loss, the success rate jumps abruptly at 500-1000 steps from 0% to over 90% . It then seems to saturate and fluctuates between 94% and 98% for the rest of the training. 
My interpretation is that
1. though failure rate is likely to decreases with training loss, the two measures could behave differently; training loss is not an absolute predictor of failure rate.
2. though it seems like that the model based on ACT has high success rate, its performance is not perfect and learning seems to saturate, which worthes further investigation. 


### Diffusion
![Learning Curve](figures/diffusion_LC.png)

The training loss decreases as the training steps increases.
The success rate roughly increases with the training steps, but with much larger flucturations. 
The success rate reaches 84% at `step=4500`, but falls to 38% at `step=6000` before increases to 94% at `step=10000`. 
This result raises a couple questions:
1. What causes such learning behavior? Is there anyway to prevent this? Does this have anything to do with diffusion policy?
2. What happen at the stark fail at `step=6000`?
3. If we train longer, would the policy reach saturation?


### SmolVLA
![Learning Curve](figures/smolvla_LC.png)

The ROLA fine-tuning on SmolVLA shows that the training loss decreases smoothly with the training steps. 
The success rate increases fast initially (between `step=500` and `step=2000`) and then its increase slow down (between `step=200` and `step=4000`) and finally saturates at about `step=4000` to `step=5000`.  

Similar to ACT, the failure rate and the training loss, though move in the same direction, behaves slightly differently.
Also, SmolVLA though produce high performance, it does not solve all the cases and this requires further investigation.


## 4. Rollout / Failure Analysis
### ACT
**Step = 500**
This checkpoint has 0% success rate; all 50 cases failed to grasp the bowl (but reach toward the bowl successfully). 

**Step = 800**
This checkpoint reaches an success rate of 88% during the success rate jump. It has 6 failed cases, among which five were failures to grasp while the remaining one is failure to place the bowl at the center of the plate.


**Step = 10000**
As a representative of the model at saturation, the sucsess rate is 98%, with only one transport error-the bowl fell during the transport and the end-effector failed to recover from the failure. It would be interesting to see how much of the success has gone through the recovery.

According to the failure analysis at different stages on the training curve, my currently hypothesis is that learning how to grasp is probably the bottleneck for this task for ACT. At `step=500`, the model is stuck at learning how to grasp. As learning continue (`step=800`) the model learned a good amount of how to grasp and the end-effector successfully grasped many cases, leaving only 5 grasping error. For cases where the bowl is grasped successfully, only 1 encounters placement error as if the skills required for the remaining task (ie., transport, placement, and release) have already been acquired.

A quick look at the initial state id's of the failures during the saturated stage (step=1000 to 10000) shows that these failures don't center on specific initial states, indicating that limiting factor is not a few particular hard cases and worthes further investigation.

### Diffusion
**Step = 500** With 0% success rate, all 50 cases were failures to reach; the end-effectors do not even reach for the bowl

**Step = 10000**  three failure cases, two failed to grasp the bowl while one failed to transport the bowl properly. 

### SmolVLA
**Step = 500**
The success rate for this checkpoint is 14%, higher than ACT at the same step. 
In the 43 failed cases, most were failure to grasp (35) while 6 were failure to transport, naming the bowl dropped during the transport, and 2 were failure to place the bowl correctly on the plate.
Interestingly, at this checkpoint, recovery attempt starts to appear. 
During the roll-out `with initial_state_id=23`, the end-effector initially failed to grasp the bowl, but then it made another attemp and this time it grasped the bowl successfully. Though in the end it still failed to place the bowl at the right place on the plate, the appearance of this kind of recovery behaviors is worth noticing.

**Step = 2000**
The success rate reaches 90% after 1500 more training steps.
Besides one failures to grasp the bowl, the remaining four cases were failure at placement. 
In these cases, the end-effector successfully grasped the bowl, transported it to above the plate, but placed it on the edge of the plate instead of its center. 
The end-effector then attemped to regrasp the bowl but it failed.
It is interesting to see such recovery behavior, and it may be worthwhile to check how many successes are reached with recovery.


**Step = 5000**
The success rate reaches 94%, with one grasping failure and two transport failure.
However, no recovery attempt was observed.


## 5. Preliminary Findings

## 6. Limitations and Open Questions
- Each checkpoint is evaluated based on the official set, containing only 50 episodes. Therefore, the difference in success rate percentage points should not be directly interpreted as having statistical significance. 

## 7. Next Experients

## 4. Discussion and Outlook
- Based on training steps versus success rate, it seems like that ACT can be trained most efficiently, SmolVLA with LORA the second, while diffusion policy the least. Why is this the case?
- But once it is trained, they can reach comparable performance. However, none of them are perfect (at this given the amount of training I have). What is limiting their performance?
- Which algorithm has the capability of multi-task learning? SmolVLA is designed to do that. How about ACT and diffusion? 

# DEVELOPMENT OF THE FUNCTIONS USED IN THE PPO ALGORITHM
import numpy as np 
import torch
import pong_utils
from pprint import pprint

def debug(item, name, print_e=False):
    print("New item: {}".format(name))
    print(f"Type: {type(item)}")
    if print_e:
        pprint(item)
    try:
        print(f"Length: {len(item)}")
        pprint(f"First element: {item[0]}")
        pprint(f"Last element: {item[-1]}")
        try:
            print(f"Shape: {item.shape}")
        except:

            pass
        try:
            pprint("First element shape: {}".format(item[0].shape))
        except Exception as e:
            pprint(e)
    except:
        print("Object has no length")
    print("")

def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):

    
    
    """I DONT KNOW WHAT THIS FUNCTION DOES. 
    UPDATE: THIS IS ACTUALLY THE OBJECTIVE FUNCTION. IT IS CALLED SURROGATE BECAUSE IT IS A SUBSTITUTE FOR THE ACTUAL
    OBJECTIVE RUNCTION, SINCE WE CAN NOT CALCULATE THE GRADIENT OF THE ACTUAL OBJECTIVE FUNCTION.
    
    
    
    Understanding:
    1 - The function 1/t*sum(Rf*log(pi(a|s))) is not the gradient, but the reward function we'll try to maximize THIS IS THE OBJECTIVE FUNCTION
    2 - 
    
    TODO:
    1 - Convert the rewards to future rewards (summing and discounting) (R_fut)
    3 - Normalize
    2 - Convert probabilities for action Left -> 1 - prob
    3 - Calculate U (expected return) -> sum(R_fut) * prob(action|state)
    4 - Average everything over steps and trajectories
    
    5
    
    OBS: THERE ARE TWO WAYS OF CALCULATING THE DISCOUNT, AND THEY ARE BOTH IMPLEMENTED HERE
    
    
    What do I know?:
    - P(a=right|s) = old_probs
    - P(a=left|s) = 1 - old_probs
    
    """
    assert len(states) == len(rewards), "States and rewards must have the same length"
    
    # DISCOUNTING
    steps = len(states)
    discounts = np.asarray([discount]*len(rewards))**np.asarray(list(range(steps)))    
    future_rewards = []
    for i in range(steps): # For each step, it recalculates the discounts, and discount(t) is always 1, discount(t+1) = 0.995, and so on
        future_rewards.append(sum(np.asarray(rewards[i:])*np.asarray(discounts[:steps-i])[:, np.newaxis]))
    debug(future_rewards, "future_rewards")

    # # DISCOUNTING LONG VERSION
    # steps = len(states)
    # discount_cf = np.asarray(list(range(steps)))
    # discounts = np.asarray([discount]*len(rewards))
    # discounts = discounts**discount_cf
    # future_rewards = []
    # rewards = np.asarray(rewards)
    # for i in range(steps):
    #     r_fut = rewards[i:]
    #     current_discounts = discounts[:steps-i][:, np.newaxis]
    #     r_fut_discounted = r_fut*current_discounts
    #     r_total = sum(r_fut_discounted)
    #     future_rewards.append(r_total)
    # # debug(future_rewards, "validation")


    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    future_rewards = torch.tensor(future_rewards, dtype=torch.float, device=device)
    
    
    # NORMALIZATION
    # future_rewards = (future_rewards - np.mean(future_rewards, axis=1)[:,np.newaxis])/np.std(future_rewards, axis=1)[:,np.newaxis]
    mean = torch.mean(future_rewards, dim=1, keepdim=True)
    std = torch.std(future_rewards, dim=1, keepdim=True)
    future_rewards = (future_rewards - mean)/std
    future_rewards[torch.isnan(future_rewards)] = 0.0
    
    # Convert probabilities for action Left -> 1 - prob
#     print(f"Actions: {actions[:5]}")
    right = 4
    left = 5
    old_probs = torch.where(actions==right, old_probs, 1.0-old_probs) # ?? IS THIS CORRECt?
 

    # average the results over steps
    expected_return = old_probs*future_rewards
    
    average_return = torch.mean(expected_return, dim=0)  
#     print(f"Steps averaged return: {average_return}")
    # average the results over trajectories
    average_return = torch.mean(average_return, dim=0)    
    print(f"overall return: {average_return}")

#     actions = torch.tensor(actions, dtype=torch.int8, device=device)

    
    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(beta*entropy + average_return)


Lsur= surrogate(policy, prob, state, action, reward)

print(Lsur)

    
def surrogate_lucas(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):
    
    # CCONVERT TO FUTURE REWARDS AND DISCOUNT (SHORT VERSION)
    steps = len(states)
    discounts = np.asarray([discount]*len(rewards))**np.asarray(list(range(steps)))
    future_rewards = []
    rewards_array = np.asarray(rewards)
    for i in range(len(rewards)):
        future_rewards.append(sum(rewards_array[i:]*discounts[:steps-i][:, np.newaxis]))
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    future_rewards = torch.tensor(future_rewards, dtype=torch.float, device=device, requires_grad=True)
    
    # NORMALIZATION
    mean = torch.mean(future_rewards, dim=1, keepdim=True)
    std = torch.std(future_rewards, dim=1, keepdim=True)
    future_rewards = (future_rewards - mean)/(std + 1.0e-10)
    
    # CONVERT PROBS FOR LEFT ACTIONS -> P(LEFT) = 1 - P(RIGHT)
    right = 4
    old_probs = torch.where(actions==right, old_probs, 1.0-old_probs) # means that wherever it is 4, choose from old_probs, otherwise choose from 1.0-old_probs

    # EXPECTED RETURN: U(THETA) = SUM(R_FUT * log(P(A|S)))
    log_probs = torch.log(old_probs)
    expected_return = log_probs*future_rewards
    
    # AVERAGE OVER STEPS
    averaged_return = torch.mean(expected_return, dim=0)
    
    # AVERAGE OVER TRAJECTORIES
    averaged_return = torch.mean(averaged_return, dim=0)
    
    
    # ENTROPY TERM FROM CLASS
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == right, new_probs, 1.0-new_probs)
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    # return torch.mean(averaged_return + beta*entropy)
    return averaged_return + beta*entropy


Lsur = surrogate_lucas(policy, prob, state, action, reward)

print(Lsur)
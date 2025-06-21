from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv


MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
Direction  = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}  # 0:right 1:down 2:left 3:up
# New heading after TL or TR
LEFT  = {0:3, 1:0, 2:1, 3:2}
RIGHT = {0:1, 1:2, 2:3, 3:0}

def get_initial_state(env):
    x,y = env.unwrapped.agent_pos                 # position (x,y)
    heading = env.unwrapped.agent_dir             # 0~3
    has_key =0                                    # no key at initial
    door_status = 0                               # door is lock
    
    return (x, y, heading, has_key, door_status)  # tuple

def legal_action(X,env,info):
    x,y,heading,has_key,door_status = X
    dx,dy = Direction[heading]
    tx,ty = x+dx,y+dy                             # next grid position after moving forword
    legal = [TL,TR]                               # turning always ok
    
    if 0 <= tx < info["width"]and 0 <= ty < info["height"]:
        #Check what is object in front_obj 
        #   None   → empty floor
        #   Key    → a key object
        #   Door   → a door object
        #   Goal   → the goal object
        front_obj = env.grid.get(tx, ty)            
        # 1) Move Forward (MF) is allowed if:
        #    - the cell is empty (None)
        #    - or contains a Key or the Goal
        #    - or contains a Door that is either already open in the env
        #      or has been “logically” opened (door_status == 1)

        if front_obj is None or isinstance(front_obj, (Key, Goal)) or (isinstance(front_obj, Door) and (front_obj.is_open or door_status == 1)):
            legal.append(MF)
        # 2) Pick Up Key (PK) is allowed if:
        #    - the cell contains a Key
        #    - and we have not yet picked up a key (has_key == 0)
        if isinstance(front_obj, Key) and has_key==0:
            legal.append(PK)
        # 3) Unlock Door (UD) is allowed if:
        #    - the cell contains a Door
        #    - the door is locked (front_obj.is_locked == True)
        #    - and we are currently carrying a key (has_key == 1)
        if isinstance(front_obj, Door) and front_obj.is_locked and has_key == 1:
            legal.append(UD)       
    return legal

# motion_model
def transition(X,u,info,env):
    x,y,heading,has_key,door_status = X
    next_x,next_y,next_heading,next_key,next_door = x,y,heading,has_key,door_status
    # Turn Left or Right
    if u == TR:
        next_heading =RIGHT[heading]
    elif u == TL:
        next_heading= LEFT[heading]
    # Move Forward
    elif u == MF:
        dx,dy = Direction[heading]
        tx,ty = x+dx,y+dy
        if 0 <= tx < info["width"] and 0 <= ty < info["height"]:
            front_obj = env.grid.get(tx, ty)
            if (front_obj is None or isinstance(front_obj,(Key,Goal))or isinstance(front_obj,Door) and (front_obj.is_open or door_status == 1)):
                next_x,next_y= tx , ty
    #Pickup Key
    elif u==PK:
        dx,dy = Direction[heading]
        tx,ty = x+dx,y+dy
        if 0 <= tx<info["width"] and 0 <= ty < info["height"]:
            front_obj = env.grid.get(tx, ty)
            if isinstance(front_obj, Key) and has_key == 0:
                next_key = 1
    #Unlock Door
    elif u==UD:
        dx, dy = Direction[heading]
        tx, ty = x + dx, y + dy
        if 0 <= tx < info["width"] and 0 <= ty < info["height"]:
            front_obj = env.grid.get(tx, ty)
            if isinstance(front_obj, Door) and door_status==0 and has_key==1:
                next_door = 1
    cost = step_cost(u)
    return (next_x,next_y,next_heading,next_key,next_door), cost

def terminal_cost(X,info):
    """
    If the agent’s (x,y) coordinate matches the goal position, the terminal cost is zero.
    Otherwise (i.e.\ the horizon expires before reaching the goal), 
    we impose a large penalty so that anext_y optimal finite‑horizon policy will strive to reach the goal before time runs out.
    
    """
    if (X[0], X[1]) == tuple(info["goal_pos"]):
        return 0
    else:
        return 1000
    
def enumerate_state(info):
    X = []
    for x in range(info["width"]):
        for y in range(info["height"]):
            for heading in range(4):  
                for has_key in [0, 1]:
                    for door_status in [0, 1]:
                        X.append((x, y, heading, has_key, door_status))
    return X

#Dynamic-Programming      
def doorkey_problem(env, info, gamma=0.99, theta=1e-6):
    # enumerate all states
    X = enumerate_state(info)
    V = { x: 0.0 for x in X }           # initialize the value function V(s)=0 for every state


    while True:                         # value iteration loop, repeat until convergence
        delta = 0.0                     # track the maximum change in V during this iteration
        for x in X:
            best_q = float('inf')
            for u in legal_action(x, env, info):
                x_next, cost = transition(x, u, info, env)
                best_q = min(best_q, cost + gamma * V[x_next])
            stop_cost = terminal_cost(x, info)  # cost of stopping here (0 if goal, inf otherwise)

            v_new = min(best_q, stop_cost)

            delta = max(delta, abs(V[x] - v_new))# update delta with absolute change in V[x]
            V[x]  = v_new

        if delta < theta:                       # if maximum change is below threshold, we have converged
            break

    # extract stationary policy π(s)
    policy = {}
    value ={}
    for x in X:
        if terminal_cost(x, info) == 0:
            policy[x] = None
            continue

        best_u, best_q = None, float('inf')
        for u in legal_action(x, env, info):
            x_next, cost = transition(x, u, info, env)
            q = cost + gamma * V[x_next]
            if q < best_q:
                best_q, best_u = q, u
        policy[x] = best_u
        value[x]= best_q

   
    x = get_initial_state(env) 
    total_cost = 0 
    optim_act_seq = []
    action_act_seq=[]
    action_meaning = {
        0: "MF",  # Move Forward
        1: "TL",  # Turn Left
        2: "TR",  # Turn Right
        3: "PK",  # Pick Key
        4: "UD",  # Unlock Door
    }
    print("\n===== rollout start =====")
    for t in range(100):
        u = policy[x]
        if u is None:
            print(f"[t={t}] No action found. Terminating.")
            break
        x, step_cost = transition(x, u, info, env)
        total_cost += step_cost
        print(f"[t={t+1}] State: {x} | Action: {action_meaning.get(u, u)}")
        optim_act_seq.append(u)
        action_act_seq.append(action_meaning.get(u, u))
        if (x[0], x[1]) == tuple(info["goal_pos"]):
            print(f"[t={t+1}] Reached GOAL at {x}!")
            break
    print("===== rollout end =====\n")
    print(f"Optimal Action:{action_act_seq}")
    print(f"Cost:={total_cost}")

    return optim_act_seq                     

                
def partA():
    env_path = "/Users/ivanlin328/Desktop/UCSD/Spring 2025/ECE 276B/ECE276B_PR1/starter_code/envs/known_envs/doorkey-8x8-shortcut.env"
    env, info = load_env(env_path)  # load an environment
    seq= doorkey_problem(env,info)  # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0],duration=5.0)  # draw a GIF & save



if __name__ == "__main__":
    #example_use_of_gym_env()
    partA()

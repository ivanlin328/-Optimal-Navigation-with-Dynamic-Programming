from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import os, pickle

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
Direction  = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}  # 0:right 1:down 2:left 3:up
# New heading after TL or TR
LEFT  = {0:3, 1:0, 2:1, 3:2}
RIGHT = {0:1, 1:2, 2:3, 3:0}

KEY_POSITIONS  = [(2,2), (2,3), (1,6)]
GOAL_POSITIONS = [(6,1), (7,3), (6,6)]
DOOR_POSITIONS  = [(5,3), (5,7)]
WIDTH, HEIGHT   = 10, 10


class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)
def get_initial_state(env,info):
    key_idx  = KEY_POSITIONS.index(tuple(info['key_pos']))
    goal_idx = GOAL_POSITIONS.index(tuple(info['goal_pos']))
    x,y = env.unwrapped.agent_pos                 # position (x,y)
    heading = env.unwrapped.agent_dir             # 0~3
    has_key =0                                    # no key at initial
    if info["door_open"][0]:
        door1_status =1
    else:
        door1_status=0   
    if info["door_open"][1]:
        door2_status =1
    else:
        door2_status=0 
    return (key_idx, goal_idx, x, y, heading, has_key, door1_status, door2_status)  # tuple


def legal_action(state):
    key_idx, goal_idx, x, y, h, has_key, d1, d2 = state
    dx,dy = Direction[h]
    tx,ty = x+dx, y+dy
    legal = [TL, TR]

    # out‐of‐bounds or wall =>ilegal
    if not (0<=tx<WIDTH and 0<=ty<HEIGHT):
        return legal
    if tx==5 and (tx,ty) not in DOOR_POSITIONS:  
        return legal

    if (tx,ty) in KEY_POSITIONS:  
         cell='key'
    elif (tx,ty) in GOAL_POSITIONS: 
        cell='goal'
    elif (tx,ty) in DOOR_POSITIONS: 
        cell='door'
    else:                           
        cell='free'

    # Move Forward
    if cell in ('free','key','goal'):
        legal.append(MF)
        
    else: # cell=='door'
        idx = DOOR_POSITIONS.index((tx,ty))
        if idx == 0:
            opened = (d1==1)
        else:
            opened = (d2==1)
        if opened: 
            legal.append(MF)

    # Pick Key
    if cell=='key' and has_key==0:
        legal.append(PK)

    # Unlock Door
    if cell=='door' and has_key==1:
        idx = DOOR_POSITIONS.index((tx,ty))
        if idx ==0:
            locked = (d1==0)
        else:
            locked =(d2==0)
        if locked:
            legal.append(UD)

    return legal

# motion_model
def transition(state, action):
    key_idx, goal_idx, x, y, h, has_key, d1, d2 = state
    next_x_idx, ng_idx = key_idx, goal_idx
    next_x, next_y, next_heading = x, y, h
    next_key, nd1, nd2  = has_key, d1, d2

    if   action==TR:
        next_heading = RIGHT[h]
    elif action==TL:
        next_heading = LEFT[h]
    elif action==MF:
        dx,dy = Direction[h]
        tx,ty = x+dx, y+dy
       
        if 0<=tx<WIDTH and 0<=ty<HEIGHT and not (tx==5 and (tx,ty) not in DOOR_POSITIONS):
            if (tx,ty) not in DOOR_POSITIONS:
                next_x,next_y = tx,ty
            else:
                idx = DOOR_POSITIONS.index((tx,ty))
                opened = (d1 if idx==0 else d2)==1
                if opened:
                    next_x,next_y = tx,ty

    elif action==PK:
        dx,dy = Direction[h]
        tx,ty = x+dx, y+dy
        if (tx,ty)==KEY_POSITIONS[key_idx] and has_key==0:
            next_key = 1

    elif action==UD:
        dx,dy = Direction[h]
        tx,ty = x+dx, y+dy
        if (tx,ty) in DOOR_POSITIONS and has_key==1:
            idx = DOOR_POSITIONS.index((tx,ty))
            if idx==0: 
                nd1 = 1
            else:    
                nd2 = 1

    cost = step_cost(action)
    return (next_x_idx, ng_idx, next_x, next_y, next_heading, next_key, nd1, nd2), cost

def terminal_cost(state):
    """
    If the agent’s (x,y) coordinate matches the goal position, the terminal cost is zero.
    Otherwise (i.e.\ the horizon expires before reaching the goal), 
    we impose a large penalty so that anext_y optimal finite‑horizon policy will strive to reach the goal before time runs out.
    
    """
    key_idx, goal_idx, x, y, h, has_key, d1, d2= state
    if (x,y)==GOAL_POSITIONS[state[1]]:
        return 0
    else:
         return 1000
    
def enumerate_state():
    X = []
    for key_idx in  range(len(KEY_POSITIONS)):        # 0,1,2
      for goal_idx in range(len(GOAL_POSITIONS)):    # 0,1,2
        for x in range(WIDTH):
          for y in range(HEIGHT):
            for h in range(4):
              for has_key in [0,1]:
                for d1 in [0,1]:
                  for d2 in [0,1]:
                    X.append((key_idx, goal_idx, x, y, h, has_key, d1, d2))
    return X

##Dynamic-Programming      
def doorkey_problem(gamma=0.99, theta=1e-6):

    X = enumerate_state()
    V = {s: 0.0 for s in X}
    
    while True:
        delta = 0.0
        for x in X:
            best_q =float('inf')
            for u in legal_action(x):
                x_next,cost = transition(x,u)
                best_q = min(best_q, cost + gamma * V[x_next])
            stop_cost = terminal_cost(x)
            v_new = min(best_q, stop_cost)
            delta = max(delta, abs(V[x] - v_new))
            V[x] = v_new
        if delta < theta:
            break

    # extract stationary policy π(s)
    policy = {}
    value ={}
    for x in X:
        if terminal_cost(x)==0:
            policy[x] = None
            continue
        best_u, best_q = None, float('inf')
        for u in legal_action(x):
            x_next, cost = transition(x, u)
            q = cost + gamma * V[x_next]
            if q < best_q:
                best_q, best_u = q, u
            policy[x] = best_u
            value[x]= best_q
            
    return policy,value

    
def partB():
    policy, value = doorkey_problem() 
    env_folder = "/Users/ivanlin328/Desktop/UCSD/Spring 2025/ECE 276B/ECE276B_PR1/starter_code/envs/random_envs"
    env_files =[os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith(".env")]
    action_meaning = {
        0: "MF",   # Move Forward
        1: "TL",   # Turn Left
        2: "TR",   # Turn Right
        3: "PK",   # Pick Key
        4: "UD",   # Unlock Door
    }
    print("===== SINGLE POLICY) =====")
    for fname in env_files:
        env_path = os.path.join(env_folder, fname)
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        info = {
        "height": env.unwrapped.height,
        "width":  env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "key_pos":   None,
        "door_pos":  [],
        "door_open": [],
        "goal_pos":  None,
        }
        for i in range(info["height"]):
            for j in range(info["width"]):
                cell = env.unwrapped.grid.get(j, i)
                if isinstance(cell, Key):
                    info["key_pos"] = np.array([j, i])
                elif isinstance(cell, Door):
                    info["door_pos"].append(np.array([j, i]))
                    info["door_open"].append(cell.is_open)
                elif isinstance(cell, Goal):
                    info["goal_pos"] = np.array([j, i])
        x = get_initial_state(env, info)
        total_cost = 0  
        act_seq=[] 
        action_act_seq=[]
        action_meaning = {
        0: "MF",  # Move Forward
        1: "TL",  # Turn Left
        2: "TR",  # Turn Right
        3: "PK",  # Pick Key
        4: "UD",  # Unlock Door
    }
        for t in range(100):
            u = policy[x]
            if u is None:
                break
            act_seq.append(u)
            action_act_seq.append(action_meaning.get(u, u))
            x,step_cost = transition(x, u)
            total_cost += step_cost
        print(f"{fname}: Action={(action_act_seq)}, cost={total_cost}") 
        draw_gif_from_seq(act_seq,env)

      
if __name__ == "__main__":
    partB()
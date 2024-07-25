import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

import random

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

# NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 4 # NM

WAYPOINT_DISTANCE_MIN = 70
WAYPOINT_DISTANCE_MAX = 250

D_HEADING = 45 # heading delta from approach fix, one way
D_SPEED = 20 # random value for now

AC_SPD = 150

NM2KM = 1.852

ACTION_FREQUENCY = 10

NUM_AC = 5
NUM_WAYPOINTS = 1

# Final approach fix, north of EHAM in Heiloo
# FIX_LAT = 52.59382191779792 # change to FAF
# FIX_LON = 4.722605450577005

# end of polderbaan EHAM
RWY_LAT = 52.36239301495972
RWY_LON = 4.713195734579777

# Alternative - FAF at a distance
distance_faf_rwy = 400 # NM, initial assumption 
bearing_faf_rwy = 0
FIX_LAT, FIX_LON = fn.get_point_at_distance(RWY_LAT, RWY_LON, distance_faf_rwy, bearing_faf_rwy)

# spawn location ownship
ACLAT_INIT = 52.97843850741256
ACLON_INIT = 4.511017581418151

class AmanEnvS(gym.Env):
    """ 
    Single-agent arrival manager environment - only one aircraft (ownship) is merged into NPC stream of aircraft.
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment


        # Observation space should include info of all aircraft

        self.observation_space = spaces.Dict(
            {
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64),
                "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64),
                "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64),
                "x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64),
                "y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64),
                # "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64),
                "cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                # "faf_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_AC,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0

        self.window = None
        self.clock = None
        self.nac = NUM_AC
        self.wpt_reach = np.zeros(NUM_AC)
        self.rwy_lat = RWY_LAT
        self.rwy_lon = RWY_LON
        self.faf_lat = FIX_LAT
        self.faf_lon = FIX_LON

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wpt_reach = np.zeros(NUM_AC)
        
        bs.traf.reset()

        # ownship
        # random spawn
        bearing_to_pos = random.uniform(-D_HEADING, D_HEADING) # heading radial towards FAF
        distance_to_pos = random.uniform(150,300) # distance to faf 
        # distance_to_pos = 50
        rlat, rlon = fn.get_point_at_distance(FIX_LAT, FIX_LON, distance_to_pos, bearing_to_pos)


        # bs.traf.cre('KL001',actype="A320",acspd=AC_SPD, aclat= 52.97843850741256, aclon=4.511017581418151, achdg=180)
        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD, aclat= rlat, aclon= rlon, achdg=180,acalt=10000)
        bs.stack.stack(f"KL001 addwpt {FIX_LAT} {FIX_LON}")
        bs.stack.stack(f"KL001 dest {RWY_LAT} {RWY_LON}")

        # self._generate_conflicts()
        # self._generate_waypoint()
        self._gen_aircraft()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                # bs.traf.delete(idx)

        return observation, reward, terminated, False, info
    # function to generate the aircraft. so far only randomised 'intruders' are generated.
    def _gen_aircraft(self):
        for i in range(NUM_AC-1):
            # randomise position here
            bearing_to_pos = random.uniform(-D_HEADING, D_HEADING) # heading radial towards FAF
            distance_to_pos = random.uniform(50,300) # distance to faf 
            # distance_to_pos = 50
            lat_ac, lon_ac = fn.get_point_at_distance(FIX_LAT, FIX_LON, distance_to_pos, bearing_to_pos)
            
            self.wpt_lat = FIX_LAT
            self.wpt_lon = FIX_LON
            self.rwy_lat = RWY_LAT
            self.rwy_lon = RWY_LON

            # create aircraft
            bs.traf.cre(f'INT{i}',actype="A320",acspd=AC_SPD,aclat=lat_ac,aclon=lon_ac,achdg=180,acalt=10000)
            bs.stack.stack(f"INT{i} addwpt {FIX_LAT} {FIX_LON}")
            bs.stack.stack(f"INT{i} dest {RWY_LAT} {RWY_LON}")
        bs.stack.stack('reso off')
        return

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.x_difference_speed = []
        self.y_difference_speed = []

        self.waypoint_distance = []
        self.faf_distance = []
        self.wpt_qdr = []
        self.cos_drift = []
        self.sin_drift = []
        self.drift = []
        self.terminated = np.full(NUM_AC,False)

        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_spd = bs.traf.gs[ac_idx]

        for i in range(NUM_AC):
            int_idx = i
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.intruder_distance.append(int_dis * NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)
            # WPT is RWY
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[int_idx], bs.traf.lon[int_idx], self.rwy_lat, self.rwy_lon)
            self.waypoint_distance.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)
            # FAF
            faf_qdr, faf_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[int_idx], bs.traf.lon[int_idx], self.faf_lat, self.faf_lon)
            drift = self.ac_hdg - faf_qdr
            # drift = self.ac_hdg-wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)
            if self.wpt_reach[i]:
                drift = self.ac_hdg - wpt_qdr
                drift = fn.bound_angle_positive_negative_180(drift)
            self.drift.append(drift)
            self.cos_drift.append(np.cos(np.deg2rad(drift)))
            self.sin_drift.append(np.sin(np.deg2rad(drift)))
            self.faf_distance.append(faf_dis * NM2KM)

        observation = {
                "intruder_distance": np.array(self.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_pos": np.array(self.cos_bearing),
                "sin_difference_pos": np.array(self.sin_bearing),
                "x_difference_speed": np.array(self.x_difference_speed)/AC_SPD,
                "y_difference_speed": np.array(self.y_difference_speed)/AC_SPD,
                # "waypoint_distance": np.array(self.waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_drift": np.array([self.cos_drift[0]]),
                "sin_drift": np.array([self.sin_drift[0]]),
                # "faf_distance": np.array(self.faf_distance)/WAYPOINT_DISTANCE_MAX
            }
        # import code
        # code.interact(local=locals())
        return observation
    
    def _get_info(self):
        return {
            "total_reward": self.total_reward,
            "reached": self.wpt_reach,
        }

    def _get_reward(self):
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()

        reward = drift_reward + intrusion_reward
        # reward = intrusion_reward

        if self.wpt_reach[0]==0:
            self.total_reward+=0
            return reward, 0
        else:
            self.total_reward+=10
            return reward, 10
        
        
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                # bs.traf.delete(0)
                bs.stack.stack(f"KL001 delrte")
                f"KL001 addwpt {RWY_LAT} {RWY_LON}"
                reward += REACH_REWARD
                self.wpt_reach[index] = 1
                index+=1
            else:
                reward += 0
                index+=1
        return reward

    def _check_drift(self):
        return abs(np.deg2rad(self.drift[0])) * DRIFT_PENALTY

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_AC-1): # excluding ownship
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                reward += INTRUSION_PENALTY
        return reward
    
    def _check_arrival(self):
        for i in range(NUM_AC): # excluding ownship
            int_idx = i
            _, rwy_dis = bs.tools.geo.kwikqdrdist(self.faf_lat, self.faf_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if rwy_dis < INTRUSION_DISTANCE:
                # reward +=100
                self.terminated[i] = True
        # return reward
    

    def _get_action(self,action):
        action_speed = self.ac_spd + action[0] * D_SPEED
        action_heading = self.ac_hdg + action[1] * D_HEADING
        # only do change IF FAF NOT REACHED
        bs.stack.stack(f"KL001 SPD {action_speed}")
        if self.wpt_reach[0]==1:
            bs.stack.stack(f"KL001 HDG {action_heading}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 700 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235)) 


        # first draw faf at fixed location, only one of these
        for qdr, dis in zip(self.wpt_qdr, self.waypoint_distance):

            circle_x = self.window_width/4.5
            circle_y = self.window_height/2

            pygame.draw.circle(
                canvas, 
                (255,255,255),
                (circle_x,circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                (255,255,255),
                (circle_x,circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
                width = 2
            )
            # draw line to faf
            heading_length = 5000
            heading_end_x = ((np.cos(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
            pygame.draw.line(canvas,
            (0,0,0),
            (circle_x,circle_y),
            (circle_x+heading_end_x/2,circle_y-heading_end_y/2),
            width = 2
            )

            # heading boundary lines
            he_x_l = ((np.cos(np.deg2rad(180+135)) * heading_length)/max_distance)*self.window_width
            he_y_l = ((np.sin(np.deg2rad(180+135)) * heading_length)/max_distance)*self.window_width
            he_x_r = ((np.cos(np.deg2rad(180-135)) * heading_length)/max_distance)*self.window_width
            he_y_r = ((np.sin(np.deg2rad(180-135)) * heading_length)/max_distance)*self.window_width
            pygame.draw.line(canvas,
            (3,252,11),
            (circle_x,circle_y),
            (circle_x+he_x_l/2,circle_y-he_y_l/2),
            width = 4
            )
            pygame.draw.line(canvas,
            (3,252,11),
            (circle_x,circle_y),
            (circle_x+he_x_r/2,circle_y-he_y_r/2),
            width = 4
            )

            # draw rwy start
            rwy_faf_qdr, rwy_faf_dis = bs.tools.geo.kwikqdrdist(self.faf_lat, self.faf_lon, RWY_LAT, RWY_LON)
            x_pos = (circle_x)+(np.cos(np.deg2rad(rwy_faf_qdr))*(rwy_faf_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (circle_y)-(np.sin(np.deg2rad(rwy_faf_qdr))*(rwy_faf_dis * NM2KM)/max_distance)*self.window_height
            heading_length = 5000
            heading_end_x = ((np.cos(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
            pygame.draw.line(canvas,
            (255,255,255),
            (x_pos,y_pos),
            (circle_x+heading_end_x/2,circle_y-heading_end_y/2),
            width = 4
            )
        

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(self.ac_hdg)) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(self.ac_hdg)) * ac_length)/max_distance)*self.window_width

        own_qdr, own_dis = bs.tools.geo.kwikqdrdist(self.faf_lat, self.faf_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        x_pos = (circle_x)+(np.cos(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_distance)*self.window_width
        y_pos = (circle_y)-(np.sin(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_distance)*self.window_height
        pygame.draw.line(canvas,
            (0,0,0),
            (x_pos,y_pos),
            ((x_pos)+heading_end_x/2,(y_pos)-heading_end_y/2),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(self.ac_hdg)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(self.ac_hdg)) * heading_length)/max_distance)*self.window_width

        # pygame.draw.line(canvas,
        #     (0,0,0),
        #     (self.window_width/2,self.window_height/2),
        #     ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
        #     width = 1
        # )

        # draw intruders
        ac_length = 3

        for i in range(1,NUM_AC):
            int_idx = i
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(self.faf_lat, self.faf_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            # int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)
            if i==0:
                color = (252, 43, 28)

            x_pos = (circle_x)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (circle_y)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )


        # # draw target waypoint - Here it is the runway
        # for qdr, dis, reach in zip(self.wpt_qdr, self.waypoint_distance, self.wpt_reach):

        #     circle_x = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
        #     circle_y = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width

        #     color = (255,255,255)

        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
        #         radius = 4,
        #         width = 0
        #     )
            
        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
        #         radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
        #         width = 2
        #     )


        # draw Final Approach fix
        # import code
        # code.interact(local=locals())
        # for qdr, dis in zip(self.wpt_qdr, self.waypoint_distance):

        #     circle_x = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
        #     circle_y = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
        #     # print(dis, qdr)

        #     if reach:
        #         color = (155,155,155)
        #     else:
        #         color = (255,255,255)

        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
        #         radius = 4,
        #         width = 0
        #     )
            
        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
        #         radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
        #         width = 2
        #     )

        # PyGame update
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass
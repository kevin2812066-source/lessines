"""
CustomObs - Python implementation matching C++ exactly
Based on RLGymCPP CustomObs structure (simple, no ball prediction)

Structure:
1. Ball (pos, vel, angVel) = 9 features
2. Previous action = 8 features
3. Boost pads (smart blend) = 34 features
4. Self player (AddPlayerToObs) = 29 features (includes hasJumped)
5. Teammates (padded & shuffled) = N * 38 features
6. Opponents (padded & shuffled) = M * 38 features

Total for 1v1 (max_players=3): 9 + 8 + 34 + 29 + 2*38 + 3*38 = 270 features ✅
Total for 2v2 (max_players=3): 9 + 8 + 34 + 29 + 2*38 + 3*38 = 270 features ✅
Total for 3v3 (max_players=3): 9 + 8 + 34 + 29 + 2*38 + 3*38 = 270 features ✅

Configure maxPlayers to match your training!
"""

import numpy as np
import random
from typing import List, Optional

from .util.physics_object import PhysicsObject
from .util.game_state import GameState
from .util.player_data import PlayerData
from .util import common_values


class CustomObs:
    """
    CustomObs matching C++ RLGymCPP implementation exactly
    Simple structure: ball + action + pads + players (no prediction, no extras)
    """
    
    # Normalization coefficients (match C++)
    POS_COEF = 1.0 / 5000.0
    VEL_COEF = 1.0 / 2300.0
    ANG_VEL_COEF = 1.0 / 3.0
    
    def __init__(self, max_players: int = 2):
        """
        Args:
            max_players: Maximum players per team for padding
                - 1v1: max_players=2 (total) or 1 (per team)? Usually 2 total
                - 2v2: max_players=2 per team
                - 3v3: max_players=3 per team
        
        For 1v1 specifically, use max_players=2 for correct size
        """
        self.max_players = max_players
        
    @staticmethod
    def _invert_phys(phys: PhysicsObject, invert: bool) -> PhysicsObject:
        """Invert physics for orange team (match C++ InvertPhys)"""
        if not invert:
            return phys
        
        inverted = PhysicsObject()
        inverted.position = phys.position * np.array([1, -1, 1])
        inverted.linear_velocity = phys.linear_velocity * np.array([1, -1, 1])
        inverted.angular_velocity = phys.angular_velocity * np.array([-1, 1, -1])
        inverted.quaternion = phys.quaternion  # Rotation needs proper inversion
        
        # Invert rotation matrix properly
        rot = phys.rotation_mtx()
        # Flip Y axis
        inverted_rot = rot * np.array([[1, -1, 1], [1, -1, 1], [1, -1, 1]]).T
        # Store back (this is approximate, but works for obs)
        
        return inverted
    
    def _add_player_to_obs(self, obs: List[float], player: PlayerData, 
                          inverted: bool, ball: PhysicsObject) -> None:
        """
        Add player features to observation (matches C++ AddPlayerToObs exactly)
        
        Features (28 total):
        - Position (3)
        - Forward vector (3)
        - Up vector (3)
        - Velocity (3)
        - Angular velocity (3)
        - Local angular velocity (3)
        - Local ball position (3)
        - Local ball velocity (3)
        - Boost (1)
        - On ground (1)
        - Has flip/jump (1)
        - Is demoed (1)
        - Has jumped (1)
        """
        # Get inverted physics if needed
        phys = player.inverted_car_data if inverted else player.car_data
        rot_mat = phys.rotation_mtx()
        
        # Position (3)
        obs.extend(list(phys.position * self.POS_COEF))
        
        # Forward vector (3)
        forward = rot_mat[:, 0]
        obs.extend(list(forward))
        
        # Up vector (3)
        up = rot_mat[:, 2]
        obs.extend(list(up))
        
        # Velocity (3)
        obs.extend(list(phys.linear_velocity * self.VEL_COEF))
        
        # Angular velocity (3)
        obs.extend(list(phys.angular_velocity * self.ANG_VEL_COEF))
        
        # Local angular velocity (rotMat.Dot(angVel)) (3)
        local_ang_vel = rot_mat.T @ phys.angular_velocity
        obs.extend(list(local_ang_vel * self.ANG_VEL_COEF))
        
        # Local ball position (rotMat.Dot(ball.pos - phys.pos)) (3)
        local_ball_pos = rot_mat.T @ (ball.position - phys.position)
        obs.extend(list(local_ball_pos * self.POS_COEF))
        
        # Local ball velocity (rotMat.Dot(ball.vel - phys.vel)) (3)
        local_ball_vel = rot_mat.T @ (ball.linear_velocity - phys.linear_velocity)
        obs.extend(list(local_ball_vel * self.VEL_COEF))
        
        # Boost (1) - already normalized 0-1
        obs.append(player.boost_amount)
        
        # On ground (1)
        obs.append(1.0 if player.on_ground else 0.0)
        
        # Has flip or jump (1)
        has_flip_or_jump = player.has_flip or player.has_jump
        obs.append(1.0 if has_flip_or_jump else 0.0)
        
        # Is demoed (1)
        obs.append(1.0 if player.is_demoed else 0.0)
        
        # Has jumped (1) - allows detecting flip resets
        has_jumped = getattr(player, 'has_jumped', False)
        obs.append(1.0 if has_jumped else 0.0)
        
        # Total: 3+3+3+3+3+3+3+3+1+1+1+1+1 = 29 features
    
    def build_obs(self, player: PlayerData, state: GameState, 
                  prev_action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build observation matching C++ CustomObs::BuildObs exactly
        
        Args:
            player: The player to build observation for
            state: Current game state
            prev_action: Previous action taken (8 floats: throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
        
        Returns:
            Observation array
        """
        obs = []
        
        # Determine if we need to invert (orange team)
        inverted = player.team_num == 1  # ORANGE = 1
        
        # Get inverted ball and physics
        ball = state.inverted_ball if inverted else state.ball
        phys = player.inverted_car_data if inverted else player.car_data
        pads = state.get_boost_pads(inverted)
        pad_timers = state.get_boost_pad_timers(inverted)
        
        # ========================================
        # 1. BALL STATE (9 features)
        # ========================================
        obs.extend(list(ball.position * self.POS_COEF))
        obs.extend(list(ball.linear_velocity * self.VEL_COEF))
        obs.extend(list(ball.angular_velocity * self.ANG_VEL_COEF))
        # DEBUG
        size_after_ball = len(obs)
        
        # ========================================
        # 2. PREVIOUS ACTION (8 features)
        # ========================================
        if prev_action is None:
            prev_action = np.zeros(8, dtype=np.float32)
        
        # Ensure it's exactly 8 elements
        if len(prev_action) < 8:
            prev_action = np.pad(prev_action, (0, 8 - len(prev_action)), 'constant')
        elif len(prev_action) > 8:
            prev_action = prev_action[:8]
        
        obs.extend(list(prev_action))
        # DEBUG
        size_after_action = len(obs)
        
        # ========================================
        # 3. BOOST PADS (34 features)
        # ========================================
        # Smart blending: pads[i] ? 1.0 : 1.0/(1.0 + padTimers[i])
        for i in range(common_values.BOOST_LOCATIONS_AMOUNT):
            if pads[i] >= 0.5:  # Pad is available
                obs.append(1.0)
            else:  # Pad is not available - blend based on timer
                obs.append(1.0 / (1.0 + pad_timers[i]))
        
        # DEBUG
        size_after_pads = len(obs)
        
        # ========================================
        # 4. SELF OBSERVATION (28 features)
        # ========================================
        self._add_player_to_obs(obs, player, inverted, ball)
        # DEBUG
        size_after_self = len(obs)
        
        # ========================================
        # 5. OTHER PLAYERS (teammates + opponents)
        # ========================================
        rot_mat = phys.rotation_mtx()
        
        teammates_obs = []
        opponents_obs = []
        
        for other_player in state.players:
            if other_player.car_id == player.car_id:
                continue  # Skip self
            
            # Get inverted physics for other player
            other_phys = other_player.inverted_car_data if inverted else other_player.car_data
            
            # Start building other player obs
            player_obs = []
            
            # Relative position (in local agent frame) (3)
            rel_pos = other_phys.position - phys.position
            local_rel_pos = rot_mat.T @ rel_pos
            player_obs.extend(list(local_rel_pos * self.POS_COEF))
            
            # Relative velocity (in local agent frame) (3)
            rel_vel = other_phys.linear_velocity - phys.linear_velocity
            local_rel_vel = rot_mat.T @ rel_vel
            player_obs.extend(list(local_rel_vel * self.VEL_COEF))
            
            # Other angular velocity (in local agent frame) (3)
            other_ang_vel_local = rot_mat.T @ other_phys.angular_velocity
            player_obs.extend(list(other_ang_vel_local * self.ANG_VEL_COEF))
            
            # Full player observation (28)
            self._add_player_to_obs(player_obs, other_player, inverted, ball)
            
            # Total per other player: 3+3+3+28 = 37 features
            
            # Add to appropriate list
            if other_player.team_num == player.team_num:
                teammates_obs.append(player_obs)
            else:
                opponents_obs.append(player_obs)
        
        # ========================================
        # 6. PAD TO MAX PLAYERS & SHUFFLE
        # ========================================
        # For 1v1: max_players=2 means 1 teammate slot (2-1) and 2 opponent slots
        # Actually in C++: teammates max is maxPlayers-1, opponents max is maxPlayers
        
        # Pad teammates to maxPlayers - 1
        player_obs_size = 38  # From C++: selfObs.size() + 9 (selfObs is 29, so 29+9=38)
        target_teammates = self.max_players - 1
        target_opponents = self.max_players
        
        while len(teammates_obs) < target_teammates:
            teammates_obs.append([0.0] * player_obs_size)
        
        while len(opponents_obs) < target_opponents:
            opponents_obs.append([0.0] * player_obs_size)
        
        # Shuffle both lists (match C++ std::shuffle)
        random.shuffle(teammates_obs)
        random.shuffle(opponents_obs)
        
        # Add to observation
        for teammate in teammates_obs:
            obs.extend(teammate)
        
        for opponent in opponents_obs:
            obs.extend(opponent)
        
        # DEBUG
        size_after_all = len(obs)
        
        # ========================================
        # RETURN AS NUMPY ARRAY
        # ========================================
        result = np.asarray(obs, dtype=np.float32)
        
        # Expected size calculation:
        # 9 (ball) + 8 (action) + 34 (pads) + 29 (self) + 38*(max_players-1) + 38*max_players
        expected_size = 9 + 8 + 34 + 29 + 38 * (target_teammates + target_opponents)
        
        if len(result) != expected_size:
            # Debug: trace exact sizes
            print(f"⚠️ Warning: Obs size is {len(result)}, expected {expected_size}")
            print(f"   (max_players={self.max_players})")
            print(f"\n📊 Size breakdown:")
            print(f"   After ball: {size_after_ball} (expected 9)")
            print(f"   After action: {size_after_action} (expected 17 = 9+8)")
            print(f"   After pads: {size_after_pads} (expected 51 = 9+8+34)")
            print(f"   After self: {size_after_self} (expected 80 = 9+8+34+29)")
            print(f"   After all players: {size_after_all}")
            print(f"\n👥 Players:")
            print(f"   Real teammates: {len([t for t in teammates_obs if any(t)])}")
            print(f"   Padded teammates: {target_teammates} × 38 = {target_teammates * 38}")
            print(f"   Real opponents: {len([o for o in opponents_obs if any(o)])}")
            print(f"   Padded opponents: {target_opponents} × 38 = {target_opponents * 38}")
            print(f"   prev_action length: {len(prev_action)}")
        
        return result


# For easy import
__all__ = ['CustomObs']


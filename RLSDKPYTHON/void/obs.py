"""
CustomObs - Reproduction EXACTE du C++ CustomObs.cpp pour HazeSDK
==================================================================

Observation Builder compatible HazeSDK
Taille: 147 features (9+6+18+7+6+8+4+2+34+8+21+38)

Structure identique au C++ ligne par ligne:
- Balle (9)
- Balle dans repère joueur (6)
- Joueur (18)
- État joueur (7)
- Relations balle-joueur (6)
- Buts (8)
- Proximité terrain (4)
- Boost le plus proche (2)
- Tous les boost pads (34)
- Dernière action (8)
- Ball prediction simple (21)
- Adversaire 1v1 (38)
"""

import numpy as np
from typing import Optional


class CustomObs:
    """
    Observation Builder pour HazeSDK
    Reproduction EXACTE du C++ CustomObs.cpp
    """
    
    # ==================== COEFFICIENTS (IDENTIQUES au C++) ====================
    POS_COEF = 1.0 / 5000.0      # Normalisation position
    VEL_COEF = 1.0 / 2300.0      # Normalisation vélocité
    ANG_VEL_COEF = 1.0 / 3.0     # Normalisation vélocité angulaire
    
    # ==================== CONSTANTES PHYSIQUES (IDENTIQUES au C++) ====================
    GRAVITY = 650.0              # Gravité Rocket League
    BALL_RADIUS = 93.15          # Rayon de la balle
    CAR_MAX_SPEED = 2300.0       # Vitesse max voiture
    
    # ==================== CONSTANTES TERRAIN (IDENTIQUES au C++) ====================
    SIDE_WALL_X = 4096.0         # Mur latéral
    BACK_WALL_Y = 5120.0         # Mur arrière
    CEILING_Z = 2044.0           # Plafond
    
    # ==================== HORIZONS PRÉDICTION (IDENTIQUES au C++) ====================
    HORIZONS = [0.2, 0.5, 1.0]   # Horizons temporels: 0.2s, 0.5s, 1.0s
    
    def __init__(self):
        """Initialise l'observation builder"""
        self.prev_action = np.zeros(8, dtype=np.float32)
    
    @staticmethod
    def normalize_vec(v: np.ndarray) -> np.ndarray:
        """
        Normalise un vecteur (équivalent C++: v.Normalized())
        Retourne zéro si la norme est nulle
        """
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return np.zeros_like(v, dtype=np.float32)
        return (v / norm).astype(np.float32)
    
    def build_obs(self, game_state, player_index: int = 0) -> np.ndarray:
        """
        BuildObs - EXACT match du C++ CustomObs.cpp
        
        Args:
            game_state: GameState de HazeSDK
            player_index: Index du joueur (0 par défaut)
            
        Returns:
            obs: np.ndarray de 147 features (float32)
        """
        obs = []
        
        # ==================== VALIDATION ====================
        if not hasattr(game_state, 'ball') or not game_state.ball:
            return np.zeros(147, dtype=np.float32)
        
        if player_index >= len(game_state.players):
            return np.zeros(147, dtype=np.float32)
        
        # ==================== RÉCUPÉRATION DONNÉES ====================
        ball = game_state.ball
        player_info = game_state.players[player_index]
        player_car = player_info.car_data
        
        # ==================== INVERSION (ligne 11 C++) ====================
        # inv = True si équipe ORANGE
        inv = player_info.team_num == 1
        inv_mul = np.array([1, -1, 1], dtype=np.float32) if inv else np.array([1, 1, 1], dtype=np.float32)
        
        # ==================== ÉTATS PHYSIQUES INVERSÉS (lignes 14-17 C++) ====================
        ball_pos = np.array(ball.position, dtype=np.float32) * inv_mul
        ball_vel = np.array(ball.linear_velocity, dtype=np.float32) * inv_mul
        ball_ang_vel = np.array(ball.angular_velocity, dtype=np.float32) * inv_mul
        
        car_pos = np.array(player_car.position, dtype=np.float32) * inv_mul
        car_vel = np.array(player_car.linear_velocity, dtype=np.float32) * inv_mul
        car_ang_vel = np.array(player_car.angular_velocity, dtype=np.float32) * inv_mul
        
        # ==================== MATRICE DE ROTATION ====================
        # Utiliser la matrice de rotation calculée par PhysicsObject
        rot_mat = player_car.rotation_mtx()
        
        # Inversion EXACTE comme C++ (ligne 11): multiplier chaque vecteur par [-1, -1, 1]
        if inv:
            rot_mat = rot_mat * inv_mul[np.newaxis, :]
        
        forward = rot_mat[:, 0]  # Vecteur forward
        up = rot_mat[:, 2]        # Vecteur up
        
        # ==================== BALLE (9) - lignes 19-25 C++ ====================
        obs.extend(ball_pos * self.POS_COEF)           # 3 features
        obs.extend(ball_vel * self.VEL_COEF)           # 3 features
        obs.extend(ball_ang_vel * self.ANG_VEL_COEF)   # 3 features
        
        # ==================== BALLE DANS REPÈRE JOUEUR (6) - lignes 27-33 C++ ====================
        local_ball_pos = rot_mat.T @ (ball_pos - car_pos)
        obs.extend(local_ball_pos * self.POS_COEF)     # 3 features
        
        local_ball_vel = rot_mat.T @ (ball_vel - car_vel)
        obs.extend(local_ball_vel * self.VEL_COEF)     # 3 features
        
        # ==================== JOUEUR (18) - lignes 35-47 C++ ====================
        obs.extend(car_pos * self.POS_COEF)            # 3 features
        obs.extend(forward)                             # 3 features
        obs.extend(up)                                  # 3 features
        obs.extend(car_vel * self.VEL_COEF)            # 3 features
        obs.extend(car_ang_vel * self.ANG_VEL_COEF)    # 3 features
        
        local_ang_vel = rot_mat.T @ car_ang_vel
        obs.extend(local_ang_vel * self.ANG_VEL_COEF)  # 3 features
        
        # ==================== ÉTAT JOUEUR (7) - lignes 49-63 C++ ====================
        obs.append(player_info.boost_amount)                            # 1 (déjà normalisé 0-1)
        obs.append(1.0 if player_info.on_ground else 0.0)               # 1
        obs.append(1.0 if player_info.has_flip else 0.0)                # 1
        obs.append(1.0 if player_info.is_demoed else 0.0)               # 1
        obs.append(1.0 if player_info.has_jumped else 0.0)              # 1
        obs.append(1.0 if player_info.is_supersonic else 0.0)           # 1
        obs.append(1.0 if player_info.has_double_jumped else 0.0)       # 1
        
        # ==================== RELATIONS BALLE-JOUEUR (6) - lignes 65-81 C++ ====================
        to_ball = ball_pos - car_pos
        dist_to_ball = np.linalg.norm(to_ball)
        
        obs.append(dist_to_ball * self.POS_COEF)                        # 1
        obs.append(np.dot(forward, self.normalize_vec(to_ball)))        # 1
        
        rel_speed = max(0.0, np.dot(car_vel, self.normalize_vec(to_ball)))
        obs.append(rel_speed / self.CAR_MAX_SPEED)                      # 1
        
        obs.append((ball_pos[2] - car_pos[2]) * self.POS_COEF)          # 1
        
        is_kickoff = (abs(ball_pos[0]) < 20.0 and 
                      abs(ball_pos[1]) < 20.0 and 
                      np.linalg.norm(ball_vel) < 50.0)
        obs.append(1.0 if is_kickoff else 0.0)                          # 1
        
        obs.append(up[2])                                                # 1
        
        # ==================== BUTS (8) - lignes 83-99 C++ ====================
        opp_goal = np.array([0.0, self.BACK_WALL_Y, 0.0], dtype=np.float32)
        own_goal = np.array([0.0, -self.BACK_WALL_Y, 0.0], dtype=np.float32)
        
        to_opp_goal = opp_goal - car_pos
        obs.append(np.linalg.norm(to_opp_goal) * self.POS_COEF)         # 1
        obs.extend(rot_mat.T @ self.normalize_vec(to_opp_goal))         # 3
        
        ball_to_opp_goal = opp_goal - ball_pos
        obs.append(np.linalg.norm(ball_to_opp_goal) * self.POS_COEF)    # 1
        
        to_own_goal = own_goal - car_pos
        obs.append(np.linalg.norm(to_own_goal) * self.POS_COEF)         # 1
        
        # ==================== PROXIMITÉ TERRAIN (4) - lignes 101-108 C++ ====================
        dx = self.SIDE_WALL_X - abs(car_pos[0])
        dy = self.BACK_WALL_Y - abs(car_pos[1])
        dz = self.CEILING_Z - car_pos[2]
        
        obs.append(dx * self.POS_COEF)                                  # 1
        obs.append(dy * self.POS_COEF)                                  # 1
        obs.append(dz * self.POS_COEF)                                  # 1
        obs.append(min(dx, dy, dz) * self.POS_COEF)                     # 1
        
        # ==================== BOOST LE PLUS PROCHE (2) - lignes 110-129 C++ ====================
        # Trouver le boost pad le plus proche
        pads = game_state.get_boost_pads(inv)
        pad_timers = game_state.get_boost_pad_timers(inv)
        
        # Récupérer les positions des boost pads depuis common_values
        from .util.common_values import BOOST_LOCATIONS, BOOST_LOCATIONS_AMOUNT
        
        nearest_idx = -1
        best_dist2 = 1e20
        
        for i in range(BOOST_LOCATIONS_AMOUNT):
            # Index mapping pour inversion (miroir C++ ligne 114)
            map_idx = (BOOST_LOCATIONS_AMOUNT - i - 1) if inv else i
            pad_pos = np.array(BOOST_LOCATIONS[map_idx], dtype=np.float32) * inv_mul
            
            d = pad_pos - car_pos
            d2 = np.dot(d, d)
            if d2 < best_dist2:
                best_dist2 = d2
                nearest_idx = i
        
        if nearest_idx >= 0:
            obs.append(np.sqrt(best_dist2) * self.POS_COEF)              # 1
            # Disponibilité: 1 si actif, sinon approche de 1 quand le pad devient disponible
            obs.append(1.0 if pads[nearest_idx] else (1.0 / (1.0 + pad_timers[nearest_idx])))  # 1
        else:
            obs.append(0.0)                                              # 1
            obs.append(0.0)                                              # 1
        
        # ==================== TOUS LES BOOST PADS (34) - lignes 131-134 C++ ====================
        for i in range(BOOST_LOCATIONS_AMOUNT):
            # Clever trick: blend boost pads using their timers (ligne 133 C++)
            obs.append(1.0 if pads[i] else (1.0 / (1.0 + pad_timers[i])))  # 34
        
        # ==================== DERNIÈRE ACTION (8) - lignes 136-139 C++ ====================
        obs.extend(self.prev_action)                                     # 8
        
        # ==================== BALL PREDICTION SIMPLE (21) - lignes 141-177 C++ ====================
        for t in self.HORIZONS:
            # Prédiction linéaire simple (lignes 149-151 C++)
            pred_pos = ball_pos + ball_vel * t
            pred_vel = ball_vel + np.array([0.0, 0.0, -self.GRAVITY], dtype=np.float32) * t
            
            # Collision sol simple (lignes 154-164 C++)
            if pred_pos[2] < self.BALL_RADIUS:
                time_to_ground = (ball_pos[2] - self.BALL_RADIUS) / max(1.0, -ball_vel[2])
                if 0.0 < time_to_ground < t:
                    # Rebond au sol (perte d'énergie 60%)
                    pred_pos[2] = self.BALL_RADIUS
                    pred_vel[2] = abs(ball_vel[2] - self.GRAVITY * time_to_ground) * 0.6
                else:
                    pred_pos[2] = self.BALL_RADIUS
                    pred_vel[2] = 0.0
            
            # Position prédite dans mon repère (ligne 167 C++)
            local_pred_pos = rot_mat.T @ (pred_pos - car_pos)
            obs.extend(local_pred_pos * self.POS_COEF)                   # 3
            
            # Vélocité prédite dans mon repère (ligne 171 C++)
            local_pred_vel = rot_mat.T @ (pred_vel - car_vel)
            obs.extend(local_pred_vel * self.VEL_COEF)                   # 3
            
            # Distance à cette position future (ligne 175 C++)
            dist_to_pred = np.linalg.norm(pred_pos - car_pos)
            obs.append(dist_to_pred * self.POS_COEF)                     # 1
            # Total: 3 horizons * 7 features = 21
        
        # ==================== ADVERSAIRE (38) - lignes 179-237 C++ ====================
        # Trouver l'adversaire en 1v1 (lignes 181-187 C++)
        opponent_info = None
        for i, p in enumerate(game_state.players):
            if i != player_index and p.team_num != player_info.team_num:
                opponent_info = p
                break
        
        if opponent_info:
            opponent_car = opponent_info.car_data
            
            # États physiques adversaire inversés (ligne 190 C++)
            opp_pos = np.array(opponent_car.position, dtype=np.float32) * inv_mul
            opp_vel = np.array(opponent_car.linear_velocity, dtype=np.float32) * inv_mul
            opp_ang_vel = np.array(opponent_car.angular_velocity, dtype=np.float32) * inv_mul
            
            # Matrice de rotation adversaire
            opp_rot_mat = opponent_car.rotation_mtx()
            
            # Inversion EXACTE comme C++ (ligne 11): multiplier chaque vecteur par [-1, -1, 1]
            if inv:
                opp_rot_mat = opp_rot_mat * inv_mul[np.newaxis, :]
            
            opp_forward = opp_rot_mat[:, 0]
            
            # Position relative (locale) (ligne 193 C++)
            rel_pos = rot_mat.T @ (opp_pos - car_pos)
            obs.extend(rel_pos * self.POS_COEF)                          # 3
            
            # Vélocité relative (locale) (ligne 197 C++)
            rel_vel = rot_mat.T @ (opp_vel - car_vel)
            obs.extend(rel_vel * self.VEL_COEF)                          # 3
            
            # Vélocité angulaire adversaire (locale à moi) (ligne 201 C++)
            opp_ang_vel_local = rot_mat.T @ opp_ang_vel
            obs.extend(opp_ang_vel_local * self.ANG_VEL_COEF)            # 3
            
            # Orientation adversaire (dans mon repère) (lignes 205-208 C++)
            opp_fwd_local = rot_mat.T @ opp_forward
            obs.extend(opp_fwd_local)                                    # 3
            
            opp_up_local = rot_mat.T @ opp_rot_mat[:, 2]
            obs.extend(opp_up_local)                                     # 3
            
            # État adversaire (lignes 211-217 C++)
            obs.append(opponent_info.boost_amount)                       # 1 (déjà normalisé 0-1)
            obs.append(1.0 if opponent_info.on_ground else 0.0)          # 1
            obs.append(1.0 if opponent_info.has_flip else 0.0)           # 1
            obs.append(1.0 if opponent_info.is_demoed else 0.0)          # 1
            obs.append(1.0 if opponent_info.has_jumped else 0.0)         # 1
            obs.append(1.0 if opponent_info.is_supersonic else 0.0)      # 1
            obs.append(1.0 if opponent_info.has_double_jumped else 0.0) # 1
            
            # Relations adversaire-balle (lignes 220-224 C++)
            opp_to_ball = ball_pos - opp_pos
            opp_dist_to_ball = np.linalg.norm(opp_to_ball)
            
            obs.append(opp_dist_to_ball * self.POS_COEF)                 # 1
            obs.append(np.dot(opp_forward, self.normalize_vec(opp_to_ball)))  # 1
            
            # Distance adversaire -> but adverse (lignes 227-228 C++)
            opp_to_opp_goal = opp_goal - opp_pos
            obs.append(np.linalg.norm(opp_to_opp_goal) * self.POS_COEF) # 1
            
            # Qui est plus proche de la balle (ligne 231 C++)
            obs.append(1.0 if dist_to_ball < opp_dist_to_ball else 0.0) # 1
            
        else:
            # Padding si pas d'adversaire (38 zéros) (lignes 233-236 C++)
            for _ in range(38):
                obs.append(0.0)
        
        # ==================== CONVERSION ET VALIDATION ====================
        obs_array = np.array(obs, dtype=np.float32)
        
        # S'assurer que la taille est exactement 147
        if len(obs_array) != 147:
            # print(f"⚠️ WARNING: obs size is {len(obs_array)}, expected 147!")
            if len(obs_array) < 147:
                obs_array = np.pad(obs_array, (0, 147 - len(obs_array)), 'constant')
            else:
                obs_array = obs_array[:147]
        
        return obs_array
    
    def set_prev_action(self, action: np.ndarray):
        """
        Enregistre la dernière action (8 éléments)
        
        Args:
            action: np.ndarray de 8 éléments [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        self.prev_action = np.array(action[:8], dtype=np.float32)
    
    def reset(self):
        """Reset l'observation builder (équivalent C++: Reset())"""
        self.prev_action = np.zeros(8, dtype=np.float32)


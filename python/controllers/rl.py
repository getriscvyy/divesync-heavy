import os
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from ml.reward import compute_reward
from ml.td3_env_spec import DiveSyncEnvSpec, OBS_LOW, OBS_HIGH

# Stupid warning flooding my console removal
import warnings

warnings.filterwarnings(
    "ignore", 
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

class RLController:
    def __init__(self, stroke, w1=10.0, w2=150.0, w3=0.0, w4=0.0, tau=0.3,
                 exploration_sigma=0.02, gradient_steps=1000, batch_size=100,
                 online_gradient_steps=0, train_every_n_ticks=100,
                 max_action_delta=0.03, decision_interval=20, model_dir="python/ml",
                 random_warmup_decisions=50, stuck_decisions_threshold=30,
                 stuck_kick_decisions=15, stuck_velocity_eps=0.002,
                 stuck_error_threshold=0.10):
        self._stroke = float(stroke)
        self._w1, self._w2, self._w3, self._w4, self._tau = w1, w2, w3, w4, tau
        self._exploration_sigma = exploration_sigma
        self._gradient_steps = gradient_steps
        self._batch_size = batch_size
        self._online_gradient_steps = online_gradient_steps
        self._train_every_n_ticks = train_every_n_ticks
        self._max_action_delta = max_action_delta

        # Every dive so far plateaus once the actuator lands on one side of
        # the neutral-buoyancy point (~48mm) -- zero-mean exploration noise
        # is a random walk that mostly cancels out rather than making
        # directed progress back across it. If depth hasn't moved AND is
        # still far from setpoint for this many consecutive decisions, force
        # a directed push toward the opposite side instead of waiting on
        # noise to (maybe) drift there. Gated on error too, not just
        # velocity, so this never fires while it's correctly holding depth
        # at the setpoint (which is also zero-velocity, by design).
        self._stuck_decisions_threshold = stuck_decisions_threshold
        self._stuck_kick_decisions = stuck_kick_decisions
        self._stuck_velocity_eps = stuck_velocity_eps
        self._stuck_error_threshold = stuck_error_threshold
        self._stuck_decision_count = 0
        self._kick_decisions_remaining = 0

        # Save reward
        self.reward = None

        # RL decision timing
        self._decision_interval = decision_interval
        self._tick_count = 0

        # With no BC/warm-start, actor and critics both start from random
        # init. For this many initial decisions, actions are sampled
        # uniformly at random (instead of from the untrained actor) so the
        # critic gets diverse, decorrelated transitions before it's trained
        # on anything -- without this, online training fits the critic to a
        # handful of near-identical early actions and the policy can collapse
        # to a near-constant output (see README Known Issues).
        self._random_warmup_ticks = decision_interval * random_warmup_decisions

        # Last commanded action
        self._current_action_mm = self._stroke / 2.0
        self._model_path = f"{model_dir}/td3_model"
        self._buffer_path = f"{model_dir}/td3_replay_buffer.pkl"

        env = DiveSyncEnvSpec(stroke)

        if os.path.exists(self._model_path + ".zip"):
            self.model = TD3.load(self._model_path, env=env)
            if os.path.exists(self._buffer_path):
                self.model.load_replay_buffer(self._buffer_path)
            print("[RL] Loaded existing TD3 model")
        else:
            self.model = TD3("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
            print("[RL] Initialized fresh TD3 model (fully online, no warm-start)")

        self.model.set_logger(configure(None, []))

        # Previous transition
        self._prev_state_vec = None
        self._prev_action_norm = None
        self._prev_action_delta = 0.0

    @staticmethod
    def _state_to_array(state):
        return np.array([state.depth_m, state.depth_setpoint_m, state.depth_error_m, state.velocity_mps], dtype=np.float32)

    @staticmethod
    def _normalize_state(raw_state):
        return (2.0 * (raw_state - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1.0).astype(np.float32)

    def get_command(self, state):
        # Convert current state
        raw_state = self._state_to_array(state)
        scaled_state = self._normalize_state(raw_state)

        # Store previous experience
        if self._prev_state_vec is not None:
            self.reward = compute_reward(state, self._w1, self._w2, self._w3, self._tau, self._prev_action_delta, self._w4)
            print(self.reward)
            self.model.replay_buffer.add(
                self._prev_state_vec.reshape(1, -1), scaled_state.reshape(1, -1),
                self._prev_action_norm.reshape(1, -1), np.array([self.reward], dtype=np.float32),
                np.array([False]), [{}]
            )

        # Train periodically (skip while still in the random warm-up phase)
        self._tick_count += 1
        if (self._tick_count > self._random_warmup_ticks
                and self._online_gradient_steps > 0
                and self._tick_count % self._train_every_n_ticks == 0
                and self.model.replay_buffer.size() >= self._batch_size):
            self.model.train(gradient_steps=self._online_gradient_steps, batch_size=self._batch_size)

        # Only update RL action every decision interval
        if self._tick_count % self._decision_interval == 0:
            # Track how long depth has been essentially unchanged while still
            # meaningfully off-setpoint (holding steady AT the setpoint is
            # the goal, not a stuck state, so it's excluded here)
            is_off_target_and_still = (
                abs(state.velocity_mps) <= self._stuck_velocity_eps
                and abs(state.depth_error_m) > self._stuck_error_threshold
            )
            if not is_off_target_and_still:
                self._stuck_decision_count = 0
            else:
                self._stuck_decision_count += 1
                if (self._stuck_decision_count >= self._stuck_decisions_threshold
                        and self._kick_decisions_remaining <= 0):
                    self._kick_decisions_remaining = self._stuck_kick_decisions
                    print(f"[RL] Stuck for {self._stuck_decision_count} decisions -- forcing a push across neutral buoyancy")

            if self._tick_count <= self._random_warmup_ticks:
                # Random-walk target during warm-up; still passes through the
                # same delta clamp below, so hardware motion stays smooth.
                normalized_action = np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            elif self._kick_decisions_remaining > 0:
                # Drive hard toward the opposite side of neutral (normalized
                # 0 ~= actuator mid-stroke, close to ACTUATOR_EQUILIBRIUM) so
                # the delta clamp below produces steady, directed motion
                # across the boundary instead of a coin-flip random walk.
                current = self._prev_action_norm[0] if self._prev_action_norm is not None else 0.0
                target = -1.0 if current >= 0 else 1.0
                normalized_action = np.array([target], dtype=np.float32)
                self._kick_decisions_remaining -= 1
                self._stuck_decision_count = 0
            else:
                state_tensor = torch.tensor(scaled_state.reshape(1, -1), dtype=torch.float32)

                with torch.no_grad():
                    normalized_action = self.model.policy.actor(state_tensor).numpy()[0]

                # Exploration noise
                noise = np.random.normal(0, self._exploration_sigma, size=normalized_action.shape).astype(np.float32)
                normalized_action = np.clip(normalized_action + noise, -1.0, 1.0)

            # Limit action changes
            action_delta = 0.0
            if self._prev_action_norm is not None:
                raw_delta = normalized_action - self._prev_action_norm
                clipped_delta = np.clip(raw_delta, -self._max_action_delta, self._max_action_delta)
                normalized_action = self._prev_action_norm + clipped_delta
                action_delta = float(clipped_delta[0])

            # Convert [-1,1] to actuator mm
            action_mm = self._stroke * (normalized_action[0] + 1.0) / 2.0
            action_mm = float(np.clip(action_mm, 0, self._stroke))
            self._current_action_mm = action_mm

            # Save transition info
            self._prev_state_vec = scaled_state.copy()
            self._prev_action_norm = normalized_action.copy()
            self._prev_action_delta = action_delta

        return self._current_action_mm

    def finalize(self, last_state=None):
        if self._prev_state_vec is not None:
            if last_state is not None:
                final_state = self._normalize_state(self._state_to_array(last_state))
                self.reward = compute_reward(last_state, self._w1, self._w2, self._w3, self._tau, self._prev_action_delta, self._w4)
            else:
                final_state = self._prev_state_vec
                self.reward = 0.0

            self.model.replay_buffer.add(
                self._prev_state_vec.reshape(1, -1), final_state.reshape(1, -1),
                self._prev_action_norm.reshape(1, -1), np.array([self.reward], dtype=np.float32),
                np.array([True]), [{}]
            )

        if self.model.replay_buffer.size() >= self._batch_size:
            self.model.train(gradient_steps=self._gradient_steps, batch_size=self._batch_size)
        else:
            print("[RL] Not enough transitions to train")

        self.model.save(self._model_path)
        self.model.save_replay_buffer(self._buffer_path)
        print(f"[RL] Saved model + buffer ({self.model.replay_buffer.size()} transitions)")

"""
teiresias.particle_filter
=========================

The Nothung particle filter -- sequential Monte Carlo inference over
(regime, intensity, duration) triples on financial time series.

Each particle carries a discrete regime label, a continuous intensity
:math:`\\eta \\in [0, 1]`, and a duration counter (days spent in the
current regime).  Particles propagate through HSMM transitions
(:mod:`teiresias.transitions`), are weighted by the observation model
(:mod:`teiresias.observation`), and are resampled when the effective
sample size drops below a threshold.

A small fraction of particles is replaced at every step by an "informed
injection" -- particles drawn from the regimes most consistent with the
current observation.  This is the kidnapped-robot strategy from
probabilistic robotics: a safety net against catastrophic divergence
when the filter has lost track.

Reported quantities at each step:

* ``regime_probs``: posterior probability per regime (sums to 1).
* ``eta``:        importance-weighted expectation of the intensity.
* ``ess``:         effective sample size (diagnostic).
* ``duration``:    weighted mean of the duration counter.

References
----------
Thrun, Burgard, Fox -- Probabilistic Robotics, MIT Press 2005.
Doucet, Godsill, Andrieu -- On sequential Monte Carlo sampling methods,
Statistics and Computing 10, 2000.
"""

from __future__ import annotations

import numpy as np

from .transitions import get_transition_probs, N_REGIMES


class NothungParticleFilter:
    """
    Particle filter over (regime, eta, duration) on financial features.

    Parameters
    ----------
    n_particles : int
        Number of particles (default 1000).
    n_regimes : int
        Number of discrete regimes (default 7).
    injection_fraction : float
        Fraction of particles replaced by informed-injection at each step
        (default 0.05).  Set to 0 to disable.
    ess_threshold : float
        Resample when ESS / N falls below this fraction (default 0.5).
    obs_sigma : float
        Bandwidth of the observation likelihood
        :math:`p(y | r) \\propto \\exp(-d_r^2 / (2 \\sigma^2))`.

    Attributes
    ----------
    regimes : np.ndarray, shape (n_particles,)
    etas : np.ndarray, shape (n_particles,)
    durations : np.ndarray, shape (n_particles,)
    weights : np.ndarray, shape (n_particles,)
        Normalised importance weights.
    history : list[dict]
        Per-step diagnostics; each entry has keys
        ``eta``, ``duration``, ``ess``, ``regime_probs``.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        n_regimes: int = N_REGIMES,
        injection_fraction: float = 0.05,
        ess_threshold: float = 0.5,
        obs_sigma: float = 2.0,
        likelihood_mode: str = "kNN",
        rf_lambda: float = 0.5,
        rf_temperature: float = 2.0,
        rf_floor: float = 0.005,
    ):
        """
        Parameters
        ----------
        likelihood_mode : {'kNN', 'mix', 'RF'}
            Observation likelihood backend.
            * ``'kNN'`` (default, backward compatible): per-regime likelihood
              ``exp(-d_r^2 / (2*sigma^2))`` using k-NN distance to cluster cores.
            * ``'mix'``: convex combination of kNN and RF posteriors:
              ``L(r) = lambda * L_kNN(r) + (1-lambda) * L_RF(r)``.
              Following Grisetti et al.'s informed-likelihood approach in
              robotic mapping: a discriminative classifier (RF, "smart sensor")
              augments the generative geometric likelihood (kNN to cores).
            * ``'RF'``: pure RF likelihood (kNN ignored). Diagnostic mode only;
              produces high false-positive rate in flat-volatility regimes.
        rf_lambda : float
            Mixture weight on the kNN component when ``likelihood_mode='mix'``.
            ``1.0`` recovers ``'kNN'`` mode; ``0.0`` recovers ``'RF'`` mode.
            Default 0.5 (equal weighting).
        rf_temperature : float
            Temperature used to soften RF posteriors before they enter the
            likelihood: ``L_RF(r) ~ p_RF(r)^(1/T)``. Larger ``T`` -> flatter
            distribution (less peaked). Default 2.0 prevents single-class
            collapse when RF is over-confident.
        rf_floor : float
            Minimum value clipped onto each per-regime RF likelihood after
            softening. Prevents particles in low-probability regimes from
            getting zero weight. Default 0.005.
        """
        self.N = n_particles
        self.K = n_regimes
        self.injection_fraction = injection_fraction
        self.ess_threshold = ess_threshold
        self.obs_sigma = obs_sigma
        if likelihood_mode not in ("kNN", "mix", "RF"):
            raise ValueError(
                f"likelihood_mode must be 'kNN', 'mix', or 'RF'; got {likelihood_mode!r}"
            )
        self.likelihood_mode = likelihood_mode
        self.rf_lambda = float(rf_lambda)
        self.rf_temperature = float(rf_temperature)
        self.rf_floor = float(rf_floor)

        # State tensors -- allocated by initialize()
        self.regimes: np.ndarray | None = None
        self.etas: np.ndarray | None = None
        self.durations: np.ndarray | None = None
        self.weights: np.ndarray | None = None

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, regime_probs: np.ndarray | None = None) -> None:
        """
        Sample the initial particle population.

        Parameters
        ----------
        regime_probs : np.ndarray, optional
            Prior over regimes, length ``n_regimes``.  If None, uniform.
        """
        if regime_probs is None:
            regime_probs = np.ones(self.K) / self.K
        regime_probs = np.asarray(regime_probs)
        regime_probs = regime_probs / regime_probs.sum()

        self.regimes = np.random.choice(self.K, size=self.N, p=regime_probs)
        # Initial intensity: uniform on [0, 0.3] -- low but not zero
        self.etas = np.random.uniform(0.0, 0.3, size=self.N)
        # Initial duration: small geometric draw, so particles are not all
        # synchronised at d=0
        self.durations = np.random.geometric(p=0.1, size=self.N).astype(float)
        self.weights = np.ones(self.N) / self.N
        self.history = []

    # ------------------------------------------------------------------
    # One filter step
    # ------------------------------------------------------------------

    def step(self, distances: dict[int, float],
             rf_probs: np.ndarray | None = None) -> np.ndarray:
        """
        One full filter step: propagate, update, eta-update, optionally
        inject and resample.

        Parameters
        ----------
        distances : dict
            ``{regime_id: distance}`` from the observation model.
        rf_probs : np.ndarray, optional
            Per-regime posterior from the random-forest classifier, shape
            ``(n_regimes,)``. Required when ``likelihood_mode != 'kNN'``.
            Ignored otherwise. Should sum to 1.0.

        Returns
        -------
        np.ndarray
            Posterior regime probabilities, shape ``(n_regimes,)``.
        """
        self.propagate()
        self.update(distances, rf_probs=rf_probs)
        self.update_eta(distances)

        # Compute ESS; if too low, do informed injection then resample.
        ess = self.effective_sample_size()
        if ess < self.ess_threshold * self.N:
            if self.injection_fraction > 0:
                self.informed_injection(distances)
            self.systematic_resample()

        # Aggregate posterior probabilities per regime
        regime_probs = np.zeros(self.K)
        for r in range(self.K):
            mask = self.regimes == r
            regime_probs[r] = self.weights[mask].sum()

        # Diagnostics
        eta_weighted = float(np.sum(self.weights * self.etas))
        duration_weighted = float(np.sum(self.weights * self.durations))
        self.history.append({
            "eta": eta_weighted,
            "duration": duration_weighted,
            "ess": float(ess),
            "regime_probs": regime_probs.copy(),
        })

        return regime_probs

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self) -> None:
        """
        Evolve each particle one step under the HSMM dynamics.

        Particles either remain in their regime (incrementing duration and
        drifting eta by a small bounded random walk) or transition to a new
        regime, in which case duration resets to 1 and eta resets to a
        small uniform draw on ``[0, 0.3]``.
        """
        new_regimes = self.regimes.copy()
        new_etas = self.etas.copy()
        new_durations = self.durations.copy()

        for i in range(self.N):
            r = int(self.regimes[i])
            d = int(self.durations[i])
            probs = get_transition_probs(r, d)
            new_r = int(np.random.choice(self.K, p=probs))
            if new_r == r:
                new_durations[i] = d + 1
                new_etas[i] = float(np.clip(
                    self.etas[i] + np.random.normal(0, 0.02), 0.0, 1.0
                ))
            else:
                new_regimes[i] = new_r
                new_durations[i] = 1
                new_etas[i] = float(np.random.uniform(0, 0.3))

        self.regimes = new_regimes
        self.etas = new_etas
        self.durations = new_durations

    # ------------------------------------------------------------------
    # Observation update
    # ------------------------------------------------------------------

    def update(self, distances: dict[int, float],
               rf_probs: np.ndarray | None = None) -> None:
        """
        Importance-weight particles by their observation likelihood.

        Three modes (selected via ``likelihood_mode`` at init):

        * ``'kNN'`` (default): generative likelihood
          ``L_kNN(r) = exp(-d_r^2 / (2 * obs_sigma^2))``,
          where ``d_r`` is the median k-NN distance to regime r's cores.
          This is the original Mai-2026 pipeline.

        * ``'mix'``: convex combination of kNN and RF posteriors,
          following the informed-likelihood approach of Grisetti et al.
          in robotic mapping (the RF acts as a "smart sensor"):
          ``L(r) = lambda * L_kNN_norm(r) + (1-lambda) * L_RF(r)``.
          The kNN component is normalised across regimes so the two
          terms have comparable scale; RF posteriors are softened by
          ``rf_temperature`` to prevent single-class collapse.

        * ``'RF'``: discriminative-only likelihood (kNN ignored).

        Per-particle likelihood is then ``L(regime_of_particle)``.
        Missing kNN distances are imputed by the maximum-plus-slack rule.
        """
        max_d = max(distances.values()) if distances else 1.0
        sigma2 = 2.0 * self.obs_sigma ** 2

        # Always compute kNN per-regime likelihood (needed for kNN and mix modes).
        l_knn = np.zeros(self.K)
        for r in range(self.K):
            d_r = distances.get(r, max_d * 1.5)
            l_knn[r] = np.exp(-(d_r ** 2) / sigma2)

        if self.likelihood_mode == "kNN":
            # Original pipeline: per-particle exp(-d^2 / 2 sigma^2), no normalisation.
            likelihoods = np.zeros(self.N)
            for i in range(self.N):
                likelihoods[i] = l_knn[int(self.regimes[i])]
        else:
            # mix or RF: need RF posteriors and normalised likelihoods.
            if rf_probs is None:
                raise ValueError(
                    f"likelihood_mode={self.likelihood_mode!r} requires rf_probs."
                )
            rf_probs = np.asarray(rf_probs, dtype=float)
            if rf_probs.shape != (self.K,):
                raise ValueError(
                    f"rf_probs must have shape ({self.K},); got {rf_probs.shape}."
                )

            # Normalise kNN to a proper categorical likelihood.
            l_knn_sum = l_knn.sum()
            if l_knn_sum > 0:
                l_knn_norm = l_knn / l_knn_sum
            else:
                l_knn_norm = np.ones(self.K) / self.K

            # Soften RF posteriors via temperature, then floor + renormalise.
            rf_soft = np.maximum(rf_probs, 1e-12) ** (1.0 / self.rf_temperature)
            rf_soft = rf_soft / rf_soft.sum()
            rf_soft = np.maximum(rf_soft, self.rf_floor)
            rf_soft = rf_soft / rf_soft.sum()

            if self.likelihood_mode == "mix":
                lam = self.rf_lambda
                l_per_regime = lam * l_knn_norm + (1.0 - lam) * rf_soft
            else:  # 'RF'
                l_per_regime = rf_soft

            likelihoods = l_per_regime[self.regimes.astype(int)]

        new_w = self.weights * likelihoods
        total = new_w.sum()
        if total > 0:
            self.weights = new_w / total
        else:
            self.weights = np.ones(self.N) / self.N

    # ------------------------------------------------------------------
    # Eta update
    # ------------------------------------------------------------------

    def update_eta(self, distances: dict[int, float]) -> None:
        """
        Blend each particle's drifted eta with an observation-driven term.

        ``eta_new = 0.7 * eta_drift + 0.3 * clip(d_r / 12, 0, 1)``

        The mixing weight favours temporal smoothness.  The denominator
        ``12`` is the empirical typical scale of within-regime distances;
        it is a single global constant, not tuned per regime.
        """
        for i in range(self.N):
            r = int(self.regimes[i])
            if r in distances:
                eta_obs = float(np.clip(distances[r] / 12.0, 0.0, 1.0))
                self.etas[i] = float(np.clip(
                    0.7 * self.etas[i] + 0.3 * eta_obs, 0.0, 1.0
                ))

    # ------------------------------------------------------------------
    # Informed injection
    # ------------------------------------------------------------------

    def informed_injection(self, distances: dict[int, float]) -> None:
        """
        Replace a fraction of the worst particles by samples from the
        regimes most consistent with the current observation.

        This is the kidnapped-robot rescue: when the filter has lost track
        (low ESS), we inject fresh particles concentrated on regimes that
        explain the data well.  Replaced particles are chosen by lowest
        importance weight.
        """
        if not distances:
            return

        # Build a probability vector over regimes inversely proportional to
        # distance: closer regimes are more likely to receive new particles.
        regime_ids = list(distances.keys())
        d_vals = np.array([distances[r] for r in regime_ids])
        scores = np.exp(-d_vals / max(d_vals.mean(), 1e-6))
        scores = scores / scores.sum()

        n_inject = int(self.injection_fraction * self.N)
        if n_inject <= 0:
            return

        # Replace the n_inject lowest-weight particles
        worst = np.argsort(self.weights)[:n_inject]
        injected_regimes = np.random.choice(
            regime_ids, size=n_inject, p=scores
        )
        self.regimes[worst] = injected_regimes
        self.etas[worst] = np.random.uniform(0.3, 0.7, size=n_inject)
        self.durations[worst] = 1.0
        # Give them a fresh, equal weight; renormalise overall
        self.weights[worst] = 1.0 / self.N
        self.weights = self.weights / self.weights.sum()

    # ------------------------------------------------------------------
    # ESS and resampling
    # ------------------------------------------------------------------

    def effective_sample_size(self) -> float:
        """Standard ESS estimator: ``1 / sum(w_i^2)``."""
        return float(1.0 / np.sum(self.weights ** 2))

    def systematic_resample(self) -> None:
        """
        Low-variance systematic resampling.

        Produces a new particle set with equal weights, drawn proportionally
        to the current weights.  Standard reference: Kitagawa 1996.
        """
        positions = (np.random.random() + np.arange(self.N)) / self.N
        cumulative = np.cumsum(self.weights)
        idx = np.searchsorted(cumulative, positions)
        idx = np.clip(idx, 0, self.N - 1)

        self.regimes = self.regimes[idx]
        self.etas = self.etas[idx]
        self.durations = self.durations[idx]
        self.weights = np.ones(self.N) / self.N

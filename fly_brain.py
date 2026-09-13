import os
import numpy as np

class FlyBrain:
    def __init__(self, n_visual=32, n_hidden=64, n_motor=2, seed=0):
        rng = np.random.default_rng(seed)
        self.n_visual = n_visual
        self.n_hidden = n_hidden
        self.n_motor = n_motor
        n = n_visual + n_hidden + n_motor
        self.n = n
        self.v = np.zeros(n, dtype=np.float32)
        self.thr = np.ones(n, dtype=np.float32)
        self.decay = 0.85
        self.W = np.zeros((n, n), dtype=np.float32)
        self.W[:n_visual, n_visual:n_visual+n_hidden] = rng.normal(0.35, 0.12, (n_visual, n_hidden)).astype(np.float32)
        self.W[n_visual:n_visual+n_hidden, n_visual+n_hidden:] = rng.normal(0.5, 0.15, (n_hidden, n_motor)).astype(np.float32)
        rec = rng.normal(0.08, 0.05, (n_hidden, n_hidden)).astype(np.float32)
        np.fill_diagonal(rec, 0)
        self.W[n_visual:n_visual+n_hidden, n_visual:n_visual+n_hidden] = rec
        self.W = np.clip(self.W, 0, 1.2)
        self.plasticity = 0.005
        self.tried_real = False

    def load_real_subset(self):
        token = os.getenv("NEUPRINT_TOKEN", "")
        if not token:
            return False, "no token, using synthetic wiring"
        try:
            import neuprint as neu
            host = os.getenv("NEUPRINT_HOST", "https://neuprint.janelia.org")
            dataset = os.getenv("NEUPRINT_DATASET", "male-cns:v1.0")
            client = neu.Client(host, dataset=dataset, token=token)
            neu.set_default_client(client)
            crit = neu.NeuronCriteria(type=".*PPL101.*", client=client)
            df, _ = neu.fetch_neurons(crit)
            self.tried_real = True
            if df is None or len(df) == 0:
                crit2 = neu.NeuronCriteria(rois="GNG", client=client)
                df, _ = neu.fetch_neurons(crit2)
            if df is None or len(df) == 0:
                return False, "API reachable but no neurons returned, using synthetic"
            scale = min(len(df), self.n_hidden) / self.n_hidden
            self.W *= (0.5 + 0.5 * scale)
            return True, f"fetched {len(df)} real neurons, rescaled synthetic weights"
        except Exception as e:
            return False, f"API fetch failed ({e}), using synthetic"

    def step(self, visual, dopamine=0.0):
        n0 = self.n_visual
        n1 = n0 + self.n_hidden
        self.v *= self.decay
        vis = np.asarray(visual, dtype=np.float32).reshape(-1)
        if len(vis) != n0:
            nv = np.zeros(n0, dtype=np.float32)
            m = min(len(vis), n0)
            nv[:m] = vis[:m]
            vis = nv
        self.v[:n0] += vis * 1.2
        if dopamine != 0:
            self.v[n0:n1] += dopamine * 0.6
        inp = (self.v > 0).astype(np.float32)
        drive = inp @ self.W
        self.v += (drive * 0.5).astype(np.float32)
        spikes = (self.v >= self.thr).astype(np.float32)
        self.v[spikes > 0] = 0.0
        if dopamine < -0.1:
            pre = inp[:, None]
            post = spikes[None, :]
            dw = -self.plasticity * pre[n0:n1, :] if False else 0
        if dopamine != 0:
            j = np.random.default_rng().normal(0, abs(dopamine) * 0.002, self.W.shape).astype(np.float32)
            self.W = np.clip(self.W + j, 0, 1.2)
        motor = spikes[n1:]
        return motor, spikes

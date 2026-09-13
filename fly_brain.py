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
        # --- minimal adornment: trainable READOUT only, W stays frozen ---
        # R maps hidden spikes (n_hidden) -> motor logits (n_motor).
        rng2 = np.random.default_rng(seed + 7)
        self.R = rng2.normal(0.0, 0.1, (self.n_hidden, self.n_motor)).astype(np.float32)
        self.lr = 0.02          # readout learning rate
        self._last_hidden_spikes = np.zeros(self.n_hidden, dtype=np.float32)

    def load_real_subset(self, max_neurons=300, cache="data/connectome_subset.npz"):
        """Load the REAL fly connectome into W (recurrent hidden core only).

        Honest design: the hidden recurrent block (n_hidden x n_hidden) is wired
        from the actual male-cns:v1.0 adjacency (PPL101 + GNG neurons). The
        visual->hidden and hidden->motor blocks stay synthetic, because no
        biological mapping exists from NES pixels to fly neurons, nor from fly
        neurons to Mario buttons. We do NOT claim the full brain is real.

        Tries a pre-fetched cache (offline-safe) first, then the live neuPrint API.
        """
        token = os.getenv("NEUPRINT_TOKEN", "")
        n0 = self.n_visual
        n1 = n0 + self.n_hidden
        loaded = self._load_adj_from_cache(cache, n0, n1)
        if loaded:
            self.tried_real = True
            return True, f"loaded REAL adjacency from cache {cache} into hidden core"
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
            ids = df["bodyId"].tolist()[:max_neurons]
            adj = neu.fetch_adjacencies(ids, ids)
            edges = adj[1] if isinstance(adj, tuple) else adj
            self._write_adj_to_W(edges, ids, n0, n1, max_neurons)
            return True, f"fetched {len(ids)} REAL neurons, wired hidden core from adjacency"
        except Exception as e:
            return False, f"API fetch failed ({e}), using synthetic"

    def _load_adj_from_cache(self, cache, n0, n1):
        if not os.path.exists(cache):
            return False
        try:
            d = np.load(cache, allow_pickle=True)
            edges = d.get("edges")
            ids = list(d.get("body_ids")) if "body_ids" in d else None
            if edges is None or ids is None:
                return False
            self._write_adj_to_W(edges, ids, n0, n1, len(ids))
            return True
        except Exception:
            return False

    def _write_adj_to_W(self, edges, ids, n0, n1, max_neurons):
        """Coarse-grain real adjacency (max_neurons x max_neurons) into the
        n_hidden x n_hidden recurrent block, preserving sign and relative weight."""
        import pandas as pd
        if hasattr(edges, "to_numpy"):
            edges = edges.to_numpy()
        edf = edges if hasattr(edges, "shape") else np.asarray(edges)
        # map each real neuron id -> a slot in [0, n_hidden)
        # ids may be ints or strings; edges carry the same id values
        id_to_slot = {bid: i % self.n_hidden for i, bid in enumerate(ids)}
        def keyof(x):
            # edges store raw id values; match against ids list (int or str)
            try:
                return id_to_slot[x]
            except (KeyError, TypeError):
                # fall back: if ids are strings like 'N12', try int(x)
                try:
                    return id_to_slot[str(x)]
                except (KeyError, TypeError):
                    return id_to_slot.get(int(x) % self.n_hidden if isinstance(x, (int, float)) else None)
        H = np.zeros((self.n_hidden, self.n_hidden), dtype=np.float32)
        # edges rows: typically [bodyId_pre, bodyId_post, weight]
        pre = edf[:, 0] if edf.shape[1] >= 2 else edf[:, 0]
        post = edf[:, 1] if edf.shape[1] >= 2 else edf[:, 1]
        w = edf[:, 2] if edf.shape[1] >= 3 else np.ones(len(edf))
        for a, b, wt in zip(pre, post, w):
            sa = keyof(a)
            sb = keyof(b)
            if sa is None or sb is None:
                continue
            H[sa, sb] += float(wt)
        # scale into the model's [0, 1.2] clip range, keep sign
        amax = np.abs(H).max()
        if amax > 0:
            H = (H / amax) * 0.8
        # self-loops off
        np.fill_diagonal(H, 0.0)
        self.W[n0:n1, n0:n1] = np.clip(H, -1.2, 1.2).astype(np.float32)

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
        # hidden spikes (substrate output) -> trainable readout -> motor logits
        hidden_sp = spikes[n0:n1]
        self._last_hidden_spikes = hidden_sp
        logits = hidden_sp @ self.R          # (n_motor,)
        motor = (logits > 0).astype(np.float32)
        return motor, spikes, logits

    def reinforce(self, reward, baseline=0.0):
        """Minimal reward-driven update of the readout R (W frozen).
        REINFORCE: nudge R in the direction of hidden spikes that preceded a
        positive reward. Call once per step with the step's reward."""
        adv = float(reward) - float(baseline)
        if adv == 0:
            return
        h = self._last_hidden_spikes
        dR = np.outer(h, np.ones(self.n_motor, dtype=np.float32)) * (self.lr * adv)
        self.R += dR.astype(np.float32)
        self.R = np.clip(self.R, -3.0, 3.0)

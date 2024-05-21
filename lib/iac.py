import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Unit():
    def __init__(self, name, estr=.4, rest=-0.1, decay=0.1, amax=1.0, amin=-0.2, alpha=0.1, gamma=0.1, noise=0.0, lrate=0.0) -> None:
        self.name = name

        self.inp = dict()
        self.clamped = False

        self.estr = estr
        self.alpha = alpha
        self.gamma = gamma
        self.noise = noise
        self.decay = decay
        self.amax = amax
        self.amin = amin
        self.rest = rest
        self.lrate = lrate

        self.net = 0.0
        self.ext = 0.0
        self.a = rest
        self.dw = 0.0

    def __str__(self) -> str:
        return f"<Unit {self.name}>"

    def __repr__(self) -> str:
        return f"<Unit {self.name}>"
    
    @property
    def Sw(self):
        pos_weights = list(filter(lambda w: w > 0, self.inp.values()))
        return sum(pos_weights)
    
    @property
    def Sa(self):
        exciting_units = list(filter(lambda u: self.inp[u] > 0, self.inp))
        return sum([u.a for u in exciting_units])

    def clamp(self, val) -> None:
        assert val >= self.amin and val <= self.amax, "Cannot clamp outside amin and amax"
        self.ext = val
        self.clamped = True

    def unclamp(self):
        self.ext = 0.0
        self.clamped = False

    def connect(self, u, w=0.0) -> None:
        self.inp[u] = w
        u.inp[self] = w

    def addinp(self, u, w=0.0) -> None:
        self.inp[u] = w

    def addinps(self, us, ws) -> None:
        assert len(us) == len(ws), "input arguments must be the same length"
        for u, w in zip(us, ws):
            self.inp[u] = w

    def getnet(self) -> None:
        acts = np.array([u.a for u in list(self.inp.keys())])
        weights = np.array(list(self.inp.values()))
        pos = weights > 0
        external = self.estr * self.ext
        noise = np.random.normal(0, self.noise) if self.noise != 0.0 else 0.0
        excitation = self.alpha * np.sum(acts[pos] * weights[pos])
        inhibition = self.gamma * np.sum(acts[~pos] * weights[~pos])
        self.net = external + excitation + inhibition + noise

    def learn(self, norm=False):
        Sw, Sa = 1.0, 1.0
        total = 0.0
        for sender in filter(lambda u: self.inp[u] > 0, self.inp):
            Δw = self.lrate * self.a * (sender.a / Sa - self.inp[sender] / Sw)
            self.inp[sender] += Δw
            total += Δw
        if total != 0.0: print(total)
        self.dw = total

    def update(self)-> None:
        if self.net > 0:
            Δa = (self.amax - self.a) * self.net - self.decay * (self.a - self.rest)
        else:
            Δa = (self.a - self.amin) * self.net - self.decay * (self.a - self.rest)
        self.a = min(self.a + Δa, self.amax)
        self.a = max(self.a + Δa, self.amin)

    def step(self) -> None:
        self.getnet()
        self.update()
        return self.a
    
    def reset(self) -> None:
        self.net = 0.0
        self.a = 0.0
        self.ext = 0.0
        self.dw = 0.0
        self.clamped = False


class Pool():
    def __init__(self, name, units, inh=1.0) -> None:
        self.name = name
        self.inh = inh
        self.units = units
        self.connect_units()

    def __str__(self) -> str:
        return f"<Pool {self.name}>"
    
    def __repr__(self) -> str:
        return f"<Pool {self.name}>"

    def __getitem__(self, name) -> Unit:
        if name == slice(None):
            return self.units
        return next(filter(lambda unit: unit.name == name, self.units))

    def connect_units(self) -> None:
        n = len(self.units)
        for i in range(n):
            for j in range(i+1, n):
                self.units[i].addinp(self.units[j], -self.inh)
                self.units[j].addinp(self.units[i], -self.inh)

    def run(self, method) -> None:
        for u in self.units:
            getattr(u, method)()

    def step(self) -> None:
        for u in self.units:
            u.getnet()
        for u in self.units:
            u.update()

    def push(self, u) -> None:
        self.units.append(u)
        if len(self.units) > 1:
            self.connect_units()


class Network():
    def __init__(self, pools, name="iac", plastic=False) -> None:
        self.name = name
        self.pools = pools
        self.plastic = plastic

    def __getitem__(self, index) -> Pool or Unit:
        if isinstance(index, str):
            return next(filter(lambda pool: pool.name == index, self.pools))
        elif isinstance(index, tuple):
            assert len(index) == 2, "Index must be at post a 2-tuple"
            pool_idx, unit_idx = index

            # [name, :]
            if isinstance(pool_idx, str) and unit_idx == slice(None):
                return self[pool_idx][:]
            # [name, name]
            elif isinstance(pool_idx, str) and isinstance(unit_idx, str):
                p = next(filter(lambda pool: pool.name == index[0], self.pools))
                return p[index[1]]
            # [:, :]
            elif pool_idx == slice(None) and unit_idx == slice(None):
                return list(itertools.chain.from_iterable([p[:] for p in self.pools]))

    def __contains__(self, item) -> bool:
        if isinstance(item, Pool):
            return item in [p.name for p in self.pools]
        elif isinstance(item, Unit):
            return item in [u.name for u in self[:, :]]

    def reset(self):
        for u in self[:, :]:
            u.reset()

    def getnet(self) -> None:
        for p in self.pools:
            for u in p.units:
                u.getnet()
    
    def update(self) -> None:
        for p in self.pools:
            for u in p.units:
                u.update()

    def step(self) -> None:
        for p in self.pools:
            for u in p.units:
                u.getnet()
            for u in p.units:
                u.update()

    def connect(self, names1, names2, w) -> None:
        p1, u1 = names1
        p2, u2 = names2
        self[p1, u1].addinp(self[p2, u2], w)
        self[p2, u2].addinp(self[p1, u1], w)

    def set_units(self, attr, value):
        for u in self[:, :]:
            setattr(u, attr, value)


class Learner(Network):
    def fixate(self, x) -> None:
        for pool_name, unit_name in x:
            unit = self[pool_name, unit_name]
            unit.clamp(self[pool_name, unit_name].amax)
            unit.getnet()
            unit.getnet()

    def retreive(self, x):
        name = '_'.join([f"{pool_name}_{unit_name}" for pool_name, unit_name in x])
        return name if name in self else None

    def unclamp_all(self) -> None:
        for u in self[:, :]:
            u.unclamp()

    def learn1(self, x) -> None:
        name = '_'.join([f"{pool_name}_{unit_name}" for pool_name, unit_name in x])
        new_unit = Unit(name)
        self["Hidden"].push(new_unit)
        for pool_name, unit_name in x:
            new_unit.connect(self[pool_name, unit_name], w=.5 if pool_name=="Food" else 1.0)

    def learn2(self, norm=False) -> None:
        for p in self.pools:
            for u in p.units:
                u.learn(norm)
        # self.unclamp_all()


class API():
    def __init__(self, network) -> None:
        self.networks = tuple([network]) if isinstance(network, Network) else network
        self.counter = 0

        # Init data
        keys = ["network", "pool", "unit", "cycle", "net", "act", "dw"]
        self.data = {key: [] for key in keys}
        self.record_data()

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def record_data(self) -> None:
        for i, n in enumerate(self.networks):
            for p in n.pools:
                for u in p.units:
                    self.data["network"].append(n.name)
                    self.data["pool"].append(p.name)
                    self.data["unit"].append(u.name)
                    self.data["cycle"].append(self.counter)
                    self.data["net"].append(u.net)
                    self.data["act"].append(u.a)
                    self.data["dw"].append(u.dw)

    def run(self, n) -> None:
        for _ in range(n):
            for network in self.networks:
                network.getnet()
            for network in self.networks:
                network.update()
            self.counter += 1
            self.record_data()
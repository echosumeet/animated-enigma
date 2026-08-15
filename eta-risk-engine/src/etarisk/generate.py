"""Synthetic shipment generator.

The point of this module is to produce a transit-time distribution that is
*hard in the way real transit times are hard*: right-skewed, heavy-tailed, with
the tail driven by a handful of latent processes (hub congestion, weather,
customs) that are only partially observable at booking time.

Structure of the data-generating process
----------------------------------------

For a shipment booked on day ``t`` on lane ``(o, d)`` with mode ``m`` and
carrier ``c``, the realised transit time in hours is

    T = B(o, d, m) * L(o, d) * C(c, m) * S(t) * W(t)          [ multiplicative core ]
        + dwell(dow(t))                                        [ calendar dwell     ]
        + congest(o, t) + congest(d, t + B/24)                 [ hub queueing       ]
        + weather(region(o), t) + weather(region(d), t)        [ met disruption     ]
        + customs(o, d, t)                                     [ heavy-tailed hold  ]
        + disruption(t)                                        [ rare Pareto tail   ]

with ``W(t)`` a lognormal core noise term. Every additive term is
non-negative, which is the structural reason the distribution is right-skewed:
a shipment can be arbitrarily late but cannot arrive before its physical
minimum. The customs and disruption terms are mixtures (zero with high
probability, heavy-tailed when they fire), which is what produces the tail that
squared-error models chase and quantile models handle.

Observability
-------------

The generator returns *both* the observable booking-time features and the
latent realised drivers. Only the observable ones are used for modelling; the
latent columns are kept so that tests and the design notes can quantify how
much of the variance is structurally unlearnable. In particular:

* hub congestion is observable **as of the booking date** (you can see today's
  yard queue) but not for the future days the shipment is actually in transit;
* weather is observable only as a *forecast* — the true severity plus noise,
  with the noise growing in the forecast horizon;
* customs holds and the rare disruption term are not observable at all.

That gap is deliberate. A model that could predict transit time exactly would
make the conformal-coverage part of this repo trivial and dishonest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .rng import StreamBank

__all__ = [
    "Hub",
    "Mode",
    "Carrier",
    "Lane",
    "NetworkSpec",
    "GeneratorConfig",
    "default_network",
    "generate_shipments",
    "OBSERVABLE_COLUMNS",
    "LATENT_COLUMNS",
]


# --------------------------------------------------------------------------------------
# Network primitives
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hub:
    """An origin or destination node: a port, airport, rail yard or DC."""

    code: str
    region: str
    country: str
    lat: float
    lon: float
    #: Baseline queueing pressure, 0 = free flowing, 1 = chronically congested.
    congestion_base: float
    #: How strongly the annual peak-season cycle hits this hub.
    seasonal_amplitude: float


@dataclass(frozen=True)
class Mode:
    """A transport mode with its physical speed and handling overhead."""

    name: str
    speed_kmph: float
    #: Fixed pickup + drop overhead in hours, independent of distance.
    handling_h: float
    #: Multiplicative lognormal sigma of the core noise term.
    noise_sigma: float
    #: Probability that the rare heavy-tailed disruption term fires.
    disruption_p: float
    #: Lomax scale (hours) of the disruption term when it fires.
    disruption_scale: float
    #: How much of a hub's congestion index converts into waiting hours.
    congestion_sensitivity: float


@dataclass(frozen=True)
class Carrier:
    """A carrier with mode coverage and a reliability profile."""

    name: str
    modes: tuple[str, ...]
    #: Multiplier on the core transit time. <1 is genuinely faster.
    speed_factor: float
    #: Multiplier on the disruption probability. >1 is a sloppier operator.
    reliability_factor: float
    #: Share of total volume this carrier wins.
    share: float


@dataclass(frozen=True)
class Lane:
    """An origin-destination-mode triple with a quoted transit time."""

    origin: str
    dest: str
    mode: str
    distance_km: float
    #: Persistent lane-specific multiplier (route quality, border friction).
    lane_factor: float
    #: The transit time sold to the customer, in hours.
    planned_transit_h: float
    cross_border: bool
    share: float


@dataclass
class NetworkSpec:
    hubs: dict[str, Hub]
    modes: dict[str, Mode]
    carriers: list[Carrier]
    lanes: list[Lane]

    def lane_index(self) -> dict[tuple[str, str, str], Lane]:
        return {(ln.origin, ln.dest, ln.mode): ln for ln in self.lanes}


@dataclass
class GeneratorConfig:
    """Everything that controls a single generated dataset."""

    n_shipments: int = 60_000
    start_date: str = "2023-01-02"
    days: int = 730
    seed: int = 20260101
    #: Lateness tolerance in hours before a shipment counts as late.
    late_tolerance_h: float = 0.0
    #: Multiplier applied to carrier disruption probability after ``regime_shift_day``.
    regime_carrier_degradation: float = 1.8
    #: Additive shock to every hub congestion index after ``regime_shift_day``.
    regime_congestion_shock: float = 0.30
    #: Day index at which the regime shift starts. ``None`` disables it.
    regime_shift_day: int | None = 430
    #: Names of carriers hit by the degradation. Empty means all of them.
    regime_carriers: tuple[str, ...] = ("BOREAL", "FJORD")


# --------------------------------------------------------------------------------------
# Default network
# --------------------------------------------------------------------------------------


def default_network() -> NetworkSpec:
    """A small but non-degenerate global network: 10 hubs, 4 modes, 7 carriers.

    Distances are great-circle distances computed from the coordinates below, so
    the geography is internally consistent rather than hand-typed.
    """
    hubs = [
        Hub("SHA", "EAST_ASIA", "CN", 31.23, 121.47, 0.62, 0.55),
        Hub("SIN", "SEA", "SG", 1.35, 103.82, 0.44, 0.30),
        Hub("ROT", "EU", "NL", 51.92, 4.48, 0.51, 0.40),
        Hub("HAM", "EU", "DE", 53.55, 9.99, 0.46, 0.35),
        Hub("LAX", "NA_WEST", "US", 33.94, -118.41, 0.68, 0.60),
        Hub("CHI", "NA_CENTRAL", "US", 41.88, -87.63, 0.39, 0.25),
        Hub("NYC", "NA_EAST", "US", 40.71, -74.01, 0.55, 0.38),
        Hub("DXB", "MEA", "AE", 25.20, 55.27, 0.33, 0.20),
        Hub("MEX", "LATAM", "MX", 19.43, -99.13, 0.58, 0.30),
        Hub("BOM", "SOUTH_ASIA", "IN", 19.08, 72.88, 0.64, 0.45),
    ]
    modes = [
        # name        speed  handling  noise  disr_p  disr_scale  cong_sens
        Mode("air", 780.0, 14.0, 0.13, 0.022, 16.0, 5.0),
        Mode("road", 62.0, 6.0, 0.17, 0.038, 20.0, 7.0),
        Mode("rail", 42.0, 20.0, 0.21, 0.048, 46.0, 14.0),
        Mode("ocean", 33.0, 46.0, 0.15, 0.060, 108.0, 42.0),
    ]
    carriers = [
        Carrier("ATLAS", ("ocean", "rail"), 0.97, 0.80, 0.17),
        Carrier("BOREAL", ("ocean", "road"), 1.04, 1.35, 0.15),
        Carrier("CYGNUS", ("air", "road"), 0.95, 0.70, 0.14),
        Carrier("DELTAWING", ("air",), 1.01, 1.10, 0.11),
        Carrier("EVEREST", ("road", "rail"), 1.00, 0.95, 0.18),
        Carrier("FJORD", ("road",), 1.08, 1.55, 0.13),
        Carrier("GRANITE", ("rail", "ocean", "road"), 1.02, 1.20, 0.12),
    ]
    hub_map = {h.code: h for h in hubs}
    mode_map = {m.name: m for m in modes}

    # Lane list: intercontinental legs go ocean/air, continental legs road/rail.
    raw_lanes: list[tuple[str, str, tuple[str, ...]]] = [
        ("SHA", "LAX", ("ocean", "air")),
        ("SHA", "ROT", ("ocean", "air")),
        ("SIN", "ROT", ("ocean", "air")),
        ("SIN", "LAX", ("ocean",)),
        ("BOM", "HAM", ("ocean", "air")),
        ("BOM", "DXB", ("air", "ocean")),
        ("DXB", "ROT", ("air", "ocean")),
        ("ROT", "HAM", ("road", "rail")),
        ("LAX", "CHI", ("road", "rail")),
        ("CHI", "NYC", ("road", "rail")),
        ("LAX", "NYC", ("rail", "air")),
        ("MEX", "CHI", ("road", "rail")),
        ("MEX", "LAX", ("road",)),
        ("SHA", "SIN", ("ocean", "air")),
        ("HAM", "NYC", ("ocean", "air")),
    ]
    lanes: list[Lane] = []
    lane_rng = np.random.default_rng(97531)
    for origin, dest, lane_modes in raw_lanes:
        dist = _haversine_km(hub_map[origin], hub_map[dest])
        for mode_name in lane_modes:
            mode = mode_map[mode_name]
            lane_factor = float(np.exp(lane_rng.normal(0.0, 0.09)))
            cross = hub_map[origin].country != hub_map[dest].country
            planned = _quoted_transit(hub_map[origin], hub_map[dest], mode, dist, lane_factor, cross)
            share = 1.0 if mode_name in ("ocean", "road") else 0.55
            lanes.append(
                Lane(origin, dest, mode_name, float(dist), lane_factor, float(planned), cross, share)
            )
    total = sum(ln.share for ln in lanes)
    lanes = [
        Lane(
            ln.origin,
            ln.dest,
            ln.mode,
            ln.distance_km,
            ln.lane_factor,
            ln.planned_transit_h,
            ln.cross_border,
            ln.share / total,
        )
        for ln in lanes
    ]
    return NetworkSpec(hub_map, mode_map, carriers, lanes)


#: Commercial buffer applied on top of the planned allowances when quoting.
QUOTE_BUFFER = 1.06
#: Tail index of the Lomax disruption term. Above 2 so the variance exists.
DISRUPTION_ALPHA = 2.2
#: Disruption draws are truncated at this multiple of the mode's scale.
DISRUPTION_TRUNCATION = 12.0


def _quoted_transit(
    origin: Hub, dest: Hub, mode: Mode, distance_km: float, lane_factor: float, cross_border: bool
) -> float:
    """The transit time published to the customer, in whole 4-hour blocks.

    A quote is built the way planners actually build one: physical drive/sail
    time, plus *planned allowances* for the dwell and hub queueing you expect on
    an average day, plus a small commercial buffer. Rounded to 4-hour blocks
    because that is how tariffs are published.

    Critically, the quote contains **no allowance for weather, for the tail of
    the customs-hold distribution, or for rare disruption**. Nobody quotes a
    99th percentile; you would lose the business. That mismatch — a quote near
    the mean of the well-behaved part of the distribution, against a realised
    distribution with a Pareto tail — is the entire commercial problem this
    repository is about.
    """
    base = distance_km / mode.speed_kmph + mode.handling_h
    # Expected dwell under the weekday-weighted booking mix (weights 1,1,1,1,1,.35,.35).
    dwell_mix = (18.0 * 1.0 + 26.0 * 0.35 + 12.0 * 0.35) / 5.7
    dwell_allow = dwell_mix * (0.35 if mode.name in ("air", "road") else 1.0)
    mean_o = origin.congestion_base + 0.5 * origin.seasonal_amplitude
    mean_d = dest.congestion_base + 0.5 * dest.seasonal_amplitude
    congestion_allow = mode.congestion_sensitivity * (0.45 * mean_o + 0.85 * mean_d)
    # Planners allow for the *modal* customs experience, not the tail: the median
    # hold (exp(2.9) hours) times the hold probability.
    hold_p = (0.16 if cross_border else 0.0) * (1.5 if mode.name == "ocean" else 1.0)
    customs_allow = hold_p * float(np.exp(2.9))
    total = (base * lane_factor + dwell_allow + congestion_allow + customs_allow) * QUOTE_BUFFER
    return float(4.0 * np.ceil(total / 4.0))


def _haversine_km(a: Hub, b: Hub) -> float:
    r = 6371.0
    p1, p2 = np.radians(a.lat), np.radians(b.lat)
    dphi = p2 - p1
    dlam = np.radians(b.lon - a.lon)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(h)))


# --------------------------------------------------------------------------------------
# Latent daily processes
# --------------------------------------------------------------------------------------


def _ar1_path(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """A zero-mean AR(1) path. Congestion and weather are both persistent."""
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1e-9, 1 - phi**2))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _congestion_paths(spec: NetworkSpec, cfg: GeneratorConfig, streams: StreamBank) -> pd.DataFrame:
    """Daily congestion index per hub, in [0, ~1.6].

    Three components: a persistent AR(1) queue, an annual peak-season cycle
    (northern-hemisphere Q3/Q4 build), and the hub's structural baseline.
    """
    days = np.arange(cfg.days)
    doy = (pd.Timestamp(cfg.start_date) + pd.to_timedelta(days, unit="D")).dayofyear.to_numpy()
    frames = {}
    for code, hub in spec.hubs.items():
        rng = streams.get(f"congestion::{code}")
        ar = _ar1_path(rng, cfg.days, phi=0.93, sigma=0.055)
        # Peak at day-of-year ~300 (Q4 pre-holiday push).
        season = hub.seasonal_amplitude * 0.5 * (1 + np.cos(2 * np.pi * (doy - 300) / 365.25))
        idx = hub.congestion_base + season + ar
        if cfg.regime_shift_day is not None and cfg.regime_congestion_shock:
            idx = idx + cfg.regime_congestion_shock * (days >= cfg.regime_shift_day)
        frames[code] = np.clip(idx, 0.0, 2.0)
    return pd.DataFrame(frames, index=days)


def _weather_paths(spec: NetworkSpec, cfg: GeneratorConfig, streams: StreamBank) -> pd.DataFrame:
    """Daily weather severity per region, in [0, 1]. Winter-weighted, persistent."""
    days = np.arange(cfg.days)
    doy = (pd.Timestamp(cfg.start_date) + pd.to_timedelta(days, unit="D")).dayofyear.to_numpy()
    regions = sorted({h.region for h in spec.hubs.values()})
    frames = {}
    for region in regions:
        rng = streams.get(f"weather::{region}")
        ar = _ar1_path(rng, cfg.days, phi=0.86, sigma=0.30)
        northern = not region.startswith("LATAM")
        phase = 20 if northern else 200
        season = 0.5 * (1 + np.cos(2 * np.pi * (doy - phase) / 365.25))
        raw = -1.15 + 1.5 * season + ar
        frames[region] = 1.0 / (1.0 + np.exp(-raw))  # logistic squash to [0, 1]
    return pd.DataFrame(frames, index=days)


# --------------------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------------------

OBSERVABLE_COLUMNS = [
    "shipment_id",
    "ship_ts",
    "origin",
    "dest",
    "mode",
    "carrier",
    "lane",
    "distance_km",
    "planned_transit_h",
    "cross_border",
    "weight_kg",
    "pieces",
    "ship_dow",
    "ship_month",
    "ship_week",
    "day_index",
    "origin_congestion_obs",
    "dest_congestion_obs",
    "weather_forecast_origin",
    "weather_forecast_dest",
]

LATENT_COLUMNS = [
    "latent_congestion_h",
    "latent_weather_h",
    "latent_customs_h",
    "latent_disruption_h",
    "latent_dwell_h",
    "latent_core_h",
]


def generate_shipments(cfg: GeneratorConfig | None = None, spec: NetworkSpec | None = None) -> pd.DataFrame:
    """Generate a shipment-level dataset sorted by ship timestamp.

    Returns a frame with the observable booking-time columns, the latent driver
    decomposition, and the two targets: ``actual_transit_h`` and ``is_late``.
    """
    cfg = cfg or GeneratorConfig()
    spec = spec or default_network()
    streams = StreamBank(cfg.seed)
    n = int(cfg.n_shipments)

    congestion = _congestion_paths(spec, cfg, streams)
    weather = _weather_paths(spec, cfg, streams)

    # ---- who ships what, when ---------------------------------------------------
    lane_rng = streams.get("lane_choice")
    lane_p = np.array([ln.share for ln in spec.lanes])
    lane_ids = lane_rng.choice(len(spec.lanes), size=n, p=lane_p / lane_p.sum())

    day_rng = streams.get("ship_day")
    # Volume is itself seasonal and weekday-weighted; booking is not uniform.
    days = np.arange(cfg.days)
    dts = pd.Timestamp(cfg.start_date) + pd.to_timedelta(days, unit="D")
    dow = dts.dayofweek.to_numpy()
    doy = dts.dayofyear.to_numpy()
    day_weight = (1.0 + 0.45 * 0.5 * (1 + np.cos(2 * np.pi * (doy - 300) / 365.25))) * np.where(
        dow >= 5, 0.35, 1.0
    )
    day_idx = day_rng.choice(cfg.days, size=n, p=day_weight / day_weight.sum())
    day_idx.sort()  # stream arrives in time order

    lane_arr = np.array(spec.lanes, dtype=object)[lane_ids]
    origin = np.array([ln.origin for ln in lane_arr])
    dest = np.array([ln.dest for ln in lane_arr])
    mode_name = np.array([ln.mode for ln in lane_arr])
    distance = np.array([ln.distance_km for ln in lane_arr])
    lane_factor = np.array([ln.lane_factor for ln in lane_arr])
    planned = np.array([ln.planned_transit_h for ln in lane_arr])
    cross_border = np.array([ln.cross_border for ln in lane_arr])

    # ---- carrier assignment (mode-constrained) ----------------------------------
    car_rng = streams.get("carrier_choice")
    carrier_names = np.empty(n, dtype=object)
    for m in spec.modes:
        eligible = [c for c in spec.carriers if m in c.modes]
        w = np.array([c.share for c in eligible])
        w = w / w.sum()
        sel = mode_name == m
        k = int(sel.sum())
        if k:
            carrier_names[sel] = car_rng.choice([c.name for c in eligible], size=k, p=w)
    carrier_map = {c.name: c for c in spec.carriers}
    speed_factor = np.array([carrier_map[c].speed_factor for c in carrier_names])
    reliability = np.array([carrier_map[c].reliability_factor for c in carrier_names])

    if cfg.regime_shift_day is not None and cfg.regime_carrier_degradation != 1.0:
        hit = (
            np.ones(n, dtype=bool)
            if not cfg.regime_carriers
            else np.isin(carrier_names.astype(str), np.array(cfg.regime_carriers))
        )
        reliability = reliability * np.where(
            hit & (day_idx >= cfg.regime_shift_day), cfg.regime_carrier_degradation, 1.0
        )

    # ---- physical core ----------------------------------------------------------
    mode_speed = np.array([spec.modes[m].speed_kmph for m in mode_name])
    mode_handling = np.array([spec.modes[m].handling_h for m in mode_name])
    mode_noise = np.array([spec.modes[m].noise_sigma for m in mode_name])
    mode_disr_p = np.array([spec.modes[m].disruption_p for m in mode_name])
    mode_disr_s = np.array([spec.modes[m].disruption_scale for m in mode_name])
    mode_cong_s = np.array([spec.modes[m].congestion_sensitivity for m in mode_name])

    base_h = distance / mode_speed + mode_handling
    noise_rng = streams.get("core_noise")
    # Lognormal with a median-preserving correction so the core does not drift.
    core_noise = np.exp(noise_rng.normal(-0.5 * mode_noise**2, mode_noise))
    core_h = base_h * lane_factor * speed_factor * core_noise

    # ---- calendar dwell ---------------------------------------------------------
    ship_dow = dow[day_idx]
    # Friday/Saturday departures sit over the weekend at the receiving hub.
    dwell_rng = streams.get("dwell")
    dwell_h = np.select(
        [ship_dow == 4, ship_dow == 5, ship_dow == 6],
        [18.0, 26.0, 12.0],
        default=0.0,
    ) * (1.0 + 0.25 * dwell_rng.standard_normal(n))
    dwell_h = np.clip(dwell_h, 0.0, None)
    # Air and road clear weekends far better than rail and ocean.
    dwell_h *= np.where(np.isin(mode_name, ["air", "road"]), 0.35, 1.0)

    # ---- hub congestion ---------------------------------------------------------
    origin_cong = congestion.to_numpy()[day_idx, [congestion.columns.get_loc(o) for o in origin]]
    arrive_day = np.clip(day_idx + np.ceil(core_h / 24).astype(int), 0, cfg.days - 1)
    dest_cong = congestion.to_numpy()[arrive_day, [congestion.columns.get_loc(d) for d in dest]]
    cong_rng = streams.get("congestion_noise")
    congestion_h = mode_cong_s * (
        0.45 * origin_cong + 0.85 * dest_cong
    ) * np.exp(cong_rng.normal(-0.5 * 0.35**2, 0.35, n))

    # Observable congestion is the *booking-day* index at both hubs, which is what
    # a live yard/port feed would give you. The destination index that actually
    # matters is the one on the arrival day, days or weeks later.
    dest_cong_obs = congestion.to_numpy()[day_idx, [congestion.columns.get_loc(d) for d in dest]]

    # ---- weather ----------------------------------------------------------------
    region_of = {code: h.region for code, h in spec.hubs.items()}
    w_cols = {c: i for i, c in enumerate(weather.columns)}
    w_arr = weather.to_numpy()
    o_reg_idx = np.array([w_cols[region_of[o]] for o in origin])
    d_reg_idx = np.array([w_cols[region_of[d]] for d in dest])
    w_origin = w_arr[day_idx, o_reg_idx]
    w_dest_true = w_arr[arrive_day, d_reg_idx]
    w_dest_book = w_arr[day_idx, d_reg_idx]
    wx_rng = streams.get("weather_impact")
    # Weather only bites above a threshold; below it, crews absorb it.
    weather_h = (
        26.0
        * np.clip(w_origin - 0.55, 0.0, None)
        * np.where(mode_name == "air", 2.1, 1.0)
        + 30.0 * np.clip(w_dest_true - 0.55, 0.0, None) * np.where(mode_name == "air", 2.1, 1.0)
    ) * np.abs(wx_rng.normal(1.0, 0.4, n))

    # Forecast skill decays with horizon: origin weather is nowcast, destination
    # weather has to be forecast over the whole transit.
    fc_rng = streams.get("weather_forecast")
    horizon_days = np.clip(core_h / 24.0, 0.0, 30.0)
    fc_sigma_o = 0.04
    fc_sigma_d = 0.05 + 0.030 * np.sqrt(horizon_days)
    wf_origin = np.clip(w_origin + fc_rng.normal(0, fc_sigma_o, n), 0.0, 1.0)
    wf_dest = np.clip(w_dest_book + fc_rng.normal(0, fc_sigma_d, n), 0.0, 1.0)

    # ---- customs ----------------------------------------------------------------
    cust_rng = streams.get("customs")
    hold_p = np.where(cross_border, 0.16, 0.0) * np.where(mode_name == "ocean", 1.5, 1.0)
    held = cust_rng.random(n) < hold_p
    # Hold duration is lognormal: most clear in a day, a few sit for a fortnight.
    customs_h = np.where(held, np.minimum(np.exp(cust_rng.normal(2.9, 1.05, n)), 24 * 14.0), 0.0)

    # ---- rare heavy-tail disruption ---------------------------------------------
    dis_rng = streams.get("disruption")
    fires = dis_rng.random(n) < np.clip(mode_disr_p * reliability, 0.0, 0.95)
    # Lomax (Pareto type II) with alpha = 2.2: a genuinely heavy right tail --
    # finite variance, but only just, and no moments above the second. This is
    # the term that destabilises squared-error training and makes quantile loss
    # the right objective. Truncated at 12x scale because a disrupted shipment
    # eventually gets re-routed, salvaged or written off; unbounded Pareto draws
    # produce transit times measured in decades, which is not a hard problem,
    # just a wrong one.
    raw = mode_disr_s * dis_rng.pareto(DISRUPTION_ALPHA, n)
    disruption_h = np.where(fires, np.minimum(raw, DISRUPTION_TRUNCATION * mode_disr_s), 0.0)

    total_h = core_h + dwell_h + congestion_h + weather_h + customs_h + disruption_h

    ship_ts = pd.Timestamp(cfg.start_date) + pd.to_timedelta(day_idx, unit="D")
    size_rng = streams.get("shipment_size")
    pieces = 1 + size_rng.poisson(np.where(mode_name == "air", 3.0, 18.0))
    weight = np.round(
        pieces * np.exp(size_rng.normal(np.where(mode_name == "air", 3.4, 5.6), 0.6, n)), 1
    )

    df = pd.DataFrame(
        {
            "shipment_id": np.arange(n),
            "ship_ts": ship_ts,
            "origin": origin,
            "dest": dest,
            "mode": mode_name,
            "carrier": carrier_names.astype(str),
            "lane": np.array([f"{o}-{d}" for o, d in zip(origin, dest)]),
            "distance_km": np.round(distance, 1),
            "planned_transit_h": planned,
            "cross_border": cross_border,
            "weight_kg": weight,
            "pieces": pieces,
            "ship_dow": ship_dow,
            "ship_month": dts.month.to_numpy()[day_idx],
            "ship_week": dts.isocalendar().week.to_numpy()[day_idx].astype(int),
            "day_index": day_idx,
            "origin_congestion_obs": np.round(origin_cong, 4),
            "dest_congestion_obs": np.round(dest_cong_obs, 4),
            "weather_forecast_origin": np.round(wf_origin, 4),
            "weather_forecast_dest": np.round(wf_dest, 4),
            "latent_core_h": np.round(core_h, 3),
            "latent_dwell_h": np.round(dwell_h, 3),
            "latent_congestion_h": np.round(congestion_h, 3),
            "latent_weather_h": np.round(weather_h, 3),
            "latent_customs_h": np.round(customs_h, 3),
            "latent_disruption_h": np.round(disruption_h, 3),
            "actual_transit_h": np.round(total_h, 3),
        }
    )
    df["delay_h"] = (df["actual_transit_h"] - df["planned_transit_h"]).round(3)
    df["is_late"] = df["delay_h"] > cfg.late_tolerance_h
    df["arrival_ts"] = df["ship_ts"] + pd.to_timedelta(df["actual_transit_h"], unit="h")
    df = df.sort_values(["ship_ts", "shipment_id"], kind="stable").reset_index(drop=True)
    return df


def describe_distribution(df: pd.DataFrame, column: str = "actual_transit_h") -> dict[str, float]:
    """Summary statistics that show the shape of the transit distribution."""
    x = df[column].to_numpy(dtype=float)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    z = (x - mean) / std
    return {
        "n": float(x.size),
        "mean": mean,
        "median": float(np.median(x)),
        "std": std,
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "p99_over_p50": float(np.quantile(x, 0.99) / np.quantile(x, 0.50)),
        "max": float(x.max()),
    }

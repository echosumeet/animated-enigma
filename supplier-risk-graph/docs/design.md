# Design notes

## Why a typed graph and not a table

Multi-tier risk work usually dies in the data model. A spreadsheet of suppliers cannot
express that the same plant appears under four different tier-1 relationships, which is
exactly the fact that matters. `riskgraph` separates three node types: parts (what is
consumed), sites (where it is made) and suppliers (who is contracted). Disruption is a
property of the site, not the legal entity, because a supplier with two plants in two
countries is a different risk to a supplier with two plants in one industrial park.

The loader is pydantic-validated and refuses networks that would silently produce
nonsense: allocation shares that do not sum to one, dangling site references, BOM
cycles. In practice most of the effort in a real deployment is here, not in the maths.

## One kernel, three views

`flow.output_fraction` answers a single question: with this set of sites unavailable,
what fraction of the finished good can still be built. SPOF ranking, the Monte Carlo
simulation and mitigation scoring all call it. Availability of a part is the surviving
allocation share scaled by a ramp-up factor and capped at one; it propagates up the BOM
as a **minimum**, not a product, because components are not substitutes for one another.
Keeping one implementation is what stops the structural view and the simulation view
from quietly disagreeing when someone asks why the two decks show different numbers.

## Ranking in dollars, not in degrees

Graph centrality ranks the busiest node. The node that stops the line is often
unremarkable by degree: one plant, one part, one customer. Every SPOF candidate is
therefore scored by re-running the kernel with that site or supplier removed, and the
ranking is by revenue at risk. Articulation points are reported as an attribute rather
than as the ranking, since a cut vertex on a low-value branch is not an action item.
On the reference instance, degree centrality agrees with the revenue ranking on 3 of
the top 8 nodes, which is the whole argument for not shipping centrality as the answer.

## Concentration, and what it hides

HHI is computed on spend shares by tier, by supplier and by country. It is necessary
and insufficient. The reference instance reports a tier-1 HHI of 0.163 — about six
effective suppliers, a number that passes any sourcing review — while a single
depth-3 supplier sits under five of the tier-1 branches and takes 100% of revenue with
it. `hidden_dependencies` walks each sub-tier supplier back up to the set of tier-1
branches it reaches, and flags convergence. Ties on revenue share are broken by depth,
because the deeper node is the one nobody is looking at.

## Simulation choices

Each trial samples per-site disruption occurrence, start day and a lognormal recovery
duration. The year is cut at event boundaries and the kernel is evaluated per interval,
so overlapping outages compound rather than being averaged. Reported time-to-recover is
the longest single shortfall episode, not the span from first to last event: two
unrelated outages in one year are two recoveries.

Ramp-up flex defaults to 1.25. Zero flex is not the conservative default it looks like
— it is the assumption under which dual sourcing never pays, because splitting 100/0
into 65/35 just doubles the number of ways to lose part of the volume.

## Mitigation scoring

Each action is applied to a rebuilt, revalidated network and re-simulated under common
random numbers against a baseline drawn at the same trial count and seed. Comparing
against a baseline sampled with a different budget mixes Monte Carlo error into the
ranking and can flip the sign of a small effect. Cost models are explicit constants at
the top of `mitigation.py` so they can be replaced with real category numbers.

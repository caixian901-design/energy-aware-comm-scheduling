# 套磁段落（英文，可直接粘贴）

> 用法：先发**通用段落** + 针对该导师的**变体段落**，附 GitHub 链接与 Fig. T1（和可选的 T3）图。图与仓库链接比任何兴趣表述都有效。

---

## 1. 通用段落（任选一位导师，替换 [supervisor]）

```
Dear Prof. [supervisor],

I am applying for a Master's position in [department] and am writing because
my current research direction — energy- and carbon-aware operation of AI
computing centres — aligns closely with your work on [their topic].

My prior study (sole-authored preprint, doi:10.5281/zenodo.22743562) models
the PUE of hyperscale AI data centres in high-altitude low-pressure
environments, driven by measured TMYx weather, and shows that the plateau's
excellent free-cooling availability does not guarantee the best PUE once
air-side density effects are included.

I have since extended this facility model one level up: a distributed
training workload layer (compute + all-reduce phases), and an hourly carbon
accounting layer. The pilot sweeps the gradient-sync interval, the liquid
cooling fraction and the grid carbon profile, and quantifies a real
trade-off that the scheduling literature usually ignores:

  * making syncs 1x vs 64x more frequent adds ~7% to training makespan but
    cuts whole-facility carbon by ~16% (Xining, f = 0.80);
  * on a low-carbon, variable grid, shifting the training start time by a
    few hours changes emissions by up to ~57% with no hardware change.

The reproducible code and figures are at [GitHub URL]. Figure 1 shows the
makespan-vs-carbon trade-off for five liquid-cooling fractions and two
carbon profiles.

I would welcome the chance to discuss how this could grow into a thesis
under your supervision — e.g. turning time-optimal scheduling into a
time-energy-carbon multi-objective problem.
```

---

## 2. 变体 A：给 Liu / Fiech（MUN，通信调度团队）

```
I read your work on communication scheduling for distributed deep learning
(e.g. the IEEE LCN 2025 analysis of scheduling schemes, and the ProLet
APNet 2026 load-balancing line). Your analyses focus on how communication
shapes training time; my pilot asks the complementary question: how does the
same schedule shape facility energy, PUE and carbon? I quantify a concrete
tension — the time-optimal sync interval is not the carbon-optimal one
(~7% time vs ~16% carbon at Xining) — which suggests a natural extension:
communication-aware, energy-aware co-scheduling for AI clusters. I would be
very interested in pursuing this direction in your group.
```

## 3. 变体 B：给 Eager（USask，数据中心性能建模）

```
Your work on dynamic server allocation and request routing performance
modelling asks how well a data centre performs under real demand. My pilot
adds the energy side of the same coin: I model the facility's PUE from
measured weather and couple it to a training-workload model, so that
allocation and scheduling decisions can be evaluated on makespan AND
emissions. The result — a ~7% time / ~16% carbon trade-off in the sync
interval — is exactly the kind of analytically tractable problem your
modelling style is suited for.
```

## 4. 变体 C：给 Chanchal Roy（USask，LLM 能耗/碳）

```
Your Carbon-Taxed Transformers work addresses the cost and sustainability
of large-model workloads. My pilot approaches the same goal from the
facility side: a PUE model of an AI computing centre coupled with a
distributed-training workload and hourly grid carbon intensities. Combined,
these suggest that workload-level carbon efficiency (your direction) and
facility-level scheduling (my pilot) should be optimised jointly — e.g. a
carbon-aware training scheduler that shifts or re-batches jobs against a
variable grid signal. This is the direction I would like to pursue in a
Master's thesis.
```

---

## 3. 附图的引用说法（任选）

> The attached Figure 1 shows the makespan-vs-carbon frontier for five
> liquid-cooling fractions on a low-carbon (left) and a coal-heavy (right)
> grid profile; each point is a sync interval k = 1..64. All results are
> reproducible from the linked repository.

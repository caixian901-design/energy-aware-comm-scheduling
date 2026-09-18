# 套磁段落（英文，可直接粘贴）

> 用法：先发**通用段落** + 针对该导师的**变体段落**，附 GitHub 链接。
>
> **数据口径（重要）**：以下全部数字来自已发布的预印本 **A** 与 **C**，与
> `C:\Users\29102\Desktop\Canada` 里的 60 封信、CV、研究摘要完全一致。
> 早期 pilot 版本（512 加速器、210 组扫描、7.4%/14%、57%）**已废弃且无 DOI，
> 不要再引用**。
>
> - **A**（设施模型）：doi:10.5281/zenodo.22743561
> - **C**（主结果）：doi:10.5281/zenodo.22763084
> - 代码与图表：github.com/caixian901-design/energy-aware-comm-scheduling

---

## 1. 通用段落（任选一位导师，替换 [supervisor]）

```
Dear Prof. [supervisor],

I am applying for a Master's position in [department] and am writing because
my research direction - energy- and carbon-aware operation of AI computing
centres - aligns closely with your work on [their topic].

I have two open-access preprints. The first (doi:10.5281/zenodo.22743561)
models the cooling architecture and PUE of hyperscale AI data centres in
high-altitude low-pressure environments, and shows that the plateau's
free-cooling advantage does not by itself determine the best cooling
architecture once air-side density effects are included.

The second (doi:10.5281/zenodo.22763084) is the result I would most like to
discuss with you. It couples a compute-phase / communication-phase model of a
100,000-step training job on 10,000 accelerators to a temperature-dependent
cooling model and to time-varying grid carbon intensity, then compares ten
feasible schedules of the same job:

  * the mean PUE is identical to four decimal places (1.1490) across all ten
    schedules, while carbon per job spans 13.1x (4.91 to 64.25 tCO2e) - so the
    standard efficiency metric cannot rank the carbon performance of training
    workloads;
  * the job's start hour alone moves carbon by 54%, whereas the entire
    liquid-cooling range from f = 0 to 1 changes it by only 9.11%, identical
    at every point of the sweep;
  * the marginal carbon cost of one accelerator-hour of makespan is exactly
    P_comm x CI, fixed by the accelerator's idle-phase power and the grid, and
    independent of the cooling design.

All code and figures are openly available at
github.com/caixian901-design/energy-aware-comm-scheduling.

I would welcome the chance to discuss how this could grow into a thesis under
your supervision - for example, reporting energy-to-solution and carbon per
delivered step alongside PUE, and replacing the open-loop schedule choice with
model-predictive or learning-based control.
```

---

## 2. 变体 A：给做通信调度 / 分布式训练网络的导师

```
I read your work on communication scheduling for distributed deep learning.
Your analyses focus on how communication shapes training time; my preprint
(doi:10.5281/zenodo.22763084) asks the complementary question: how does the
same schedule shape facility energy and carbon? Across ten feasible schedules
of one 100,000-step job on 10,000 accelerators, the mean PUE is identical to
four decimal places while carbon per job spans 13.1x - the schedule is a
carbon decision that the standard metric cannot see. That suggests a natural
extension: communication-aware, carbon-aware co-scheduling for AI clusters,
with the marginal cost of time priced as P_comm x CI. I would be very
interested in pursuing this in your group.
```

## 3. 变体 B：给做数据中心性能建模 / 云资源分配的导师

```
Your work on server allocation and request routing performance asks how well a
data centre performs under real demand. My preprints add the energy and carbon
side of the same coin: a temperature-dependent facility model coupled to a
distributed-training workload and time-varying grid carbon intensity
(doi:10.5281/zenodo.22763084). The result is analytically clean - across ten
schedules of the same job the mean PUE is constant to four decimal places
while carbon spans 13.1x, and the marginal carbon cost of an accelerator-hour
of makespan is exactly P_comm x CI - which is the kind of tractable problem
your modelling style is suited to.
```

## 4. 变体 C：给做 LLM 能耗 / 绿色 AI 的导师

```
Your work on the cost and sustainability of large-model workloads addresses
the same goal from the algorithm side. My preprints approach it from the
facility side: a cooling and PUE model (doi:10.5281/zenodo.22743561) coupled
to a distributed-training workload and hourly grid carbon intensity
(doi:10.5281/zenodo.22763084). The finding that PUE is structurally invariant
to the compute duty cycle - and therefore cannot rank training workloads by
carbon - suggests that workload-level carbon efficiency and facility-level
scheduling should be optimised jointly, and reported as energy-to-solution and
carbon per delivered step. This is the direction I would like to pursue in a
Master's thesis.
```

---

## 5. 附件的引用说法（任选）

```
The attached two-page summary gives the model, the ten-schedule comparison and
the limitations; the preprints and the code are linked inside it. All results
are model-derived and reproduced by openly available code.
```

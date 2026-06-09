# Project Brief for Job-Search Agent · 求职 Agent 项目说明文档

> **Purpose / 用途.** A self-contained knowledge pack so an AI agent can pitch,
> tailor, and defend this project across CVs, cover letters, recruiter chats and
> interviews — at any depth, in English or Chinese.
> **本文件用途**：让 AI agent 在简历、求职信、招聘官沟通、面试中，以任意深度、中英文，
> 准确地介绍、定制并答辩这个项目。
>
> **Owner / 作者:** Jiaming Wei (UCL). **Repo:** https://github.com/Quarkgluonmixture/africa_china_poverty
> **Primary target roles / 主要目标岗位:** Research Engineer / ML Research (primary), ML / Deep-Learning Engineer (secondary).
>
> **Hard rule for the agent / 给 agent 的硬性规则:** never state a number not in
> the [Fact Sheet](#fact-sheet--事实清单); if unsure, speak qualitatively. Do not
> overclaim production scale or real-world deployment — this is a rigorous
> research project, not a shipped product.
>
> **Claims discipline (do NOT reintroduce these) / 措辞红线（不得复活）:**
> - Pretraining does **not** "double" accuracy. From-scratch ResNet-50 reaches
>   r²=0.615 vs pretrained 0.651 — a **modest** gap; the real benefit is faster
>   convergence (best epoch 14 vs 51). Quote epochs, not invented multipliers
>   (no "~10×"). 预训练**不**翻倍精度；优势主要在收敛速度，用 epoch 数说话，别用编造的倍数。
> - From-scratch did **not** "barely learn" — 0.615 is a solid result. 从零**不是**学不动。
> - Transfer is **architecture-insensitive**: do NOT say one backbone transfers
>   better. The single-seed "1.49 vs 1.13" did **not** replicate (paired Δgap CI
>   includes 0). Only "transfer is robust (both gaps' CIs clear of 0)" and
>   "ConvNeXt is better *in-domain*" are supported. 迁移对架构不敏感；"1.49 vs 1.13"未复现，
>   不得说某个 backbone 迁移更好。
> - For quantitative transfer claims use the **multi-seed CIs**, not single-seed
>   point values (those are illustrative only). 定量主张一律用多 seed CI，单 seed 数字仅作举例。

---

## 1. One-liner · 一句话

**EN:** A deep-learning study that predicts local economic well-being from
satellite imagery across five African countries, and tests whether the model
generalizes *zero-shot* to rural China — combining transfer-learning ablations,
multi-architecture benchmarking, and Grad-CAM interpretability.

**中文:** 用卫星影像预测非洲五国的局部经济发展水平，并检验模型能否**零样本**泛化到中国农村
的深度学习研究——涵盖迁移学习消融、多架构基准对比与 Grad-CAM 可解释性分析。

---

## 2. Elevator pitches (three lengths) · 三种长度的电梯陈述

### 15 seconds · 15 秒
**EN:** I built a PyTorch system that estimates a wealth index from satellite
images. It reaches test r²≈0.69 on African survey data and, with no retraining,
transfers to a hand-built adversarial dataset in rural China — separating rich
from poor areas with a clear margin.

**中文:** 我用 PyTorch 做了一个从卫星图估计财富指数的系统，在非洲调查数据上达到
test r²≈0.69，并且无需重训就能迁移到我自建的中国农村对抗数据集，清晰区分贫富区域。

### 30 seconds · 30 秒
**EN:** Predicting poverty from space is a sustainable-development problem: ground
surveys are expensive, satellites are free. I trained ImageNet-pretrained
backbones (ResNet-50, ConvNeXt, ViT) to regress the DHS wealth index over five
African countries, and ran an ablation showing transfer learning mainly buys
much faster convergence (plus a modest accuracy gain) on ~2k images. Then I
stress-tested
generalization with a purpose-built dataset of 20 Guizhou locations — including
cave dwellings invisible to optical sensors and resettlement villages that mimic
suburbs — and used Grad-CAM to confirm the model keys on built-up structures.

**中文:** "从太空看贫困"是个可持续发展问题：地面调查昂贵，卫星影像免费。我用 ImageNet 预训练
的 backbone（ResNet-50、ConvNeXt、ViT）回归非洲五国的 DHS 财富指数，并通过消融实验表明：
在仅 ~2k 张图上，迁移学习主要带来更快的收敛（外加小幅精度提升）。随后我用自建的 20 个贵州地点数据集压力测试
泛化能力——包括光学卫星不可见的洞穴民居、以及外观酷似富裕郊区的易地搬迁村——并用 Grad-CAM
验证模型确实聚焦于建筑密度。

### 2 minutes · 2 分钟
**EN:** The starting point is a known idea — Jean/Yeh et al. showed you can
predict economic well-being from satellite imagery. I rebuilt that pipeline from
scratch in modern PyTorch and pushed it in a research direction I cared about:
**does it generalize across continents?**

First, the Africa model. I regress the continuous DHS wealth index from
Sentinel-2 RGB tiles for Nigeria, Malawi, Rwanda, Uganda and Tanzania. I
benchmarked three pretrained backbones under identical training (AdamW, cosine
schedule, bf16, early stopping) and got test r² of 0.65–0.69, with ConvNeXt-Tiny
best. Crucially I ran a from-scratch ablation: the same ResNet-50 without
ImageNet weights lands a little lower (0.62 vs 0.65) and, more tellingly, only
reaches its best around epoch 51 versus epoch 14 for the pretrained model — so on
small data the main thing transfer learning buys is much faster, more stable
convergence rather than a big final-accuracy gap.

Second, the transfer study. I hand-curated 20 Guizhou locations with an
adversarial design — places chosen to break optical poverty mapping. Applying
the Africa-trained model with zero fine-tuning, it reliably separates developed
from poor areas: across 8 seeds plus a bootstrap on the 20 tiles, the
developed/poor gap is ~1.1 with 95% CIs well clear of 0 for both backbones. The
methodologically interesting part: a single seed had suggested ResNet-50
transferred noticeably better than the stronger in-domain model (ConvNeXt), but
when I replicated it across 8 paired seeds the difference vanished (paired Δgap
CI includes 0). So I reported the robust finding — transfer works and is
architecture-insensitive at this scale — rather than the flashy single-seed one.

Finally, interpretability: Grad-CAM on the trained model shows attention landing
on airport terminals, CBD towers and dense housing for urban tiles, and staying
diffuse for rural ones. Everything is config-driven and reproducible; I trained
it on an ARM64 Grace-Blackwell GPU and included an HPC batch script.

**中文:** 出发点是个已知思路——Jean/Yeh 等人证明可以用卫星影像预测经济发展水平。我用现代
PyTorch 从零重建了这条管线，并把它推向我关心的研究方向：**它能跨大洲泛化吗？**

第一步是非洲模型。我从尼日利亚、马拉维、卢旺达、乌干达、坦桑尼亚的 Sentinel-2 RGB 图块回归
连续的 DHS 财富指数。在完全相同的训练设置（AdamW、cosine 调度、bf16、早停）下对比三个预训练
backbone，test r² 在 0.65–0.69 之间，ConvNeXt-Tiny 最好。关键是我做了从零训练的消融：同样的
ResNet-50 去掉 ImageNet 权重后 r² 略低（0.62 vs 0.65），更说明问题的是它要到约第 51 个 epoch 才
到达最佳，而预训练模型在第 14 个 epoch 就到了——所以在小数据上，迁移学习主要带来更快、更稳的收敛，
而非巨大的最终精度差。

第二步是迁移研究。我手工构建了 20 个贵州地点，采用对抗性设计——专门挑选会让光学贫困识别失效
的地方。把非洲训练的模型零微调地用上去，它能稳定区分发达与贫困区域：在 8 个 seed + 对 20 个点
的 bootstrap 下，发达/贫困差距约 1.1，两个 backbone 的 95% CI 都明显大于 0。方法论上最有意思的
是：单个 seed 曾显示 ResNet-50 迁移得比域内更强的 ConvNeXt 明显更好，但我在 8 个配对 seed 上复现
后，这个差异消失了（配对 Δgap 的 CI 包含 0）。所以我报告的是稳健的结论——迁移有效、且在此规模下
对架构不敏感——而非那个抢眼但脆弱的单 seed 结果。

最后是可解释性：训练后模型的 Grad-CAM 显示，城市图块的注意力落在机场航站楼、CBD 高楼和密集
住宅上，乡村图块则保持弥散。整个项目都是配置驱动、可复现的；我在 ARM64 的 Grace-Blackwell
GPU 上训练，并附带了 HPC 批处理脚本。

---

## 3. Problem & motivation · 问题与动机

**EN:** Reliable poverty statistics require household surveys that are costly and
infrequent, leaving large gaps in low-income regions. Satellite imagery is
global, cheap and frequent, so learning to read socioeconomic signal from it is
a high-leverage tool for the UN Sustainable Development Goals. The open research
question this project targets is **geographic transfer**: a model trained where
labels exist (Africa) being applied where they are scarce (here, rural China),
and understanding *when that transfer holds and when it breaks*.

**中文:** 可靠的贫困统计依赖入户调查，成本高、频率低，导致低收入地区数据严重缺失。卫星影像
全球覆盖、廉价、高频，因此"从影像读出社会经济信号"是服务联合国可持续发展目标的高杠杆工具。
本项目针对的开放研究问题是**地理迁移**：在有标签的地方（非洲）训练的模型，应用到标签稀缺的
地方（这里是中国农村），并理解**迁移在什么情况下成立、什么情况下失效**。

---

## 4. What I built · 我做了什么

**EN (technical contributions):**
- A clean, installable PyTorch library (`src/acp`): dataset + country-stratified
  and leave-one-country-out splits, a timm backbone factory with a regression
  head, a training/eval engine (AMP, cosine+warmup, early stopping, proper
  metrics), and a from-scratch Grad-CAM for a regression output.
- Config-driven experiments (one YAML per run) and a one-command pipeline that
  trains every backbone, runs the from-scratch ablation, and emits a comparison
  table + plot.
- An original, adversarial 20-location Guizhou evaluation set with a written
  geospatial rationale, plus zero-shot inference and Grad-CAM scripts for it.
- Reproducibility & infra: seeded runs, committed figures, an environment spec,
  and a UCL Myriad (SGE) batch script; trained on an NVIDIA GB10
  (Grace-Blackwell, ARM64) with a CUDA-12.8 PyTorch build.

**中文（技术贡献）:**
- 一个干净、可安装的 PyTorch 库（`src/acp`）：数据集 + 国家分层划分与留一国划分、基于 timm 的
  backbone 工厂 + 回归头、训练/评估引擎（混合精度、cosine+warmup、早停、规范指标），以及为
  回归输出自实现的 Grad-CAM。
- 配置驱动的实验（每个 run 一个 YAML）+ 一条命令的流水线：训练所有 backbone、跑从零消融、
  输出对比表与图。
- 原创的、对抗性的 20 地点贵州评估集，附带书面的地理空间设计依据，以及对应的零样本推理与
  Grad-CAM 脚本。
- 可复现与基础设施：固定随机种子、提交图表、环境规范、UCL Myriad（SGE）批处理脚本；在 NVIDIA
  GB10（Grace-Blackwell，ARM64）+ CUDA-12.8 PyTorch 构建上完成训练。

---

## 5. Results & what they mean · 结果及其含义

| Result | Number | Interpretation · 解读 |
|---|---|---|
| Best Africa model (ConvNeXt-Tiny) | test r² **0.692**, RMSE 0.503 | strong in-domain regression of a continuous wealth index · 对连续财富指数的强域内回归 |
| In-domain edge (paired, 8 seeds) | ConvNeXt − ResNet-50 Δr²=+0.025, 95% CI [0.012, 0.038], p=0.003, 8/8 | ConvNeXt is *significantly* better in-domain (small but real) · ConvNeXt 域内显著更好（小而真实） |
| Transfer-learning ablation (same ResNet-50) | pretrained 0.651 (best ep 14) vs from-scratch 0.615 (best ep 51) | pretraining's main win on small data is convergence speed/stability, not a big accuracy gap · 小数据上预训练主要赢在收敛速度/稳定性，而非巨大精度差 |
| Africa→China zero-shot (8 seeds + bootstrap) | gap ResNet-50 1.18 [1.03,1.33], ConvNeXt 1.08 [0.99,1.16] | transfer is robust: both CIs well clear of 0 · 迁移稳健：两条 CI 都明显大于 0 |
| In-domain ≠ transfer advantage | paired Δgap +0.10, 95% CI **[−0.05, +0.25]** (incl. 0), p=0.15 | the in-domain winner is **not** a better transferrer — they tie; the single-seed "1.49 vs 1.13" did not replicate · 域内赢家迁移上并不更好——打平；单 seed 的 "1.49 vs 1.13" 未能复现 |
| Adversarial cases (representative model) | cave −0.05, relocation 0.12 | model resists the two designed traps · 模型抵住了两个设计陷阱 |

**EN:** The story is coherent and — importantly — honestly tested. Pretraining
matters most when data is scarce (mainly convergence speed); the learned
representation generalises across continents (robust positive gap under
multi-seed + bootstrap); and a suggestive single-seed difference between
architectures did **not** survive replication, so the reported conclusion is the
robust one (transfer works, architecture-insensitive). Testing your own
flattering result and reporting that it failed is the point.

**中文:** 整个故事自洽，而且——关键是——经过了诚实检验。数据稀缺时预训练最关键（主要是收敛
速度）；学到的表征能跨大洲泛化（多 seed + bootstrap 下稳健的正 gap）；而架构之间一个抢眼的单
seed 差异**没能**通过复现，所以我报告的是稳健结论（迁移有效、对架构不敏感）。去检验自己那个好看
的结果、并如实报告它不成立，正是要点所在。

---

## 6. Skills matrix (maps project → JD keywords) · 技能矩阵（项目→JD 关键词）

| JD keyword · JD 关键词 | Evidence in this project · 本项目中的证据 |
|---|---|
| PyTorch, deep learning | full pipeline from data to Grad-CAM in PyTorch + timm |
| Transfer learning / representation learning | pretrained-vs-scratch ablation; cross-continent zero-shot |
| Experimental design / ablation | controlled backbone comparison; isolated pretraining variable |
| Generalization / OOD / domain shift | Africa→China zero-shot; leave-one-country-out protocol |
| Model interpretability / XAI | Grad-CAM implemented from scratch for a regression head |
| CV / remote sensing | Sentinel-2 imagery, satellite tiles, geospatial reasoning |
| Reproducible research | config-driven runs, seeding, committed figures, env spec |
| GPU / HPC | Grace-Blackwell ARM64 training; SGE/qsub batch script |
| Statistical rigor | multi-seed paired t-tests, sign test, t- & bootstrap CIs; replicated a single-seed result and reported it didn't hold |
| Scientific communication | honest reporting incl. a non-replication; clear README |
| Data curation | original adversarial Guizhou dataset with documented rationale |

---

## 7. STAR stories (behavioral interviews) · STAR 行为故事

**Story A — Quantifying transfer learning with a controlled ablation.**
- **S/T:** I needed the best accuracy I could get from only ~2k labelled tiles,
  and wanted to *know* (not assume) how much ImageNet pretraining actually helps.
- **A:** I ran the same ResNet-50 with and without pretrained weights under
  identical training, holding the data split fixed.
- **R:** Pretraining gave a modest final-accuracy gain (test r² 0.65 vs 0.62) but
  a large convergence gain — it reached its best by epoch 14 vs epoch 51 — so I
  could recommend pretraining on evidence and characterise *why* it helps.
- **中文:** 我想在仅 ~2k 张图上拿到最好的精度，并且要*知道*（而非假设）ImageNet 预训练到底帮多少。
  于是在固定数据划分、相同训练设置下，对同一个 ResNet-50 做了"有/无预训练"消融。结果：预训练带来
  小幅最终精度提升（test r² 0.65 vs 0.62），但收敛优势很大——第 14 个 epoch 就到最佳，而从零要到
  第 51 个——于是我能基于证据推荐预训练，并讲清它*为什么*有用。

**Story B — Replicating my own suggestive result instead of shipping it.**
- **S/T:** A single seed showed the Africa→China transfer gap was larger for
  ResNet-50 (1.49) than for the stronger in-domain model ConvNeXt (1.13) — a
  tempting "in-domain ≠ transfer" headline.
- **A:** Instead of reporting it, I ran an 8-seed paired study (same split per
  seed) plus a stratified bootstrap on the 20 China points, with paired t-tests
  and a sign test.
- **R:** The difference did **not** replicate (paired Δgap 95% CI [−0.05, +0.25],
  includes 0). I reported the robust finding instead — transfer is reliable
  (both gaps' CIs clear of 0) but architecture-insensitive — and corrected the
  single-seed number. Testing my own flattering result is the takeaway.
- **中文:** 单个 seed 显示 ResNet-50 的迁移 gap(1.49) 大于域内更强的 ConvNeXt(1.13)，很容易写成
  "域内≠迁移"的抢眼结论。我没有直接报告，而是做了 8 seed 配对研究 + 对 20 个中国点的分层 bootstrap，
  配上配对 t 检验与 sign test。结果该差异**未能复现**（配对 Δgap 95% CI [−0.05,+0.25] 含 0）。于是我
  报告稳健结论——迁移可靠（两条 CI 都离 0 很远）但对架构不敏感——并更正了那个单 seed 数字。检验
  自己好看的结果，才是要点。

**Story C — Shipping on constrained, shared, bleeding-edge hardware.**
- **S/T:** Train on an ARM64 Grace-Blackwell GPU (no off-the-shelf wheels) that
  was shared with other heavy jobs.
- **A:** Stood up a CUDA-12.8 PyTorch build for sm_121, used bf16 + small-batch
  to coexist with neighbours, watched memory before launching, and wrote an HPC
  fallback (SGE/qsub) script.
- **R:** Reproducible multi-backbone training (~14 s/epoch) without disrupting
  shared users.
- **中文:** 在没有现成 wheel 的 ARM64 Grace-Blackwell、且与他人共享的 GPU 上训练；我搭好
  sm_121 的 CUDA-12.8 PyTorch、用 bf16+小 batch 与他人共存、启动前先看显存、并写了 HPC
  (SGE/qsub) 退路脚本，最终在不打扰共享用户的前提下完成可复现的多 backbone 训练。

---

## 8. Anticipated technical Q&A · 预期技术问答

**Q: Why regression, not rich/poor classification?**
The DHS wealth index is continuous; regressing it preserves ordering and
magnitude and matches the source literature (reported as squared Pearson r²).
中文：DHS 财富指数是连续量，回归能保留次序与幅度，并与原文献（用 squared Pearson r²）一致。

**Q: Isn't ~2k images tiny? How do you avoid overfitting?**
That's exactly why transfer learning is the core lever; plus dropout, weight
decay, strong augmentation, early stopping on validation, and held-out test +
optional leave-one-country-out evaluation.
中文：正因如此迁移学习才是核心手段；并辅以 dropout、weight decay、强增广、按验证集早停，以及
留出测试集 +（可选）留一国评估。

**Q: How is the China transfer "zero-shot"?**
No Chinese image or label is used in training or fine-tuning; the Africa-trained
weights are applied directly to Guizhou tiles.
中文：训练与微调都不使用任何中国图像或标签；直接把非洲训练的权重用到贵州图块上。

**Q: Could leakage explain the African scores?**
Splits are country-stratified and image-disjoint by cluster id; the
leave-one-country-out protocol further tests generalization to unseen countries.
中文：划分按国家分层、按 cluster id 图像不重叠；留一国协议进一步检验对未见国家的泛化。

**Q: What does Grad-CAM actually prove?**
It's evidence, not proof — but consistent activation on built-up structures
(vs diffuse on rural tiles) is a sanity check that the model uses plausible cues.
中文：它是证据而非证明——但在建筑结构上的一致激活（相对乡村的弥散）能合理印证模型用了可信线索。

**Q: Limitations?** See §9.

---

## 9. Honest limitations & next steps · 诚实的局限与后续

**EN:**
- RGB-only; the source paper also uses nightlights/multispectral, which would
  likely raise accuracy.
- The China set is small (20 points). The transfer gap is multi-seed + bootstrap
  CI'd (so the n=20 uncertainty is quantified, not ignored), but a larger
  labelled Chinese benchmark is the natural next step.
- Multi-seed (8) is done for ResNet-50 & ConvNeXt; extending the seed study to
  ViT and a full leave-one-country-out sweep would broaden the claims further.
- Wealth index is a proxy for assets, not income; interpret accordingly.

**中文:**
- 仅用 RGB；原文献还用了夜间灯光/多光谱，加入后精度可能更高。
- 中国集很小（20 点）。迁移 gap 已用多 seed + bootstrap 给出 CI（n=20 的不确定性被量化而非忽略），
  但更大的中国标注基准是自然的下一步。
- 已对 ResNet-50 与 ConvNeXt 做了 8 seed；把 seed 研究扩到 ViT、并做完整留一国扫描会进一步拓宽结论。
- 财富指数是资产的代理量，而非收入；解读时需注意。

---

## 10. Tailoring guidance for the agent · 给 agent 的定制建议

**EN:**
- **Research Engineer / ML Research JD →** lead with the ablation, the
  multi-seed transfer study with paired tests + bootstrap CIs (and the
  self-replication that corrected a single-seed result), and Grad-CAM; frame as
  "designed and ran controlled, statistically tested experiments to answer a
  generalization question — and reported what replicated, not what flattered."
- **ML / DL Engineer JD →** lead with the PyTorch library design, config-driven
  reproducible pipeline, multi-backbone benchmarking, AMP/HPC, and clean repo.
- **Data Scientist JD →** lead with problem framing, the wealth-index target,
  metrics interpretation, and the sustainable-development impact.
- **Always:** quote only Fact-Sheet numbers; pair every claim with its evidence;
  if asked about scale/deployment, be honest that it's a research project.

**中文:**
- **研究向 JD →** 以消融、零样本迁移、negative result、Grad-CAM 为主线；定位为"设计并执行受控
  实验来回答泛化问题"。
- **ML/DL 工程 JD →** 以 PyTorch 库设计、配置驱动可复现流水线、多 backbone 基准、混合精度/HPC、
  干净仓库为主线。
- **数据科学 JD →** 以问题建模、财富指数目标、指标解读、可持续发展影响为主线。
- **始终：** 只引用事实清单中的数字；每个论断都配上证据；被问到规模/部署时，诚实说明这是研究项目。

---

## Fact Sheet · 事实清单

> The agent must not state numbers outside this list. · agent 不得使用本清单以外的数字。

- **Data / 数据:** ~2,006 labelled Sentinel-2 RGB tiles (of 3,136 DHS clusters);
  5 countries — Nigeria 889, Malawi 827, Rwanda 492, Uganda 470, Tanzania 458.
  Wealth index ≈ [−1.68, 4.73], mean ≈ 0.04, std ≈ 0.84.
- **Split / 划分:** country-stratified ~70/15/15 → train 1,404 / val 301 / test 301;
  optional leave-one-country-out.
- **Africa test results / 非洲测试结果 (held-out, single representative seed):**
  ConvNeXt-Tiny r²=0.692 (R²=0.689, RMSE=0.503); ViT-S/16 r²=0.687 (R²=0.683);
  ResNet-50 r²=0.651 (R²=0.650); ResNet-50 from scratch r²=0.615 (R²=0.614,
  RMSE=0.560, best epoch 51 vs pretrained's epoch 14).
- **Multi-seed (8 seeds, paired by split) / 多 seed（8 个，按 split 配对）:**
  Africa r² — ResNet-50 0.653 [0.633,0.673], ConvNeXt 0.678 [0.648,0.709];
  paired Δr² (ConvNeXt−ResNet) +0.025, 95% CI [0.012,0.038], p=0.003, 8/8 seeds
  → ConvNeXt significantly better in-domain.
- **China zero-shot / 中国零样本:** 20 tiles (10 developed, 10 poor).
  Multi-seed gap — ResNet-50 1.18 [1.03,1.33], ConvNeXt 1.08 [0.99,1.16]
  (hier-bootstrap lower bounds ≈0.78, both clear of 0). Paired Δgap +0.10,
  95% CI [−0.05,+0.25], p=0.15 → architectures statistically tied on transfer
  (the single-seed 1.49 vs 1.13 did NOT replicate). Representative-model
  adversarial preds: Zhongdong cave −0.05, Huawu relocation 0.12.
- **Backbones / 模型:** ResNet-50 (23.5M params), ConvNeXt-Tiny, ViT-S/16; all
  ImageNet-pretrained via `timm`; single linear regression head.
- **Training / 训练:** AdamW, cosine schedule + warmup, bf16 AMP, dropout +
  weight decay, early stopping on val r², seed 42.
- **Compute / 算力:** NVIDIA GB10 (Grace-Blackwell, ARM64, sm_121), PyTorch
  2.x + CUDA 12.8; ~14 s/epoch for ResNet-50 at batch 64; UCL Myriad (SGE) script provided.
- **Stack / 技术栈:** Python 3.11, PyTorch, timm, torchvision, scikit-learn,
  pandas, matplotlib. Data via Google Earth Engine (Sentinel-2) + DHS surveys.
- **Provenance / 来源:** PyTorch reimplementation extending Yeh et al. 2020
  (*Nature Communications*); MIT licensed. Author: Jiaming Wei.

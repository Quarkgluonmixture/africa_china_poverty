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
African countries, and ran an ablation showing transfer learning more than
doubles performance versus from-scratch on ~2k images. Then I stress-tested
generalization with a purpose-built dataset of 20 Guizhou locations — including
cave dwellings invisible to optical sensors and resettlement villages that mimic
suburbs — and used Grad-CAM to confirm the model keys on built-up structures.

**中文:** "从太空看贫困"是个可持续发展问题：地面调查昂贵，卫星影像免费。我用 ImageNet 预训练
的 backbone（ResNet-50、ConvNeXt、ViT）回归非洲五国的 DHS 财富指数，并通过消融实验证明：
在仅 ~2k 张图上，迁移学习比从零训练性能翻倍以上。随后我用自建的 20 个贵州地点数据集压力测试
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
ImageNet weights reaches a clearly lower r² and, more strikingly, needs ~10×
more epochs to get there — which quantifies how much transfer learning buys you
on small data, especially in convergence speed.

Second, the transfer study. I hand-curated 20 Guizhou locations with an
adversarial design — places chosen to break optical poverty mapping. Applying
the Africa-trained model with zero fine-tuning, it still separates developed
from poor areas with a predicted-wealth gap of about 1.5. The interesting
negative result: the best in-domain model (ConvNeXt) actually transfers with a
*smaller* gap than ResNet-50 — in-domain accuracy isn't the same as transfer
robustness, which matters if you deploy these models in new regions.

Finally, interpretability: Grad-CAM on the trained model shows attention landing
on airport terminals, CBD towers and dense housing for urban tiles, and staying
diffuse for rural ones. Everything is config-driven and reproducible; I trained
it on an ARM64 Grace-Blackwell GPU and included an HPC batch script.

**中文:** 出发点是个已知思路——Jean/Yeh 等人证明可以用卫星影像预测经济发展水平。我用现代
PyTorch 从零重建了这条管线，并把它推向我关心的研究方向：**它能跨大洲泛化吗？**

第一步是非洲模型。我从尼日利亚、马拉维、卢旺达、乌干达、坦桑尼亚的 Sentinel-2 RGB 图块回归
连续的 DHS 财富指数。在完全相同的训练设置（AdamW、cosine 调度、bf16、早停）下对比三个预训练
backbone，test r² 在 0.65–0.69 之间，ConvNeXt-Tiny 最好。关键是我做了从零训练的消融：同样的
ResNet-50 去掉 ImageNet 权重后 r² 明显更低，更突出的是要多花约 10 倍的 epoch 才能到达——
这量化了小数据上迁移学习的价值，尤其在收敛速度上。

第二步是迁移研究。我手工构建了 20 个贵州地点，采用对抗性设计——专门挑选会让光学贫困识别失效
的地方。把非洲训练的模型零微调地用上去，它仍能以约 1.5 的财富预测差距区分发达与贫困区域。
有趣的 negative result 是：域内最好的模型（ConvNeXt）迁移时差距反而更小——域内精度不等于迁移
鲁棒性，这在把模型部署到新区域时很重要。

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
| Transfer-learning ablation (same ResNet-50) | pretrained 0.651 vs from-scratch 0.615, but ~10× faster convergence | pretraining's main win on small data is convergence speed/stability, not a huge accuracy gap · 小数据上预训练的主要优势是收敛速度/稳定性，而非巨大精度差 |
| Africa→China zero-shot (ResNet-50) | rich/poor gap **1.49** | cross-continent signal survives with no fine-tuning · 跨大洲信号在零微调下仍成立 |
| Transfer-robustness nuance | ConvNeXt gap 1.13 < ResNet-50 1.49 | best in-domain ≠ best transfer (negative result) · 域内最好≠迁移最好 |
| Adversarial cases | cave −0.05, relocation 0.12 | model resists the two designed traps · 模型抵住了两个设计陷阱 |

**EN:** The story is coherent: pretraining matters most when data is scarce; the
learned representation is general enough to cross continents; and the honest
caveat — transfer robustness is not predicted by in-domain accuracy — is exactly
the kind of finding that matters for real deployment.

**中文:** 整个故事自洽：数据稀缺时预训练最关键；学到的表征足够通用以跨大洲；而那个诚实的
注脚——迁移鲁棒性不能由域内精度预测——恰恰是真实部署时最该关心的发现。

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
| Scientific communication | honest reporting incl. a negative result; clear README |
| Data curation | original adversarial Guizhou dataset with documented rationale |

---

## 7. STAR stories (behavioral interviews) · STAR 行为故事

**Story A — Making a non-learning model work (transfer learning).**
- **S/T:** A from-scratch CNN on only ~2k labelled tiles barely learned the
  regression. I needed usable accuracy without more labels.
- **A:** I diagnosed it as a small-data problem rather than a tuning problem,
  switched to ImageNet-pretrained backbones, and built a controlled ablation to
  *prove* the effect rather than assume it.
- **R:** Test r² jumped from ~half to ~0.69; I could state the value of
  pretraining quantitatively.
- **中文:** ~2k 张图上从零训练的 CNN 几乎学不动；我判断这是小数据问题而非调参问题，改用
  ImageNet 预训练并设计消融来**证明**效果，r² 从约一半提升到 ~0.69，并能定量说明预训练的价值。

**Story B — Designing an adversarial test instead of an easy one.**
- **S/T:** Reporting one accuracy number is weak evidence of generalization.
- **A:** I hand-built a 20-location Guizhou set targeting optical poverty-mapping
  failure modes (cave dwellings, suburb-mimicking resettlement) and ran the
  Africa model zero-shot.
- **R:** A clear rich/poor gap (1.49) *and* an honest negative result (in-domain
  ≠ transfer), which is more informative than a single score.
- **中文:** 单一精度数字不足以证明泛化；我手工构建对抗性的 20 地点贵州集，专攻光学识别的失效
  模式，零样本测试，得到清晰的贫富差距(1.49)与诚实的 negative result，比单一分数更有信息量。

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

**Story D — Choosing the honest result over the flattering one.**
- **S/T:** The best-accuracy model wasn't the best at transfer.
- **A:** I reported the discrepancy explicitly rather than cherry-picking the
  headline model for every figure.
- **R:** A more credible, decision-useful conclusion about deployment.
- **中文:** 精度最高的模型并非迁移最好的；我明确报告了这个矛盾，而非为每张图都挑那个最亮眼的
  模型，从而得到对部署更可信、更有决策价值的结论。

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
- The China set is small (20 points) — a qualitative stress test, not a
  quantitative Chinese benchmark; next step is a larger labelled Chinese set.
- In-country and leave-one-country-out are implemented; a full LOCO sweep across
  all five countries would strengthen the generalization claim.
- Wealth index is a proxy for assets, not income; interpret accordingly.

**中文:**
- 仅用 RGB；原文献还用了夜间灯光/多光谱，加入后精度可能更高。
- 中国集很小（20 点）——是定性压力测试，而非定量的中国基准；后续应构建更大的中国标注集。
- 已实现域内与留一国评估；对五国做完整 LOCO 扫描会进一步强化泛化结论。
- 财富指数是资产的代理量，而非收入；解读时需注意。

---

## 10. Tailoring guidance for the agent · 给 agent 的定制建议

**EN:**
- **Research Engineer / ML Research JD →** lead with the ablation, the zero-shot
  transfer, the negative result, and Grad-CAM; frame as "designed and ran
  controlled experiments to answer a generalization question."
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
- **Africa test results / 非洲测试结果 (held-out):**
  ConvNeXt-Tiny r²=0.692 (R²=0.689, RMSE=0.503); ViT-S/16 r²=0.687 (R²=0.683);
  ResNet-50 r²=0.651 (R²=0.650); ResNet-50 from scratch r²=0.615 (R²=0.614,
  RMSE=0.560, best epoch 51 vs pretrained's epoch 14).
- **China zero-shot / 中国零样本:** 20 tiles (10 developed, 10 poor).
  ResNet-50 gap 1.49 (developed mean 1.835 vs poor 0.344); ConvNeXt gap 1.13.
  Adversarial: Zhongdong cave −0.05, Huawu relocation 0.12.
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

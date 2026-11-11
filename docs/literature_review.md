# Literature review: e-nose coffee quality, machine olfaction and data fusion

**Scope.** This review covers six topics, following brief Sec. 7, Phase 2: (1) e-nose principles and MOX sensors, (2) e-noses for coffee, including every coffee paper cited by the Source 1 article, (3) machine olfaction with learning methods and larger datasets, (4) data fusion in food quality assessment, (5) coffee cupping and the CQI/SCA scoring system, and (6) validation practice on small datasets.

**How references were verified.** Every DOI was resolved against the Crossref API on 2026-11-11. The returned title, authors, year and venue are stored in `references_verified.csv` (36 of 36 found). Abstracts were retrieved from Semantic Scholar, OpenAlex or Crossref into `references_abstracts.json`. For a few papers only an auto-generated "TLDR" was available, and the table marks those entries; their details beyond the title are kept minimal on purpose. Items without a DOI were checked on the publisher's page: JMLR (Cawley & Talbot 2010; Demšar 2006), PMLR (Radford et al. 2021), Open Library (Gardner & Bartlett 1999), DataCite (the S1 dataset) and GitHub (the CQI database). The Source 1 article's own reference list was retrieved from Europe PMC (PMC3270826), so the list of "works it cites" is exact. The SmellNet paper was read in full (arXiv v5, the ICLR 2026 conference version).

---

## 1. E-nose principles and metal-oxide (MOX) sensors

An electronic nose combines an array of partially selective chemical sensors with a pattern-recognition system. **Persaud & Dodd (1982)** built an early version from semiconductor transducers and showed that discrimination between many odours does not need highly specific receptors: the *pattern* across broadly tuned sensors carries the information. The textbook **Gardner & Bartlett (1999)** is the standard reference for the principles; Rodríguez et al. cite pp. 74–75 of it for their feature. **Gardner (1991)** studied PCA and cluster analysis of multisensor-array data and reports that a theoretically derived normalisation improves the classification of vapours and beverages (auto-generated summary). **Wilson & Baietto (2009)** review e-nose sensor types and applications across agriculture, food, medicine and industry. **Marco & Gutierrez-Galvez (2012)** review signal and data processing for machine olfaction: classification, regression, clustering, and robustness to drift, background and environmental changes.

Two properties of MOX sensors matter for this project. First, the response is a change in resistance (conductance rises for reducing gases). This is why the Source 1 article and our baseline use conductance G = 1/R. Second, MOX sensors **drift**. **Vergara et al. (2012)** address drift with classifier ensembles (auto-generated summary). Drift is relevant to our finding that S1's pre-exposure baselines differ between quality classes (`data_notes.md`).

## 2. E-noses applied to coffee

### 2.1 The reference article (Source 1)
**Rodríguez, Durán & Reyes (2010), *Sensors* 10(1):36–46.** The A-NOSE (University of Pamplona) uses 8 Figaro/FIS MOX sensors. Its static feature is the max–min variation of conductance per sensor, mean-centred, followed by PCA and a feed-forward MLP (10 hidden neurons, backpropagation with an adaptive learning rate), validated by leave-one-out. Samples came from the Almacafé quality-control laboratory in Cúcuta, Colombia. Reported results: 99.62 % PC1+PC2 variance and 100 % LOO accuracy (16 measurements, 4 groups), and > 99 % and 92.5 % (35 measurements, 9 categories). Green beans could not be classified. A preliminary conference version exists (**Rodríguez et al., 2009, AIP Conf. Proc.**, with M. Pardo as co-author). Crossref lists the journal article's online date as 2009 and the issue as 2010 (vol. 10).

### 2.2 Coffee works cited by the reference article (exact list from PMC3270826)
- **Gardner, Shurmer & Tan (1992)**, *Sens. Actuators B* 6:71–75: application of an e-nose to the discrimination of coffees. This is one of the first coffee e-nose studies. No abstract was retrievable, so details are not summarised here.
- **Singh, Hines & Gardner (1996)**, *Sens. Actuators B* 30: compares a fuzzy neural network with a standard back-propagation network for classifying coffee and tainted-water samples from a semiconducting-oxide e-nose (auto-generated summary).
- **Tan, Loubet, Labreche & Amine (1997)**, ISIE'97: validation of a commercial FOX4000 e-nose for coffee QC, including instrument stability (one month uncalibrated, six months with calibration materials) and transfer of databanks from R&D to production instruments.
- **Pardo, Niederjaufner, Benussi, Comini, Faglia, Sberveglieri, Holmberg & Lundström (2000)**, *Sens. Actuators B* 69: shows that preprocessing chemical-sensor data improves e-nose classification of espresso brands, reaching 100 % correct classification with two sensors (auto-generated summary).
- **Pardo & Sberveglieri (2002)**, *IEEE Trans. Instrum. Meas.* 51(6):1334–1339: the Pico-1 thin-film semiconductor e-nose on two groups of seven coffees. PCA and MLPs give classification rates above 90 %. The paper also predicts trained-panel descriptors and a Hedonic Index, with errors comparable to the panel's own uncertainty.
- **Falasconi et al. (2003)**, IEEE Sensors: the EOS835 e-nose (thin-film MOX) with a new feature-extraction algorithm (phase-space integral), applied to Italian espresso quality to choose the packaging time.
- **Falasconi, Pardo, Sberveglieri, Riccò & Bresciani (2005)**, *Sens. Actuators B* 110:73–80: EOS835 data analysis of roasted-coffee ripening; the indium–tin-oxide sensors performed best (auto-generated summary).
- **Wang, Ye & Duanmu (2009)**, *Sens. Actuators B* 140:143–148: relevance vector machines for e-nose data, with accuracy similar to other kernel methods using far fewer kernel functions (auto-generated summary). This is a general classification-method paper, not a coffee paper.
- **Huang et al. (2007)**, *Anal. Chim. Acta* 588:216–223: GC-MS chromatographic fingerprint of coffee flavour with chemometrics. This is the GC-MS reference point that motivates cheaper e-noses (auto-generated summary).

### 2.3 Later coffee e-nose studies
- **Romani et al. (2012)**, *J. Food Sci.*: a 10-MOX-sensor e-nose with PCA (600 values per sensor) feeding MLP and GRNN networks to predict roasting degree and physicochemical parameters.
- **Brudzewski, Osowski & Dwulit (2012)**, *IEEE Trans. Instrum. Meas.*: a "differential" e-nose for detecting coffee forgery (blends of a mediocre and a high-quality brand). It performs much better than a classical e-nose.
- **Fujioka et al. (2015)**, *Sensors*: an e-nose trained on the wine aroma kit "Le Nez du Vin", used to describe coffee aroma by similarity to reference aromas.
- **Bona & dos Santos Ferreira da Silva (2016)**: book chapter "Coffee and the Electronic Nose" in *Electronic Noses and Tongues in Food Science* (Academic Press). No abstract was available.
- **Dong et al. (2017)**, *Food Chem.* 229:743–751: e-nose and e-tongue with chemometrics on 7 Chinese robusta cultivars at several roasting degrees. A PLSR model on the **fused** data set was best (auto-generated summary). This is directly relevant as a coffee data-fusion example.
- **Giungato et al. (2017)**, *Food Anal. Methods*: industrial roasting degree of coffee beans with an e-nose. No abstract was available.
- **Flambeau, Lee & Yoon (2017)**, *Food Sci. Biotechnol.*: geographical-origin discrimination of washed specialty Bourbon coffee. Aroma predicted origin more reliably than taste profile (auto-generated summary).
- **Gonzalez Viejo, Tongson & Fuentes (2021)**, *Sensors* 21(6):2016: a low-cost e-nose with ANNs on 9 commercial coffees (triplicates). It reports 98 % accuracy for low/medium/high intensity and predicts the abundance of 45 aromas.

**Gap.** Coffee e-nose studies use small sample sets, often with validation that is not independent of the measurement session, and almost none compare fusion levels on the same data. Public raw e-nose coffee data is rare; Source 1 is one of very few releases.

## 3. Machine olfaction with learning methods and larger datasets
- **Feng, Dai, Li, Pernigo, Wen & Liang (2026)**, *SmellNet*, ICLR 2026 (arXiv 2506.00239v5): about 828K time steps and 68 h of data from low-cost MOX sensors. SmellNet-Base has 50 substances × 6 days, 6 channels at 1 Hz. The pipeline is temporal differencing, sliding windows and standardisation, followed by MLP/CNN/LSTM/Transformer ("ScentFormer"). Optional GC-MS supervision uses a **symmetric contrastive loss** between sensor windows and an **ingredient-level** FooDB-derived GC-MS embedding during training only. Reported Acc@1 (Table 2): Transformer w=100, p=25 rises from 56.1 % (sensor-only) to 63.3 % (cross-modal). GC-MS helps the weaker models but has mixed effects on temporal models (CNN w=50 p=25: −1.2; LSTM w=100 p=25: −1.0). Leave-one-day-out for ScentFormer (p=25): 49.4 ± 8.0 % (w=50) and 53.0 ± 6.0 % (w=100). Its Table 1 lists the Rodríguez et al. coffee dataset as prior sensor-based olfaction data.
- **Lee et al. (2023)**, *Science* 381:999–1006: a "principal odor map" learned with graph neural networks from molecular structure. It matched a trained panel better than the median panellist on 400 out-of-sample odorants. This is the structure→percept side of machine olfaction, complementary to the sensor side.
- **Radford et al. (2021)**, CLIP, ICML (PMLR 139): the symmetric contrastive image–text objective that SmellNet adapts to sensor–GC-MS alignment.

## 4. Data fusion in food and beverage quality assessment
- **Hall & Llinas (1997)**, *Proc. IEEE* 85(1):6–23: a tutorial on multisensor data fusion covering process models and the distinction between data-, feature- and decision-level fusion.
- **Borràs, Ferré, Boqué, Mestres, Aceña & Busto (2015)**, *Anal. Chim. Acta* 891:1–14: a review of data fusion methodologies for food and beverage authentication and quality assessment. It is the standard reference for the low/mid/high-level terminology used in this project (auto-generated summary; terminology checked against our usage).
- **Doeswijk, Smilde, Hageman, Westerhuis & van Eeuwijk (2011)**, *Anal. Chim. Acta* 705:41–47: studies when high-level fusion improves prediction. Combining a non-predictive classifier with a discriminative one can help if within-group correlations are strong (auto-generated summary). This matters for our A3 and C3 results, where weak per-sensor or per-group models are combined.
- **Wolpert (1992)**, *Neural Networks* 5(2):241–259: stacked generalisation, the basis of our stacking combiner.
- **Zhi, Zhao & Zhang (2017)**, *Sensors* 17(5):1007: multi-level fusion of e-nose and e-tongue for tea quality (4 grades). Feature-level fusion of time- and frequency-domain features is combined with decision-level D-S evidence fusion and beats single-level fusion. This is the closest methodological template to our Experiment A (time-domain plus FFT features, then decision fusion).
- **Dong et al. (2017)**, see Sec. 2.3: e-nose + e-tongue fusion for coffee.

## 5. Coffee cupping and the CQI/SCA scoring system (for interpreting S3)
- **Revi (2019)**, chapter 13 of *Coffee* (Royal Society of Chemistry): an introduction to the specialty-industry cupping methodology for arabica, from sample roasting to numeric scoring, and to cupper competencies.
- **Di Donfrancesco, Gutierrez Guzman & Chambers (2014)**, *J. Sensory Stud.*: 13 Colombian (Huila) coffees evaluated by a trained descriptive panel and by four Q-certified cuppers using the SCAA cupping protocol. The two methods overlapped little, and cuppers' tasting notes agreed poorly. Cupping scores and descriptive analysis are not interchangeable. This supports the Source 1 article's premise that cup tests are partly subjective.
- **Worku, Duchateau & Boeckx (2016)**, *J. Sensory Stud.*: reproducibility of cupping scores at Ethiopian cupping centres. Variability is higher between than within cuppers, but the mean absolute difference in total cup quality is about 3 points on the 100-point scale. This is a useful yardstick for S3: our tertile thresholds (81.67 / 83.17) are only 1.5 points apart, **smaller than the reported between-centre noise**, which bounds how well any model can separate AQ from its neighbours.
- **CQI database (jldbc/coffee-quality-database, GitHub, MIT licence)**: 1,339 reviews by CQI-trained reviewers scraped in January 2018 (our count; see `data_notes.md`). We verified that `Total.Cup.Points` is the sum of the 10 attribute scores (95 % exact).

## 6. Validation practice on small datasets
- **Varma & Simon (2006)**, *BMC Bioinformatics* 7:91: using the CV error of a classifier whose parameters were tuned by that same CV is optimistically biased (shown with LOOCV for SVMs on null data). **Nested CV** removes the bias. This is why we tune only in inner loops.
- **Cawley & Talbot (2010)**, *JMLR* 11:2079–2107: over-fitting in model selection and the resulting selection bias in performance evaluation.
- **Kaufman, Rosset, Perlich & Stitelman (2012)**, *ACM TKDD* 6(4): formal treatment of **leakage**, i.e. information about the target that would not legitimately be available. The class-level GC-MS vectors in SmellNet are a textbook case if attached to test samples by label (our Experiment D, DX).
- **Dietterich (1998)**, *Neural Comput.* 10(7):1895–1923: McNemar's test has low type-I error, while paired t-tests on repeated random splits have high type-I error. We use McNemar on LOO predictions and the Wilcoxon signed-rank test on per-repeat scores, and treat p-values cautiously.
- **Demšar (2006)**, *JMLR* 7:1–30: statistical comparison of classifiers, recommending the Wilcoxon signed-rank test for comparing two classifiers.

---

## Summary table

| Paper | Year | Data | Sensors | Method | Validation | Result (as reported) | Limitations |
|---|---|---|---|---|---|---|---|
| Rodríguez, Durán & Reyes, *Sensors* | 2010 | Colombian coffee, Almacafé lab; 16 meas. (4 groups) and 35 meas. (9 cat.) | 8 MOX (Figaro TGS, FIS SP) | ΔG (max–min conductance), mean-centring, PCA, MLP (10 hidden) | LOO | PC1+PC2 99.62 %; LOO 100 % (4 groups), 92.5 % (9 cat.) | Very small n; LOO only; exact feature formula not given; green beans not classifiable |
| Gardner, Shurmer & Tan, *Sens. Act. B* | 1992 | Coffees | Semiconducting oxide array | Pattern recognition | n/a (abstract unavailable) | Coffee discrimination (title) | Not summarised (no abstract) |
| Singh, Hines & Gardner, *Sens. Act. B* | 1996 | Coffee and tainted water | Semiconducting oxide | Fuzzy NN vs back-propagation NN | n/a* | Comparison of the two networks* | *Auto-generated summary only |
| Tan et al., ISIE'97 | 1997 | 3 coffees + pure products | FOX4000 commercial e-nose | QC validation, calibration, databank transfer | Over 1–6 months, 3 instruments | Stable discrimination 1 month uncalibrated, 6 months calibrated | Instrument-validation focus, few products |
| Pardo et al., *Sens. Act. B* | 2000 | Espresso brands | Thin-film MOX | Data preprocessing + classifier | n/a* | 100 % with 2 sensors* | *Auto-generated summary only |
| Pardo & Sberveglieri, *IEEE TIM* | 2002 | 2 groups × 7 coffees + trained panels | Pico-1 thin-film semiconductor | PCA, MLP; regression of panel descriptors | n/a in abstract | > 90 % classification; Hedonic Index error ≈ panel uncertainty | Small sample sets |
| Falasconi et al., IEEE Sensors | 2003 | Italian espresso | EOS835 thin-film MOX | Phase-space-integral features | n/a | Packaging-time determination | Conference paper |
| Falasconi et al., *Sens. Act. B* | 2005 | Roasted-coffee ripening | EOS835 (3 sensor types) | Data analysis | n/a* | ITO sensors best* | *Auto-generated summary only |
| Wang, Ye & Duanmu, *Sens. Act. B* | 2009 | E-nose data sets | n/a | Relevance vector machine | n/a* | Similar accuracy, fewer kernels* | Not coffee-specific |
| Romani et al., *J. Food Sci.* | 2012 | Coffee at 6 roasting times | 10 MOX | PCA (600 values/sensor) → MLP, GRNN | n/a in abstract | Good prediction of roasting degree; GRNN best | Single roasting process |
| Brudzewski, Osowski & Dwulit, *IEEE TIM* | 2012 | Coffee blends (forgery) | Semiconductor array (differential) | Differential e-nose + signal processing | n/a | Better than classical e-nose | Specific adulteration task |
| Fujioka et al., *Sensors* | 2015 | Drip/canned coffee, wine aroma kit | E-nose | Similarity to learned reference aromas | n/a | Descriptive aroma mapping | Qualitative |
| Dong et al., *Food Chem.* | 2017 | 7 robusta cultivars × roasting degrees | E-nose + e-tongue | Chemometrics, PLSR, fusion | n/a* | Fused-data PLSR best* | *Auto-generated summary only |
| Zhi, Zhao & Zhang, *Sensors* | 2017 | Tea, 4 grades | E-nose + e-tongue | Feature-level (time + frequency) + decision-level (D-S) fusion | n/a in abstract | Multi-level fusion beats single-level | Tea, not coffee |
| Flambeau, Lee & Yoon, *Food Sci. Biotech.* | 2017 | Washed Bourbon coffees, several origins | E-nose (+ taste) | Origin prediction | n/a* | Aroma predicts origin better than taste* | *Auto-generated summary only |
| Gonzalez Viejo, Tongson & Fuentes, *Sensors* | 2021 | 9 commercial coffees × 3 | Low-cost e-nose | ANN classification + regression | n/a in abstract | 98 % intensity class; 45-aroma regression | Triplicates of 9 samples; small n |
| Feng et al., SmellNet, ICLR | 2026 | 50 foods × 6 days (Base) + 43 mixtures | 6 MOX channels (Grove V2, MQ-3, MQ-5) | Differencing, windows, MLP/CNN/LSTM/Transformer; contrastive GC-MS prior | Day 6 test; leave-one-day-out | 56.1 → 63.3 % (Transformer w=100, p=25); LODO 53.0 ± 6.0 % | GC-MS is class-level (FooDB), not measured; checkpoint protocol ambiguous (Sec. 5.2 vs App. G.2) |
| Lee et al., *Science* | 2023 | ~400 prospective odorants + panels | None (molecular structure) | Graph neural network "principal odor map" | Prospective set | Beat median panellist | Structure-based, not sensor-based |
| Borràs et al., *Anal. Chim. Acta* | 2015 | Review | Various | Low/mid/high-level fusion | n/a | Taxonomy of food data fusion | Review |
| Doeswijk et al., *Anal. Chim. Acta* | 2011 | Simulation/real | n/a | High-level fusion analysis | n/a | Conditions for high-level fusion gains* | *Auto-generated summary only |
| Di Donfrancesco et al., *J. Sens. Stud.* | 2014 | 13 Colombian coffees | Human panels | SCAA cupping vs descriptive analysis | n/a | Little overlap; low agreement among cuppers | 4 cuppers |
| Worku et al., *J. Sens. Stud.* | 2016 | 30 + 70 Ethiopian samples | Human cuppers | Reproducibility of cupping scores | Within/between cuppers and centres | ~3-point mean absolute difference in total score | Ethiopia only |
| Varma & Simon, *BMC Bioinf.* | 2006 | Simulated null/non-null | n/a | CV for model selection; nested CV | LOOCV, 10-fold | Non-nested CV error is optimistically biased | Simulation |
| Kaufman et al., *ACM TKDD* | 2012 | Competitions, industry cases | n/a | Formal definition of leakage | n/a | Taxonomy of leakage | Conceptual |

**Key take-aways for this project.**
1. The Source 1 pipeline (static ΔG feature, PCA, small MLP, LOO) is typical of 2000s coffee e-nose work. LOO on 16–35 samples is optimistic, especially when samples of a class share a session (Varma & Simon 2006; our AB7).
2. Multi-level fusion helps in related tasks (Zhi et al. 2017; Dong et al. 2017), but high-level fusion of weak components can fail as well as succeed (Doeswijk et al. 2011). So the fusion-level comparison is an empirical question worth answering under one protocol.
3. In SmellNet, GC-MS is a class-level prior. Its legitimate use is training-time alignment (as in CLIP), and attaching it to test samples by label is leakage (Kaufman et al. 2012).
4. Cupping scores are noisy at the 1–3-point level (Worku et al. 2016) and partly subjective (Di Donfrancesco et al. 2014), which limits how accurately any model can predict narrow quality bins from S3.

## References (verified; DOI or URL)
1. Rodríguez, J.; Durán, C.; Reyes, A. Electronic Nose for Quality Control of Colombian Coffee through the Detection of Defects in "Cup Tests". *Sensors* 2010, 10(1), 36–46. doi:10.3390/s100100036
2. Rodríguez, J.; Durán, C.; Reyes, A.; Pardo, M. (conference version, same title). *AIP Conf. Proc.* 2009. doi:10.1063/1.3156580
3. Rodríguez Gamboa, J. C. Dataset: Electronic Nose for Quality Control of Colombian Coffee through the Detection of Defects in "Cup Tests". Mendeley Data, V1, 2020. doi:10.17632/7spd6fpvyk.1
4. Feng, D.; Dai, W.; Li, C.; Pernigo, A.; Wen, Y.; Liang, P. P. SmellNet: A Large-scale Dataset for Real-world Smell Recognition. ICLR 2026. arXiv:2506.00239v5. Code: https://github.com/MIT-MI/SmellNet; data: https://huggingface.co/datasets/DeweiFeng/SmellNet
5. LeDoux, J. (GitHub user jldbc). coffee-quality-database: Coffee Quality Institute review data, GitHub, 2018. https://github.com/jldbc/coffee-quality-database (MIT licence). *Name from the public GitHub profile of the repository owner.*
6. Persaud, K.; Dodd, G. Analysis of discrimination mechanisms in the mammalian olfactory system using a model nose. *Nature* 1982, 299, 352–355. doi:10.1038/299352a0
7. Gardner, J. W.; Bartlett, P. N. *Electronic Noses: Principles and Applications.* Oxford University Press, 1999.
8. Gardner, J. W. Detection of vapours and odours from a multisensor array using pattern recognition. Part 1. Principal component and cluster analysis. *Sens. Actuators B* 1991, 4, 109–115. doi:10.1016/0925-4005(91)80185-M
9. Wilson, A. D.; Baietto, M. Applications and Advances in Electronic-Nose Technologies. *Sensors* 2009, 9(7), 5099–5148. doi:10.3390/s90705099
10. Marco, S.; Gutierrez-Galvez, A. Signal and Data Processing for Machine Olfaction and Chemical Sensing: A Review. *IEEE Sens. J.* 2012, 12(11), 3189–3214. doi:10.1109/JSEN.2012.2192920
11. Vergara, A.; Vembu, S.; Ayhan, T.; Ryan, M. A.; Homer, M. L.; Huerta, R. Chemical gas sensor drift compensation using classifier ensembles. *Sens. Actuators B* 2012, 166–167, 320–329. doi:10.1016/j.snb.2012.01.074
12. Gardner, J. W.; Shurmer, H. V.; Tan, T. T. Application of an electronic nose to the discrimination of coffees. *Sens. Actuators B* 1992, 6, 71–75. doi:10.1016/0925-4005(92)80033-T
13. Singh, S.; Hines, E. L.; Gardner, J. W. Fuzzy neural computing of coffee and tainted-water data from an electronic nose. *Sens. Actuators B* 1996, 30, 185–190. doi:10.1016/0925-4005(96)80047-5
14. Tan, T. T.; Loubet, F.; Labreche, S.; Amine, H. Quality control of coffee using the FOX4000 electronic nose. *Proc. IEEE ISIE'97*, 1997. doi:10.1109/ISIE.1997.651750
15. Pardo, M.; Niederjaufner, G.; Benussi, G.; Comini, E.; Faglia, G.; Sberveglieri, G.; Holmberg, M.; Lundström, I. Data preprocessing enhances the classification of different brands of Espresso coffee with an electronic nose. *Sens. Actuators B* 2000, 69, 397–403. doi:10.1016/S0925-4005(00)00499-8
16. Pardo, M.; Sberveglieri, G. Coffee analysis with an electronic nose. *IEEE Trans. Instrum. Meas.* 2002, 51(6), 1334–1339. doi:10.1109/TIM.2002.808038
17. Falasconi, M.; Pardo, M.; Riccò, I.; Nardini, F.; Della Torre, M.; Sberveglieri, G.; Bresciani, A. The novel EOS835 electronic nose sniffs out the Italian espresso coffee quality. *Proc. IEEE Sensors 2003*, vol. 1, pp. 26–29. doi:10.1109/ICSENS.2003.1278889
18. Falasconi, M.; Pardo, M.; Sberveglieri, G.; Riccò, I.; Bresciani, A. The novel EOS835 electronic nose and data analysis for evaluating coffee ripening. *Sens. Actuators B* 2005, 110, 73–80. doi:10.1016/j.snb.2005.01.019
19. Wang, X.; Ye, M.; Duanmu, C. J. Classification of data from electronic nose using relevance vector machines. *Sens. Actuators B* 2009, 140, 143–148. doi:10.1016/j.snb.2009.04.030
20. Huang, L.-F.; Wu, M.-J.; Zhong, K.-J.; Sun, X.-J.; Liang, Y.-Z.; Dai, Y.-H.; Huang, K.-L.; Guo, F.-Q. Fingerprint developing of coffee flavor by gas chromatography–mass spectrometry and combined chemometrics methods. *Anal. Chim. Acta* 2007, 588, 216–223. doi:10.1016/j.aca.2007.02.013
21. Romani, S.; Cevoli, C.; Fabbri, A.; Alessandrini, L.; Dalla Rosa, M. Evaluation of Coffee Roasting Degree by Using Electronic Nose and Artificial Neural Network for Off-line Quality Control. *J. Food Sci.* 2012, 77(9). doi:10.1111/j.1750-3841.2012.02851.x
22. Brudzewski, K.; Osowski, S.; Dwulit, A. Recognition of Coffee Using Differential Electronic Nose. *IEEE Trans. Instrum. Meas.* 2012, 61(6), 1803–1810. doi:10.1109/TIM.2012.2184011
23. Fujioka, K.; Tomizawa, Y.; Shimizu, N.; Ikeda, K.; Manome, Y. Improving the Performance of an Electronic Nose by Wine Aroma Training to Distinguish between Drip Coffee and Canned Coffee. *Sensors* 2015, 15(1), 1354–1364. doi:10.3390/s150101354
24. Bona, E.; dos Santos Ferreira da Silva, R. S. Coffee and the Electronic Nose. In *Electronic Noses and Tongues in Food Science*; Academic Press, 2016, pp. 31–38. doi:10.1016/B978-0-12-800243-8.00004-4
25. Dong, W.; Zhao, J.; Hu, R.; Dong, Y.; Tan, L. Differentiation of Chinese robusta coffees according to species, using a combined electronic nose and tongue, with the aid of chemometrics. *Food Chem.* 2017, 229, 743–751. doi:10.1016/j.foodchem.2017.02.149
26. Giungato, P.; Laiola, E.; Nicolardi, V. Evaluation of Industrial Roasting Degree of Coffee Beans by Using an Electronic Nose and a Stepwise Backward Selection of Predictors. *Food Anal. Methods* 2017, 10(10), 3424–3433. doi:10.1007/s12161-017-0909-z
27. Flambeau, K. J.; Lee, W.-J.; Yoon, J. Discrimination and geographical origin prediction of washed specialty Bourbon coffee from different coffee growing areas in Rwanda by using electronic nose and electronic tongue. *Food Sci. Biotechnol.* 2017, 26(5), 1245–1254. doi:10.1007/s10068-017-0168-1
28. Gonzalez Viejo, C.; Tongson, E.; Fuentes, S. Integrating a Low-Cost Electronic Nose and Machine Learning Modelling to Assess Coffee Aroma Profile and Intensity. *Sensors* 2021, 21(6), 2016. doi:10.3390/s21062016
29. Lee, B. K.; Mayhew, E. J.; Sanchez-Lengeling, B.; Wei, J. N.; et al. A principal odor map unifies diverse tasks in olfactory perception. *Science* 2023, 381, 999–1006. doi:10.1126/science.ade4401
30. Radford, A.; Kim, J. W.; Hallacy, C.; et al. Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*, PMLR 139. https://proceedings.mlr.press/v139/radford21a.html
31. Hall, D. L.; Llinas, J. An introduction to multisensor data fusion. *Proc. IEEE* 1997, 85(1), 6–23. doi:10.1109/5.554205
32. Borràs, E.; Ferré, J.; Boqué, R.; Mestres, M.; Aceña, L.; Busto, O. Data fusion methodologies for food and beverage authentication and quality assessment – A review. *Anal. Chim. Acta* 2015, 891, 1–14. doi:10.1016/j.aca.2015.04.042
33. Doeswijk, T. G.; Smilde, A. K.; Hageman, J. A.; Westerhuis, J. A.; van Eeuwijk, F. A. On the increase of predictive performance with high-level data fusion. *Anal. Chim. Acta* 2011, 705, 41–47. doi:10.1016/j.aca.2011.03.025
34. Wolpert, D. H. Stacked generalization. *Neural Netw.* 1992, 5(2), 241–259. doi:10.1016/S0893-6080(05)80023-1
35. Zhi, R.; Zhao, L.; Zhang, D. A Framework for the Multi-Level Fusion of Electronic Nose and Electronic Tongue for Tea Quality Assessment. *Sensors* 2017, 17(5), 1007. doi:10.3390/s17051007
36. Revi, I. Chapter 13. Coffee Cupping: Evaluation of Green Coffee Quality. In *Coffee*; Royal Society of Chemistry, 2019, pp. 335–360. doi:10.1039/9781782622437-00335
37. Di Donfrancesco, B.; Gutierrez Guzman, N.; Chambers, E. Comparison of Results from Cupping and Descriptive Sensory Analysis of Colombian Brewed Coffee. *J. Sensory Stud.* 2014, 29(4), 301–311. doi:10.1111/joss.12104
38. Worku, M.; Duchateau, L.; Boeckx, P. Reproducibility of coffee quality cupping scores delivered by cupping centers in Ethiopia. *J. Sensory Stud.* 2016, 31(5), 423–429. doi:10.1111/joss.12226
39. Varma, S.; Simon, R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006, 7, 91. doi:10.1186/1471-2105-7-91
40. Cawley, G. C.; Talbot, N. L. C. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *JMLR* 2010, 11, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html
41. Kaufman, S.; Rosset, S.; Perlich, C.; Stitelman, O. Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD* 2012, 6(4). doi:10.1145/2382577.2382579
42. Dietterich, T. G. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Comput.* 1998, 10(7), 1895–1923. doi:10.1162/089976698300017197
43. Demšar, J. Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR* 2006, 7, 1–30. https://jmlr.org/papers/v7/demsar06a.html

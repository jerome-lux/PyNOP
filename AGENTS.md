# PyNOP — Notes de session

## Objectif global

Analyser, nettoyer, vérifier les implémentations et trouver les bugs dans la codebase
des **opérateurs neuronaux** (PyNOP = Py-Neural Operators), afin de **repartir sur de bonnes
bases** pour les sessions suivantes. Le travail se fait **modèle par modèle**, de façon
itérative, sur la branche `dev`.

> Principe : l'architecture des modèles est **conservée** (utilisée comme référence de
> comparaison). On ne cherche qu'à corriger les bugs, nettoyer, et documenter.

> **Journal détaillé** : l'analyse complète par modèle/bloc (contexte, symptômes, mesures,
> arbitrages, vérifications) est dans `docs/session-notes.md`. Les références `§N` ci-dessous
> pointent vers ce fichier ; les numéros sont stables.

---

## État des modèles (à ce jour)

| Modèle | Fichier | Statut |
|---|---|---|
| **DeepONet** | `pynop/models/deeponet.py` | ✅ Corrigé (IndexError/AttributeError/NotImplementedError via `nn.Sequential`, logique `scale` flexible, docstrings raw strings) |
| **SwinNO / MultiscaleSWINNO** | `pynop/models/SWIN.py` | ✅ Nettoyé + validé (§3) |
| **UNO** (U-Net Neural Operator) | `pynop/models/UNO.py` | ✅ U-Net multi-échelle avec skips first-class, fusion `"add"`/`"concat"`, bottleneck transformer global, invariance de résolution (§13) |
| **LNO (`LatentNO`)** | `pynop/models/LNO.py` | ⚠️ Architecture correcte, mais **ne s'entraîne pas bien** — cause identifiée : *setup d'entraînement*, pas l'init (§4) |
| **LatentGalerkin** | `pynop/models/LNO.py` | ✅ Analysé + bugs corrigés (PE, RoPE, activation) + **branche locale** ajoutée (§5, §7) |
| **ITLNO** | `pynop/models/LNO.py` | ✅ Analysé + bugs corrigés (§6) + **branche locale** ajoutée (§7) |
| **LatentTransolver** | `pynop/models/LNO.py` | ✅ Vérifié (Transolver3 correct) + nettoyage + **branche locale** ajoutée (§7) |
| **FNO** | `pynop/models/FNO.py` | ✅ Analysé + bugs corrigés (crash forward, Tucker, invariance rés.) ; bloc aligné sur la référence + **branche locale** `FiniteDifferenceLayer` (§14) |
| **Transolver** | `pynop/models/transolver.py` | ✅ Analysé + bugs corrigés (dispatch mode, `ln1` morte, dropout non câblé) ; **`TransolverBlockv3` réécrit = port fidèle de Transolver-3** (arXiv:2602.04940 : poids par tête, température clampée, concat des têtes), sans tiling ; 3 modes `"linear"`/`"t3"`/`"physics"` (§17) ; **re-vérifié contre l'officiel + OOM corrigé** : `out_dropout` mort et init ~43× trop petite fixés, option `use_checkpoint` (gradient checkpointing des blocs, activée en config T3) → 75 GB → ~15 GB/pas (§22) |
| **ITNet** | `pynop/models/ITNet.py` | ⏳ À analyser |
| **UNet** (classique) | `pynop/models/unet.py` | ✅ Réécrit : U-Net classique (non-NO) validé — 4 blocs acceptés, PatchEmbed option, bottleneck transformer (PE **ou** RoPE) (§8) |
| **Resnet** (classique) | `pynop/models/resnet.py` | ✅ Unifiée : `block_type="basic"`/`"bottleneck"` + `bottleneck_ratio` + `se` ; `Resnetb` supprimé ; `resnet18`/`34`/`50`/`resnext50` fonctionnent (§11) |
| **Blocs communs** | `pynop/core/blocks.py`, `pynop/core/ops.py` | Blocs Swin déplacés ici ; **`TransformerBlock` unifié** avec option RoPE (`AdaptiveRoPE2D`) (§9, §10) |
| **Blocs IT** | `pynop/core/blocks.py` | ✅ `SepLITBlock`/`SepNLITBlock`/`LITBlock` vérifiés — 3 bugs corrigés (§12) ; **`IAETBlock`** (style IAE-Net) vérifié contre l'implémentation officielle — 4 bugs corrigés (§16) |
| **Dataloaders unifiés** | `pynop/dataloader/dataloader.py` | ✅ Réécrit : classe de base `TemporalWindowDataset` (fenêtres, normalisation, RAM/disk, contrat batch `(fields, t0[, cond])`) + 5 sous-classes par format — `PDEBenchDataSet`, `CustomDataset` (data+porosity, split natif par attribut), `Ayz2Dataset` (ex-NSH5 : fichiers séparés train/val, coefficients scalaires **et** champ spatial `a` → cond toujours renvoyé si présent), `TheWellDataSet` (+indexes/+shift/scale), `npyPDEDataset` ; protocole de split unifié `resolve_split_ids` (IDs explicites → splits natifs avec `n_train_samples`/`n_val_samples` → protocole PDEBench bit-identique) + `build_dataloaders` partagé train/eval (§23) |
| **Training (Lightning)** | `pynop/training/lightning_model.py` | ✅ Réécrit : `predict_derivative` (sortie dérivée **ou** absolue), derivative-loss configurable, base unique `MultiStepLightningBase`, compat. tous les modèles du dépôt, `DeepONetModel` dédié ; wrappers AE/`CausalNO` supprimés (§15) |
| **Driver CLI d'entraînement** | `pynop/training/train.py`, `configs/` | ✅ Réécrit : driver config-driven (`ModelConfig.json` + `TrainingConfig.json`, flag `--pretrain`) reproduisant le protocole 2 phases des notebooks PDEBench ; **tous les types de dataset** via `build_dataloaders` (pdebench/custom/ayz2/thewell/npy) (§18, §23) ; checkpoints `last` (chaque époque) / best (dès max pas AR atteint) ; `--resume`/`--init-from`. **Configs complètes** : un `model_<name>.json` par modèle des notebooks (10 fichiers) + 1 `training_PDEBench.json` partagé, tous validés build+forward+backward (§20) |
| **Eval CLI + métriques** | `pynop/eval/eval.py`, `pynop/training/metrics.py` | ✅ Nouveau : évaluation d'un checkpoint par **rollout autorégressif de la trajectoire complète** (protocole notebooks PDEBench) sur des simulations **non utilisées en entraînement** ; **tous les types de dataset** — split val, ou test = `test_path` explicite (ex. TheWell `data/test/`, disjoint par construction) sinon pool inutilisé ; lecture des trajectoires via la classe du format (+ cond renvoyé aux modèles `cond=`) (§19, §23) ; métriques de champ MSE/RMSE/**nRMSE = relative L2**/MAE/max-err/**R²** + bandes spectrales radiales style PDEBench (unités d'index bin FFT), **courbes d'erreur par pas de temps** et graphiques PNG en anglais ; `eval_results_<split>.json` auto-contenu. **Déplacé dans le package `pynop/eval`** (§24) — shim déprécié à l'ancienne location |
| **Visualisation interactive** | `pynop/eval/visualisation.py`, `notebooks/interactive_error_viz.ipynb` | ✅ Nouveau : figure plotly animée GT/Pred/Abs Error (play + scrub), **échelles de couleur indépendantes data/erreur**, chacune `"fixed"` (défaut = min/max sur tous les pas) ou `"adaptive"`, vmin/vmax explicites ; courbes RMSE/MAE/nRMSE par pas ; notebook explorateur (dropdowns sim/canal/échelles, rollouts mis en cache) — testé SwinNOx4 + PDEBench 2d_reaction_diffusion (§24) |
| **Aim (logging)** | `pynop/training/aim_logger.py`, `pynop/training/import_logs.py` | ✅ Nouveau : `AimCallback` (métriques live → run Aim, mêmes noms/axes que le CSV) + CLI d'**import des logs CSV/TB existants** dans le même experiment pour comparer modèles anciens/nouveaux dans l'UI (`aim up --repo ~/.aim`) ; home par défaut `~/.aim` (le défaut CWD d'aim 3.x polluerait le repo git) ; **configs complètes stockées en params imbriqués** `model_config`/`training_config` (live + import, mêmes clés) (§21) |

---

## Décisions clés & gotchas (à retenir sans relire le journal)

### LNO / famille latente (`pynop/models/LNO.py`)
- **LNO ne s'entraîne pas bien : cause = setup d'entraînement, PAS l'init** (§4). Dans la
  référence (`L-I-M-I-T/LatentNeuralOperator`), `_init_weights` (std=0.0002) est une
  **méthode morte** → la référence utilise Kaiming par défaut ; l'init « petite » est instable.
  Candidats : loss relatif `LpLoss` vs MSE brut, grad clipping absent, scheduler.
- **Défaut hautes fréquences commun à LatentGalerkin/ITLNO/LatentTransolver** (§5–§7) :
  opérateur global et lisse = passe-bas structurel + rang faible `M≪N` + pas de skip → le
  remède est une **branche locale** (`FiniteDifferenceLayer`, `core/ops.py`), PAS plus de
  passes globales. Branche ajoutée aux 3 modèles (paramètre `n_local_conv`, défaut 0).
- PE retenu pour LatentGalerkin : `"adaptive"` (Nyquist adaptatif) ; SIREN testé sans succès (§5).

### Convention famille IT (`pynop/core/blocks.py`) — SepLIT/SepNLIT/LIT/IAET (§12, §16)
- Norme **channels-first** (`LayerNorm2d`), jamais channels-last sur du `(B, C, H, W)`.
- Échelle de la transformée : exposant **apprenable** `(H·W)**sigmoid(IT_scale)` (init −1.1 →
  sigmoid≈0.25), JAMAIS `1/(H·W)` fixe — la base apprise n'est pas unit-énergie, le facteur
  constant rendait la branche morte (~100–400× sous le shortcut).
- Shape des poids de mixage fréquentiel : **(out, in, m1, m2)** pour l'einsum `"bixy,oixy->boxy"`.
- `IAETBlock` : norm spectrale complexe-safe (`ComplexLayerNorm`) après encodeur = `ln1` du
  officiel (sans elle, explosion O(5e3) à res 64) ; opère sur la **partie réelle** de l'entrée.
  Référence officielle : `github.com/IAE-Net/iae_net` (arXiv:2203.05142).

### Blocs Swin (§3, §9)
- Les blocs adaptatifs **fixent le nombre de fenêtres** et déduisent la taille → invariance de
  résolution, mais l'entrée doit être **divisible** par `num_windows` (sinon `ValueError`).
  Coût `O((H·W)²)` — inhérent au design.
- `TransformerBlock` unifié avec option RoPE (`AdaptiveRoPE2D`, fréquences jusqu'au Nyquist de la
  grille, paramètre `base`) ; `RoPETransformerBlock` déprécié (sous-classe).
- **`SwinNO` ≠ `MultiscaleSWINNO` à une seule étape** (vérifié empiriquement, oct. 2026) : même avec
  un seul `grid_resolution`, le modèle multiscale garde son bottleneck encodeur/décodeur
  (`WindowCrossAttention` de pooling + `BlockCrossAttentionUpsampler`) — ce n'est pas la pile de
  fenêtres pleine résolution de `SwinNO`. Config PDEBench : 1.19 M vs 1.37 M params, sorties sans
  corrélation à l'init aléatoire.

### Notebooks d'entraînement PDEBench (oct. 2026)
- Un notebook par modèle dans `/run/media/jlux/SSD2/pyNOP/training_<Model>_PDEBench.ipynb` :
  ITLNO, LatentGalerkin, Transolver_linear, Transolver3, LNO, FNOvanilla, tuckerFNO, MSSwinNO (2 étapes),
  et un pour la famille Swin (`SwinNO` + `MSSwinNO_1stage`, commutable via `name`). Basés sur l'ancien
  `training_multistep_PDEBench.ipynb` adapté à l'API refondue ; `utils.py` (helpers de plot) copié
  dans le même dossier.
- Protocole **2 phases conservé** : `train_mode = "pretraining"` (40 sims, t∈[0,50], stride 2,
  curriculum AR 2→10 sur epochs 0–200) → `"fulltraining"` (400 sims, tout t, stride 4, AR fixe à 10).
- Adaptations d'API : SwinNO `winsize=16` → `num_windows=8` ; MSSwinNO `enc/dec_windows` →
  `enc/dec_num_windows` ; LatentGalerkin sans `mlp_layers`/`mlp_dim` (retirés de la signature) ;
  `shared_params` aplati au niveau supérieur du dict `training_params` (bug latent KeyError dans
  l'ancien notebook : il était niché sous `"pdebench"`).
- Logs : `/run/media/jlux/SSD2/PDETraining/PDEBENCH/<phase>/<model>/10steps_<timestamp>` — le montage
  est passé de `/media/jlux` à `/run/media/jlux`.

### Configs JSON PDEBench (`configs/`, §20)
- **Un `model_<name>.json` par modèle des notebooks** (10 au total : FNOvanilla, tuckerFNO, ITLNO, LNO,
  LatentGalerkin, SwinNO, MSSwinNO_1stage, MSSwinNO, Transolver_linear, Transolver3) + **un seul
  `training_PDEBench.json` partagé** (mêmes params d'entraînement pour tous). Usage :
  `python -m pynop.training.train configs/model_<name>.json configs/training_PDEBench.json [--pretrain]`.
- Conventions : `in_channels = 2 × input_steps(10) = 20` matérialisé ; `"activation": "GELU"` en string
  (résolu par `build_model`) ; tuples → listes ; **`verbose` omis** pour ITLNO/LatentGalerkin
  (`build_model` le désactive alors, comme les notebooks avant fit). Le champ `"name"` = nom d'exécution
  des notebooks → même arborescence de logs.
- **Gotcha `MSSwinNO_1stage`** : `dec_num_windows=[(16,16)]` sur grille 16×16 = fenêtres **1×1**
  (attention par pixel au décodeur). C'est la config d'origine (ancien "OSSwinNO", `dec_windows` était
  déjà un nombre de fenêtres/axe — vérifié git) ; conservé pour fidélité, signalé à l'utilisateur.
- **Configs volatiles** : les params des `model_*.json` sont modifiés régulièrement par l'utilisateur
  (tests de configurations différentes) — au moment du commit, prendre les valeurs telles qu'elles sont,
  sans « corriger » un écart avec les notebooks ni s'étonner d'un drift ; resynchroniser la doc §20
  (table + comptages de params) quand on y touche.

### FNO (§14)
- Ordre du bloc aligné sur neuralop / Li et al. 2021 : `[norm] → spectral_conv → [activation] →
  pointwise(1×1) → + shortcut`, norm/activation optionnelles, 3 convolutions spectrales
  interchangeables (`standard`/`separable`/`tucker`).
- Modes effectifs = `min(configuré, disponible)` appliqués à l'entrée **et** aux poids — jamais de
  mutation d'état du module (invariance de résolution).
- Branche locale passe-haut `FiniteDifferenceLayer` en parallèle du shortcut (`local_conv`, défaut 0).

### Famille Transolver (`pynop/models/transolver.py`, `core/blocks.py`) (§17)
- **Deux papiers distincts, ne pas confondre** : *Transolver-3* (arXiv:2602.04940, repo officiel
  `lbruand-db/distributed-transolver3`) = bloc v3 avec slicing/deslicing « faster » ; *LinearNO*
  (« Transolver is a Linear Transformer », arXiv:2511.06294) = reformulation en attention linéaire
  canonique (softmax asymétriques). Le mécanisme de slicing `w` + normalisation diagonale vient de la
  famille Transolver (arXiv:2402.02366), pas de LinearNO.
- Modes du wrapper : `"linear"` (`LinearNOBlock`, arXiv:2511.06294), `"t3"`
  (`TransolverBlockv3`, arXiv:2602.04940), `"physics"` (`TransolverBlock`, Gumbel-Softmax,
  arXiv:2402.02366). Mode inconnu → `ValueError` (l'ancien `elif "t3":` était toujours vrai).
- **`TransolverBlockv3` = port fidèle de `PhysicsAttentionV3` + `Transolver3Block`** (sans tiling ni
  state-caching, qui sont des optimisations mémoire/inférence mathématiquement identiques) :
  - **poids de slicing par tête** (`w` en `[B,H,N,M]`, `in_project_slice: C → H·M` init orthogonale),
    température apprenable **par tête** clampée à `[0.1, 5]` (init 0.5) ;
  - attention = une seule couche SDPA sur les M slices par tête (`to_q/k/v: D→D`, partagés entre têtes),
    puis sous-couche MLP séparée `mlp(ln_2(fx)) + fx` — pas de pile de `TransformerBlock` ;
  - deslice : projection par slice **puis** concaténation des têtes (`C = H·D`, asserté).
- `SlicingBlock`/`DeslicingBlock` (forme à jeu de poids unique, partagé avec `LatentTransolver`) :
  projeter avant ou après la somme pondérée est équivalent (lignes de `w/d` somment à 1 → le biais est
  ajouté une fois par slice) ; l'ancien bug était la projection **après** la somme *sans* cette
  normalisation. Docstrings corrigés (citations Transolver-3, pas LinearNO).
- `LinearNOBlock` conforme aux Eq. (8)–(9) d'arXiv:2511.06294 ; `TransolverBlock` en pre-norm
  (`ln1` avant projection), dropout sur la sortie attention projetée.

### Training Lightning (§15)
- `predict_derivative=True` : sortie = dérivée → `preds = u_t + dt·out` ; `False` : sortie absolue.
- DeepONet exige **`DeepONetModel`** (paradigme branch/trunk) — `MultiStepNOModel` le refuse
  explicitement avec un message pointant vers la bonne classe.

### Dataloaders unifiés (`pynop/dataloader/dataloader.py`, §23)
- **Base `TemporalWindowDataset`** : génération de fenêtres (boucle **bit-identique** à l'ancienne
  PDEBench → reproductibilité des runs existants), normalisation `(x+shift)*scale` (scalaire ou
  par-canal), modes RAM/disk, contrat batch `(fields [T,C,H,W], t0)` ou `(+ cond [D,H,W])`.
  Hooks par sous-classe : `_discover_sample_ids`, `_sample_lengths`, `_read_range(sid,t0,t1)`,
  `_load_to_ram`, `_extra_output` ; méthodes publiques `load_trajectory` / `load_conditioning`
  (utilisées par eval.py — **une seule voie de lecture** pour tous les formats).
- **Formats** : `"pdebench"` (`PDEBenchDataSet`), `"custom"` (`CustomDataset` : data+porosity concat
  canaux, split natif via attribut HDF5 `split` par échantillon — fichier réel : 400 train/50 val),
  `"ayz2"` (`Ayz2Dataset`, ex-NSH5 **renommé sans alias**), `"thewell"` (`TheWellDataSet`), `"npy"`
  (legacy RAM-only). Registry `DATASET_REGISTRY` + `get_dataset_class`.
- **ayz2** : un fichier par split en pratique (`NS_1024.h5`=train, `NS_256.h5`=valid) → config
  `path` + `val_path`. IDs = indices entiers dans le groupe. Conditionnement **toujours renvoyé si
  présent** (choix utilisateur) : champ spatial `a` `(N,H,W)` (fichiers NS) et/ou coefficients
  scalaires `(N,)` auto-détectés (Burgers : ax/ay/cx/cy/nu), broadcast en champs constants →
  triplet batch ; modèle doit accepter `cond=` (FNO avec `cond_dim=D`). PDEs 1D (`u` 3D) → embed
  `(T,1,1,X)`.
- **Protocole de split unifié** `resolve_split_ids(ds_cfg, phase_cfg)` : (1) `train_ids`/`val_ids`
  explicites ; (2) splits natifs — fichiers séparés (`path`/`val_path`), groupes HDF5 ou attribut
  `split` — pools tronquées par `n_train_samples`/`n_val_samples` puis `n_samples` de la phase sur le
  pool train (ordre fichier stable → nesting 2 phases préservé) ; (3) sinon protocole PDEBench
  (shuffle seedé + `n_samples` + `split_ratio`) **bit-identique** à l'ancien driver. Retour :
  `train/val {path, group, ids}` + `train_pool_order` (ordre du pool complet — shuffle pour le cas
  protocole) + `source`.
- **TheWell** : splits = fichiers séparés (`data/train|valid|test`) ; le test set reste structuré
  disjoint via `"test_path"` dans la config dataset (utilisé par eval `--split test`).
- **run_info.json** : clés plates `train_ids`/`val_ids` conservées + `dataset.type` et
  `dataset.val_path` ajoutés. IDs de fichiers différents = namespaces distincts (pas de comparaison
  d'overlap entre fichiers — ex. ayz2 train `0..7` vs val `0..255`).

### Driver CLI d'entraînement (`pynop/training/train.py`, §18)
- Usage : `python -m pynop.training.train MODEL_CONFIG.json TRAINING_CONFIG.json [--pretrain]` —
  `--pretrain` = section `"pretraining"`, sinon `"fulltraining"`. Exemples dans `configs/`
  (protocole notebooks à l'identique). Pas de visualisation : entraînement + logs + checkpoints.
- **Checkpoints** : `last.ckpt` **chaque époque dès la première** (poids toujours disponibles) ;
  `best_RMSE`/`best_val_RMSE` (top-k=2) seulement **dès que le rollout atteint le max de pas AR**
  (`epoch >= final_AR`) — pendant la rampe du curriculum les métriques ne sont pas comparables.
- **Gotcha Lightning 2.x** : sous PL ≥ 2, `ModelCheckpoint` sauvegarde par défaut dans
  `on_train_epoch_end` (pas `on_validation_end`) → le gate de `CurriculumCheckpoint` est appliqué
  dans `_should_skip_saving_checkpoint` (choke point commun) et les checkpoints du driver passent
  `save_on_train_epoch_end=False`. Ne pas régresser vers un override de hook unique.
- Scheduler : tokens `"$total_steps"`/`"$steps_per_epoch"` ; OneCycleLR sans `max_lr` → lr de la
  phase. ⚠️ Resume OneCycleLR = **même total de pas** (on ne change pas `max_epochs` après coup).
  - Split train/val via le protocole unifié du dataloader (§23) : pour PDEBench = shuffle seedé +
    troncature `n_samples` (identique aux notebooks → phases cohérentes) ; splits natifs ou IDs
    explicites pour les autres formats.
  - **Loggers** : `"loggers": ["csv", "aim"]` dans la config partagée PDEBench (TensorBoard remplacé par
    Aim, §21). L'entrée `aim` installe un *callback* (pas un logger Lightning) ; section optionnelle
    `"aim": {"experiment", "home"}`. Les runs importés (`import_logs`) et les runs live partagent la même
    convention d'axes x — ne pas la changer sans resynchroniser les deux.
  - **Aim : `experiment` ≠ dossier** (§21) : tout est stocké dans `<home>/.aim` (dossier *caché* —
    invisible au `ls` simple), un seul stockage pour tous les experiments ; `experiment` n'est qu'un
    label de regroupement dans l'UI (`aim up --repo <home>`). Il n'existe pas de `<home>/<experiment>/`.
    `"aim": {"home"}` absent/null → repli sur `~/.aim` (les runs y restent, les repos Aim ne se fusionnent
    pas — re-importer via `import_logs` si besoin).
  - **Aim : configs complètes en params imbriqués** (§21) : live (`build_aim_callback`) et import
    (`_load_params`) stockent les dicts JSON bruts verbatim sous `model_config` / `training_config`, en
    plus des clés plates de confort (`lr`, `seed`, ...) — mêmes clés des deux côtés, runs comparables.
    Aim stocke un dict comme **arbre** : arborescence dépliable dans l'onglet "Parameters" du détail run ;
    recherche par chemin feuille à feuille (`run.model_config.params.modes == 8`). Lecture SDK :
    `run.get("model_config")` ou chemin tuple `run.get(("model_config", "params"))` — la clé pointée
    `"a.b"` ne marche pas. L'ancienne clé partielle `model_params` (import) est supprimée.

### Évaluation (`pynop/eval/eval.py`, `pynop/training/metrics.py`) (§19, §24)
- **Protocole = rollout autorégressif de la trajectoire complète** (cellules `## Multi-step model eval`
  des notebooks PDEBench) : pour chaque simulation, les `input_steps` premiers snapshots en entrée, puis
  prédiction de **tous** les pas suivants jusqu'à la fin (`S = T_avail − input_steps` prédictions ; la
  1ʳᵉ est teacher-forcée, le reste AR). Erreurs poolées sur toutes les trajectoires + courbes par pas de
  temps (l'index k cible `t = min_id + input_steps + k` pour toutes les traj.). La version initiale par
  fenêtres stride a été **remplacée** — ne pas la réintroduire.
- **nRMSE = relative L2** par échantillon/canal (le « normalized RMSE » de PDEBench) : invariante
  d'échelle, et toujours significative sur données à moyenne nulle (dénominateur = écart-type du
  champ). Normalisation par `mean(target)²` **non utilisée** (dégénère sur champs à moyenne nulle).
- **R²** centrée spatialement uniquement, par échantillon/canal → invariante à toute rénormalisation
  affine ; identité exacte sur données à moyenne nulle : `R²_{i,c} = 1 − nRMSE²_{i,c}`.
- **Bandes spectrales** : coquilles radiales en **unités d'index bin FFT** (bornes PDEBench
  iLow=4/iHigh=12), dépendantes de la résolution comme chez PDEBench ; relatives par défaut
  (invariantes d'échelle), mode absolu via `"eval": {"spectral_relative": false}`.
- Jeu d'évaluation : split **val** par défaut — IDs pris dans le `run_info.json` à côté du checkpoint
  s'il existe, sinon re-déduits de la config ; `--split test` = `test_path` explicite si présent
  (ex. TheWell `data/test/`, disjoint par construction), sinon pool inutilisé (au-delà de
  `n_samples`, même shuffle seedé que l'entraînement) avec repli sur val si vide ; chevauchement
  eval/train → `RuntimeError`. Sélection d'IDs via le protocole unifié du dataloader (§23) — tous les
  formats supportés ; trajectoires lues via la classe du format (`load_trajectory`), et le **cond** est
  renvoyé au rollout pour les modèles `cond=` (sinon métriques silencieusement fausses).
- **Mémoire constante** : chaque prédiction est consommée via le callback `_rollout(on_step=...)` +
  `FieldMetricAccumulator.update_per_step(p, t, step)` (index explicite — appeler `update([p],[t])` par
  pas écraserait tout au pas 0). Batchs groupés par longueur égale (`_group_by_length`) ;
  `_rollout.num_predictions` est **par trajectoire** (× B pour le total). L'optimizer du wrapper
  Lightning est optionnel (inférence).
 - **Sortie** : table stdout + `eval_results_<split>.json` (courbes complètes par pas avec `time`) +
   graphiques PNG en anglais (`eval_error_evolution_*.png` : RMSE/nRMSE/MAE log + R² ;
   `eval_spectral_evolution_*.png` : bandes radiales) via matplotlib Agg, désactivables par `--no-plots`.
 - **Package `pynop/eval`** (§24) : le driver a été déplacé dans `pynop/eval/eval.py` (usage
   `python -m pynop.eval.eval ...`, entry point `pynop-eval`) ; un shim déprécié re-exporte l'API à
   l'ancienne location (`pynop/training/eval.py`). Le même package contient `visualisation.py` :
   `plot_interactive(pred, gt, channel_idx, data_scale, error_scale, *_vmin/vmax, step_labels)` —
   figure animée 3 panneaux (GT/Pred sur l'échelle **data**, Abs Error sur l'échelle **error**),
   chaque échelle `"fixed"` (défaut : min/max de la quantité affichée sur tous les pas) ou
   `"adaptive"` (auto par frame), bornes explicites en override ; `compute_step_errors` renvoie
   RMSE/MAE/nRMSE (relative L2, même définition que §19) + champ d'erreur. Plotly = dépendance
   **optionnelle** (`pip install pynop[viz]`), importée paresseusement. ⚠️ Pitfalls plotly 7.x :
   `Figure.add_frame(s)` supprimé (assignation `fig.frames = [...]`), frames non auto-nommées
   (`name=str(i)` explicite), bouton play via `method="animate"` (pas `"extend"`) ; et les bounds
   zmin/zmax doivent être répétés **sur chaque frame** (un frame affiché remplace la trace de base).

---

## Prochaines étapes (session suivante)
1. [ ] **ITLNO — normalisation principielle** (optionnel) : orthonormaliser la base
      (`Newton_Schulz`/QR dans `core/utils.py`) → projection orthogonale propre au lieu de
      l'échelle apprenable ad hoc (§6). À discuter.
2. [ ] **LNO** : obtenir le script/config d'entraînement exact → identifier et corriger le point
      de stagnation (candidats : loss relatif, grad clipping, LR+warmup, scheduler) (§4).
3. [ ] Vérifier les classes restantes de `pynop/core/blocks.py` et `pynop/core/ops.py`.

> Historique des étapes terminées (DeepONet, SwinNO, UNet, UNO, Resnet unifiée, blocs IT, FNO,
> lightning_model, IAETBlock, Transolver, driver CLI train.py, …) : voir `docs/session-notes.md`.

## Conventions
- Environnement : conda `pytorch` (Python 3.12, Linux).
- Brancher sur `dev` ; committer chaque étape validée.
- Toujours vérifier : import du package, forward pass, circulation des gradients.
- Écrire docstrings + type hints ; tester dans `tests/`.
- **Demander confirmation** avant tout commit / opération git destructive.

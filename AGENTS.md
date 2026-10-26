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
| **Transolver** | `pynop/models/transolver.py` | ✅ Analysé + bugs corrigés (dispatch mode, `ln1` morte, dropout non câblé) ; **`TransolverBlockv3` réécrit = port fidèle de Transolver-3** (arXiv:2602.04940 : poids par tête, température clampée, concat des têtes), sans tiling ; 3 modes `"linear"`/`"t3"`/`"physics"` (§17) |
| **ITNet** | `pynop/models/ITNet.py` | ⏳ À analyser |
| **UNet** (classique) | `pynop/models/unet.py` | ✅ Réécrit : U-Net classique (non-NO) validé — 4 blocs acceptés, PatchEmbed option, bottleneck transformer (PE **ou** RoPE) (§8) |
| **Resnet** (classique) | `pynop/models/resnet.py` | ✅ Unifiée : `block_type="basic"`/`"bottleneck"` + `bottleneck_ratio` + `se` ; `Resnetb` supprimé ; `resnet18`/`34`/`50`/`resnext50` fonctionnent (§11) |
| **Blocs communs** | `pynop/core/blocks.py`, `pynop/core/ops.py` | Blocs Swin déplacés ici ; **`TransformerBlock` unifié** avec option RoPE (`AdaptiveRoPE2D`) (§9, §10) |
| **Blocs IT** | `pynop/core/blocks.py` | ✅ `SepLITBlock`/`SepNLITBlock`/`LITBlock` vérifiés — 3 bugs corrigés (§12) ; **`IAETBlock`** (style IAE-Net) vérifié contre l'implémentation officielle — 4 bugs corrigés (§16) |
| **Training (Lightning)** | `pynop/training/lightning_model.py` | ✅ Réécrit : `predict_derivative` (sortie dérivée **ou** absolue), derivative-loss configurable, base unique `MultiStepLightningBase`, compat. tous les modèles du dépôt, `DeepONetModel` dédié ; wrappers AE/`CausalNO` supprimés (§15) |
| **Driver CLI d'entraînement** | `pynop/training/train.py`, `configs/` | ✅ Réécrit : driver config-driven (`ModelConfig.json` + `TrainingConfig.json`, flag `--pretrain`) reproduisant le protocole 2 phases des notebooks PDEBench ; dataset PDEBench seul pour l'instant ; checkpoints `last` (chaque époque) / best (dès max pas AR atteint) ; `--resume`/`--init-from` (§18) |
| **Eval CLI + métriques** | `pynop/training/eval.py`, `pynop/training/metrics.py` | ✅ Nouveau : évaluation d'un checkpoint par **rollout autorégressif de la trajectoire complète** (protocole notebooks PDEBench) sur des simulations **non utilisées en entraînement** (split val par défaut, ou split test = pool inutilisé) ; métriques de champ MSE/RMSE/**nRMSE = relative L2**/MAE/max-err/**R²** + bandes spectrales radiales style PDEBench (unités d'index bin FFT), **courbes d'erreur par pas de temps** (RMSE/MAE/R²/nRMSE + spectre) et graphiques PNG en anglais ; `eval_results_<split>.json` auto-contenu (§19) |

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
- Split train/val sur les IDs de simulation (shuffle seedé + troncature `n_samples`) : identique aux
  notebooks → phases cohérentes entre elles.

### Évaluation (`pynop/training/eval.py`, `metrics.py`) (§19)
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
  s'il existe, sinon re-déduits de la config ; `--split test` = pool inutilisé (au-delà de
  `n_samples`, même shuffle seedé que l'entraînement) avec repli sur val si vide ; chevauchement
  eval/train → `RuntimeError`.
- **Mémoire constante** : chaque prédiction est consommée via le callback `_rollout(on_step=...)` +
  `FieldMetricAccumulator.update_per_step(p, t, step)` (index explicite — appeler `update([p],[t])` par
  pas écraserait tout au pas 0). Batchs groupés par longueur égale (`_group_by_length`) ;
  `_rollout.num_predictions` est **par trajectoire** (× B pour le total). L'optimizer du wrapper
  Lightning est optionnel (inférence).
- **Sortie** : table stdout + `eval_results_<split>.json` (courbes complètes par pas avec `time`) +
  graphiques PNG en anglais (`eval_error_evolution_*.png` : RMSE/nRMSE/MAE log + R² ;
  `eval_spectral_evolution_*.png` : bandes radiales) via matplotlib Agg, désactivables par `--no-plots`.

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

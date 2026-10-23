# PyNOP — Notes de session

## Objectif global

Analyser, nettoyer, vérifier les implémentations et trouver les bugs dans la codebase
des **opérateurs neuronaux** (PyNOP = Py-Neural Operators), afin de **repartir sur de bonnes
bases** pour les sessions suivantes. Le travail se fait **modèle par modèle**, de façon
itérative, sur la branche `dev`.

> Principe : l'architecture des modèles est **conservée** (utilisée comme référence de
> comparaison). On ne cherche qu'à corriger les bugs, nettoyer, et documenter.

---

## État des modèles (à ce jour)

| Modèle | Fichier | Statut |
|---|---|---|
| **DeepONet** | `pynop/models/deeponet.py` | ✅ Corrigé (IndexError/AttributeError/NotImplementedError via `nn.Sequential`, logique `scale` flexible, docstrings raw strings) |
| **SwinNO / MultiscaleSWINNO** | `pynop/models/SWIN.py` | ✅ Nettoyé + validé (voir ci-dessous) |
| **LNO (`LatentNO`)** | `pynop/models/LNO.py` | ⚠️ Architecture correcte, mais **ne s'entraîne pas bien** — cause identifiée comme étant le *setup d'entraînement*, pas l'init (voir ci-dessous) |
| **LatentGalerkin** | `pynop/models/LNO.py` | ✅ Analysé + bugs corrigés (PE, RoPE, activation). ⚠️ Limité en **hautes fréquences** (compression globale lisse, pas de biais local) — voir ci-dessous |
| **ITLNO** | `pynop/models/LNO.py` | ✅ Analysé + bugs corrigés (voir §6). ⚠️ Défaut **hautes fréquences** (même cause que LatentGalerkin) — voir §6 |
| **FNO** | `pynop/models/FNO.py` | ⏳ À analyser |
| **Transolver** | `pynop/models/transolver.py` | ⏳ À analyser |
| **ITNet** | `pynop/models/ITNet.py` | ⏳ À analyser |
| **UNet** | `pynop/models/unet.py` | ⚠️ Typos corrigés, à vérifier |
| **Blocs communs** | `pynop/core/blocks.py`, `pynop/core/ops.py` | Blocs Swin déplacés ici depuis SWIN.py |

---

## Modifications apportées (session)

### 1. DeepONet (`pynop/models/deeponet.py`)
- Remplacement de l'indexation manuelle par `nn.Sequential` → corrige `IndexError`,
  `AttributeError`, `NotImplementedError`.
- Logique `scale` flexible : `None` → facteur fixe 1.0 ; `"auto"` → `nn.Parameter`
  initialisé à 1.0 ; numérique → facteur fixe.
- Correction des `SyntaxWarning` (docstrings en `r"""`).

### 2. SIGReg (`pynop/core/loss.py`)
- Vectorisation de la perte SIGReg.

### 3. SwinNO / MultiscaleSWINNO (`pynop/models/SWIN.py` + `pynop/core/blocks.py`)
- **Ménage** : SWIN.py réduit de 1651 → 613 lignes.
  - *Conservés* (modèles + enc/dec) : `SwinNO`, `MultiscaleSWINNO`, `MultiscaleCAEncoder`,
    `MultiscaleCADecoder`.
  - *Déplacés vers `core/blocks.py`* (blocs de base) : `WindowAttention`,
    `WindowCrossAttention`, `AdaptiveSwinBlock`, `PaddedSwinBlock`,
    `BlockCrossAttentionUpsampler`.
  - *Supprimés* (non utilisés) : `CondTransformerBlock`, `SwinEncoder`, `ResidualMLPBlock`,
    `CrossAttentionUpsampler`, `BidirectionalUpsampler`.
- **Bug corrigé** : `PaddedSwinBlock._build_coords` utilisait `.expand(b, -1, -1, -1)` (4
  tailles) sur un tenseur 5D → crash sur tout `pad=True`. Corrigé en
  `.expand(b, -1, -1, -1, -1)`.
- **Design validé** : les blocs **fixent le nombre de fenêtres** `(nw_h, nw_w)` et déduisent
  la taille de fenêtre de l'entrée (`win = H // nw_h`) → invariance de résolution.
  - ⚠️ Contrainte : l'entrée doit être **divisible** par le nb de fenêtres (sinon crash).
    Ajout d'un `assert`/`ValueError` clair dans les deux blocs.
  - ⚠️ Trade-off de coût : `O((H·W)²)` (vs `O(H·W)` du Swin classique) — inhérent au design.
- **Renommage** (clarté) : `winsize`/`windows` → `num_windows` (les paramètres désignaient le
  *nombre* de fenêtres, pas la taille). Affects : `SwinNO.num_windows`,
  `MultiscaleCAEncoder.num_windows`, `MultiscaleCADecoder.num_windows`,
  `MultiscaleSWINNO.enc_num_windows`/`dec_num_windows`. Le notebook
  `notebooks/training_multistep.ipynb` mis à jour en conséquence.

### 4. LNO (`pynop/models/LNO.py`) — investigation
- **Conclusion** : l'architecture et l'init sont **correctes** (identiques à la référence).
- **Idée reçue écartée** : dans le dépôt de référence
  (`L-I-M-I-T/LatentNeuralOperator`), la méthode `_init_weights` (std=0.0002) est
  **définie mais jamais appelée** (méthode morte) → la référence utilise l'init Kaiming par
  défaut, comme `LatentNO`. L'init « petite » est **instable** (testé : soit ne apprend pas,
  soit explose). **Aucun changement retenu** sur LNO (modifié puis reverté).
- **Cause probable du stagnation** : le *setup d'entraînement* (différences avec la référence :
  loss relatif `LpLoss` vs `MSELoss` brut ; gradient clipping absent ; scheduler
  `ReduceLROnPlateau` vs CosineAnnealing). **À creuser dans la prochaine session** avec le
  script d'entraînement exact de l'utilisateur.

### 5. `LatentGalerkin` — analyse critique + fixes (`pynop/models/LNO.py`, `core/encoding.py`, `core/blocks.py`)

**Contexte** : le modèle comprime le champ dans `M` tokens latents via une *Galerkin
Cross-Attention* (en réalité une **attention linéaire** sans softmax, `out = Q(KᵀV/n)`),
traite en latent, puis décode. Fonctionne, mais **peu adapté aux hautes fréquences**.

- **Analyse — cause racine du défaut HF** (plus fin que « une seule étape ») :
  - L'opérateur est **global et lisse** (passe-bas structurel) : les poids de projection
    sont des fonctions lisses de la position (bornées par la fréquence du PE), appliquées
    *position-by-position* avec la même transformation → **aucun biais spatial local**.
  - Compression de **rang faible** (`M ≪ N`) : les `M` tokens sont un résumé *lisse* ;
    l'information HF non contenue est **perdue** avant le traitement latent.
  - **Pas de skip** : toute la sortie passe par le goulot → elle ne peut exister que dans le
    cône engendré des `M` modes lisses.
  - **Pourquoi « plusieurs passes de Galerkin » ne suffisent pas** : empiler le *même*
    opérateur global lisse ajoute de la profondeur mais **aucun biais local** (c'est
    toujours un passe-bas). Le remède est un opérateur **local** (conv / noyau
    différentiel), pas plus de passes globales.
  - ⚠️ À retenir : le fix pertinent est une **branche locale** (conv 3×3 + skip, ou noyau
    différentiel NeuralOp-style = invariant de résolution, voir TODO `blocks.py`), ou
    augmenter `M`/fréquence PE (moins efficace).

- **Choix du PE** (analyse de `core/encoding.py`, toutes implémentations vérifiées correctes
  sauf `AdaptiveRoPE2D`) :
  - **Retenu : `"adaptive"` (`AdaptiveFourierEmbedding`)** — fréquences = **Nyquist de la
    résolution courante** (adaptatif, déterministe, HF en haute rés., anti-aliasing en basse).
    C'est l'invariance de résolution qui manquait.
  - `"random_ipe"` (ancien défaut) : IPE gaussien **anti-aliasé** (atténuation par
    `cell_size`) → partiellement adaptatif, bon compromis.
  - `"mlp"`/`"siren"` (MLP aux coordonnées) : le plus invariant mais pousse tout le HF sur
    le MLP → plus dur. **SIREN testé sans succès** : `w0=30` fixe (hardcoding), biais
    HF/instabilité, pas d'anti-aliasing.
  - `LatentGalerkin.pe` passé à `"adaptive"` par défaut.

- **Bugs corrigés** :
  1. **`pe_param` inopérant** — les callers passaient `max_freq_val=pe_param` mais `PEBlock`
     ne lisait que `scale`/`max_freq` → paramètre ignoré pour toutes les méthodes fréquence.
     Fix : `PEBlock` (`blocks.py`) accepte désormais `max_freq_val` en fallback
     (`random_fourier`/`fourier`/`ipe`/`random_ipe`/`adaptive`). Vérifié effectif.
  2. **`GalerkinTransformerBlock` crash au build** — défaut `activation=nn.GELU()` (une
     *instance*) puis appel `activation()` → `TypeError`. Cassait tout modèle en mode
     `latent_attention="galerkin"` (LatentGalerkin, LatentTransolver, autoencoder,
     Galerkin.py). Fix : `activation: Callable = nn.GELU` + appel défensif (classe ou
     instance). Vérifié forward+backward.
  3. **`AdaptiveRoPE2D` (RoPE 2D des blocs Swin) band-limitée** — fréq max plafonnée à `π`
     (constante) alors que le Nyquist d'une grille de H points est `π·H/2` ; échelle inversée
     (`h^(-exp)` : la fréquence *diminuait* avec la résolution) ; calcul basé sur la **taille
     de fenêtre** (inférée de `coords.shape`) et non la **résolution de la grille**.
     Conséquence : phase relative max dans une fenêtre = 0.22 cycle (< 1) → positions
     relatives non distinguées. **Fix** : fréquences en log-spacé depuis `Nyquist/base`
     jusqu'au **Nyquist de la grille** (`π·H/2`), nouveau paramètre `base` (défaut 16) ;
     `rotate()`/`WindowAttention`/`AdaptiveSwinBlock`/`PaddedSwinBlock` (2 branches)
     transmettent `resolution=(H,W)`. Vérifié : fréq max s'adapte (H=32 → 50.3 = Nyquist),
     phase relative = 3.5 cycles, RoPE adaptif à la résolution, SwinNO OK.

- **Bug repéré (non corrigé, hors périmètre)** : `ITLNO` (`LNO.py:~169`) —
  `self.ortho_init_weights(self.kernel_mixing, …)` appelé **inconditionnellement** quand
  `orthogonal_init=True` (défaut), mais `kernel_mixing` n'existe que si
  `linear_kernel=False` → `AttributeError` avec les défauts. À corriger dans la session ITLNO.

- **Vérifié** (env. conda `pytorch`) : `LatentGalerkin` (8 PE × 2 latent), `LatentNO`,
  `LatentTransolver`, `AdaptiveSwinBlock`, `PaddedSwinBlock` (pad F/T), `SwinNO` (swin/padded)
  → forward + circulation des gradients OK.

### 6. `ITLNO` — analyse critique + fixes (`pynop/models/LNO.py`)

**Contexte** : transformée intégrale apprenable (bases continues via MLP) → `M` modes latents
→ self-attention en latent → reconstruction spatiale. Même famille que `LatentGalerkin`, mais
avec des bases **trunk (coordonnées) / branch (signal)** et un noyau mixé.

- **Bugs bloquants corrigés** :
  1. **`kernel_mixing` `AttributeError`** — `ortho_init_weights(self.kernel_mixing, …)` appelé
     **inconditionnellement** (`orthogonal_init=True` par défaut) mais `kernel_mixing` n'existe
     que si `linear_kernel=False`. Crasht tout build avec `linear_kernel=True` (défaut).
     Fix : garde `if not linear_kernel:`.
  2. **`pe=None` (défaut) → crash au forward** — `trunk_generator` attendait
     `hidden_channels` et le `mixer` `2*hidden_channels`, mais `pe=None` renvoie les
     coordonnées brutes (2 dim) via `PEBlock(method=None)=Identity`. Fix : `basis_in_ch =
     ndim if pe is None else hidden_channels` câblé dans `trunk_generator` (`in_ch`) et
     `mixer` (`in_ch = hidden_channels + basis_in_ch`). `pe=None` = bases sur coordonnées
     brutes (comportement d'origine du backup).
- **Bugs latents corrigés** :
  3. **`latent_pe` mort** — défini mais **jamais appelé** (le backup l'appliquait via
     `xhat = xhat + PE`) → l'attention latente était **invariante par permutation** (les `M`
     modes indistinguables). Fix : PE latent sur la grille `m1×m2` ajouté à `xhat` avant
     l'attention. Vérifié : les 4 paramètres de `latent_pe` reçoivent maintenant des gradients.
  4. **`basis_mode="cheb"`** — `ChebyshevBasis` reçoit `pe` (256 dim) et utilise `pe[…,:2]`
     comme « coordonnées » (sens faux). Fix : la branche cheb passe les **coordonnées brutes**
     (`trunk_input = coords if basis_mode=="cheb" else pe`).
  5. **Nettoyage imports** — retirés (non utilisés dans `LNO.py`, re-exportés par ailleurs) :
     `SirenBlock`, `SpatioTemporalTransformer`, `AdaptiveLayerNorm`, `AdaRMSNorm`, `add_noise`,
     `MMDLoss`, et `from calendar import c` (import parasite). `F`/`math` conservés (usés par
     `LatentNO`).
- **Normalisation de la transformée (point soulevé) — analysé + amélioré** :
  - **Avant** : forward `xhat = Kᵀx / d` (`d`=norme L2 par mode ⇒ modes unitaires, OK) mais
    inverse `x_rec = K xhat / M` (facteur `1/M` **ad hoc**, incohérent avec le `/d` du forward ;
    le commentaire `# ** F.relu(self.IT_scale_inverse)` montrait que l'échelle apprenable
    d'origine avait été perdue au refactoring).
  - **Fix** : `it_scale_inverse` **apprenable** (init `1/M` ⇒ comportement initial préservé),
    symétrique du pattern `IT_scale_inverse` de `LITBlock`. Vérifié : init `1/64`, reçoit des
    gradients, se met à jour à l'entraînement.
  - **Option plus principielle (à discuter)** : orthonormaliser la base (QR /
    `Newton_Schulz`/`gs_orthogonalization` présents dans `core/utils.py`) ⇒ forward `Kᵀx`,
    inverse `K xhat` = **projection orthogonale propre** sur les modes adaptatifs
    (l'idée « le modèle trouve des bases adaptées, meilleures que Fourier »).
  - Garde `+1e-8` ajoutée à la division par `d` (évite NaN si un mode ≈ 0).
- **Défaut hautes fréquences** : **même cause que `LatentGalerkin`** — opérateur global et
  lisse (passe-bas structurel), compression de rang faible `M≪N`, pas de biais spatial local,
  pas de skip. Le vrai remède : une **branche locale** (conv 3×3 + skip, ou noyau
  différentiel NeuralOp-style, voir TODO `blocks.py`), pas plus de passes globales.
- **Vérifié** (env. conda `pytorch`, `PYTHONPATH` = source) : tous les configs
  (`linear_kernel` T/F, `pe=None`/`adaptive`/`random_fourier`, `basis_mode` cheb/learned,
  `modes` int/sequence, `cond_dim`) → forward fini + backward OK, **aucun paramètre mort**
  hors `branch_generator`/`cond_embedding` (attendu : non utilisés en mode linéaire / sans
  condition). Config exacte du notebook (`linear_kernel=False`, `pe=random_ipe`,
  `basis_mode=free`) → 0 param mort, invariance de résolution 16×16 & 32×32.

> ⚠️ **Installation** : `pynop` est installé **non-editable** dans `site-packages` (copie
> **obsolète**). Les modifications dans l'arborescence source ne sont vues que si on lance
> depuis le répertoire PyNOP (CWD/notebook) — sinon la copie `site-packages` est importée.
> Pour que les fixes soient toujours effectifs : `pip install -e .` (recommandé) ou
> `pip install .` pour resynchroniser.

---

## Prochaines étapes (session suivante)
1. [x] **ITLNO** (`pynop/models/LNO.py`) : corriger le bug bloquant `kernel_mixing`
       (accédé sans exister quand `linear_kernel=True`) → `AttributeError` avec les défauts. ✅ Fait
2. [ ] **ITLNO — normalisation principielle** (optionnel) : orthonormaliser la base
       (`Newton_Schulz`/QR dans `core/utils.py`) pour une projection orthogonale propre,
       au lieu de l'échelle apprenable ad hoc. À discuter.
3. [ ] **ITLNO/LatentGalerkin — hautes fréquences** (optionnel) : ajouter une **branche locale**
       autour du goulot (conv 3×3 + skip, ou noyau différentiel NeuralOp-style). C'est le vrai
       remède au passe-bas global.
4. [ ] **LNO** : obtenir le script/config d'entraînement exact → identifier et corriger le
       point de stagnation (candidats : loss relatif, grad clipping, LR+warmup, scheduler).
5. [ ] **FNO** (`pynop/models/FNO.py`) : analyser, vérifier l'implémentation, trouver les bugs.
6. [ ] **Transolver** (`pynop/models/transolver.py`) : idem.
7. [ ] **ITNet** (`pynop/models/ITNet.py`) : idem.
8. [ ] **UNet** (`pynop/models/unet.py`) : compléter la vérification.
9. [ ] Vérifier les autres classes de `pynop/core/blocks.py` et `pynop/core/ops.py`.

## Conventions
- Environnement : conda `pytorch` (Python 3.12, Linux).
- Brancher sur `dev` ; committer chaque étape validée.
- Toujours vérifier : import du package, forward pass, circulation des gradients.
- Écrire docstrings + type hints ; tester dans `tests/`.
- **Demander confirmation** avant tout commit / opération git destructive.

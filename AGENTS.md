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
| **UNO** (U-Net Neural Operator) | `pynop/models/UNO.py` | ✅ Nouveau : U-Net multi-échelle avec skips first-class, fusion `"add"`/`"concat"`, bottleneck transformer global, invariance de résolution (voir §13) |
| **LNO (`LatentNO`)** | `pynop/models/LNO.py` | ⚠️ Architecture correcte, mais **ne s'entraîne pas bien** — cause identifiée comme étant le *setup d'entraînement*, pas l'init (voir ci-dessous) |
| **LatentGalerkin** | `pynop/models/LNO.py` | ✅ Analysé + bugs corrigés (PE, RoPE, activation) + **branche locale** ajoutée (voir §7) |
| **ITLNO** | `pynop/models/LNO.py` | ✅ Analysé + bugs corrigés (voir §6) + **branche locale** ajoutée (voir §7) |
| **LatentTransolver** | `pynop/models/LNO.py` | ✅ Vérifié (Transolver3 correct) + nettoyage + **branche locale** ajoutée (voir §7) |
| **FNO** | `pynop/models/FNO.py` | ⏳ À analyser |
| **Transolver** | `pynop/models/transolver.py` | ⏳ À analyser |
| **ITNet** | `pynop/models/ITNet.py` | ⏳ À analyser |
| **UNet** (classique) | `pynop/models/unet.py` | ✅ Réécrit : U-Net classique (non-NO) validé — 4 blocs acceptés, PatchEmbed option, bottleneck transformer (PE **ou** RoPE) (voir §8) |
| **Resnet** (classique) | `pynop/models/resnet.py` | ✅ Unifiée : `block_type="basic"`/`"bottleneck"` + `bottleneck_ratio` + `se` ; `Resnetb` supprimé ; `resnet18`/`34`/`50`/`resnext50` fonctionnent (voir §11) |
| **Blocs communs** | `pynop/core/blocks.py`, `pynop/core/ops.py` | Blocs Swin déplacés ici ; **`TransformerBlock` unifié** avec option RoPE (`AdaptiveRoPE2D`) (voir §9) |
| **Blocs IT** | `pynop/core/blocks.py` | ✅ `SepLITBlock`/`SepNLITBlock`/`LITBlock` vérifiés — 3 bugs corrigés : norm channels-first, sur-normalisation `1/(H·W)` → échelle apprenable, shape du mixage de canaux (voir §12) |

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
- **Amélioration `WindowAttention`** (implémentation unique conservée) :
  - Remplacé le softmax manuel par `F.scaled_dot_product_attention` (kernel flash /
    fallback math) ; le masque additif décalé est compatible (vérifié : diff ≈ 1.5e-7
    vs référence manuelle, sans masque ≈ 9e-8).
  - Ajout de `qkv_bias` (défaut True) et `attn_dropout`/`proj_dropout` (défaut 0 = aucun
    changement de comportement).
- **Nouveau bloc `SwinBlock`** (`pynop/core/blocks.py`) : version « classique » de VAE
  (fenêtre **taille fixe** `window_size` défaut 7, padding droit/bas vers le multiple
  suivant, fenêtres décalées + masque Swin, maps channels-last `(B,H,W,C)`), **bâti sur la
  `WindowAttention` unique** de PyNOP (RoPE adaptatif via coordonnées absolues,
  `resolution=(H,W)` pour le Nyquist). Les 3 SwinBlocks (`SwinBlock`,
  `AdaptiveSwinBlock`, `PaddedSwinBlock`) partagent désormais cette seule attention.
- **Bug fixé** : `PaddedSwinBlock` (`pad=True`) avec **fenêtre impaire** → crash
  (`h_pad = h + 2·(win//2)` non divisible par `win`). Fix : pad d'un **total exactement
  `win` par axe** (split équitable, symétrique si `win` pair) tout en gardant ≥ `win//2`
  de contexte sur chaque côté.

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

### 7. `LatentTransolver` + **branche locale différentielle** (`pynop/models/LNO.py`, `pynop/core/ops.py`)

**Contexte** : `LatentTransolver` utilise la projection Transolver3 (slicing → latent attn →
deslicing) mais ne sous-performait pas le Transolver complet. Cause identifiée : **goulot
d'étranglement informationnel** — une seule projection de rang M ne peut pas exprimer la même
capacité que L compositions de slice→deslice (le Transolver complet retourne en espace spatial
à chaque couche, permettant d'extraire de nouvelles caractéristiques à chaque passe).

- **Vérification Transolver3** : `SlicingBlock` et `DeslicingBlock` sont **partagés** entre
  `TransolverBlockv3` et `LatentTransolver` — le mécanisme central (slicing data-dependent →
  attention latent → deslicing avec les mêmes poids) est identique. ✅ Correct.
- **Nettoyage** : paramètres morts retirés (`use_gumbel_softmax`, `dim`, `verbose`) +
  notebook `training_multistep.ipynb` mis à jour.

#### `FiniteDifferenceLayer` — nouvelle classe dans `pynop/core/ops.py`

Convolution différentielle NeuralOperator-style (Liu-Schiaffini et al., ICML 2024) :

```
K_diff = (K - mean(K)) / h
```

- **Centre le noyau** (soustraction de la moyenne) → annule le DC, opérateur passe-haut.
- **Rescale par 1/h** (h = espacement grille) → invariance de résolution : l'opérateur
  converge vers le même opérateur différentiel continu quand h→0, au lieu de s'annuler.
- Corrige le bug de `ConvFNOBlock` qui référençait `FiniteDifferenceLayer` sans la définir.

#### Branche locale intégrée aux 3 modèles latents

API commune (défaut `n_local_conv=0` = désactivé) :

| Paramètre | Description |
|---|---|
| `n_local_conv` | Nombre de conv différentielles (0 = off) |
| `local_filters` | Filtres par couche (int ou sequence) |
| `local_kernel_size` | Taille noyau (impair, défaut 3) |
| `grid_width` | Largeur domaine physique (défaut 2.0 pour [-1,1]) |

Architecture : `x_out = global_branch(x) + local_branch(x)`

- **`LatentTransolver`** : `x_in` = entrée brute → `FiniteDifferenceLayer` stack
- **`ITLNO`** : `x_in` sauvegardé avant permutation channel-last → `FiniteDifferenceLayer` stack
- **`LatentGalerkin`** : `x` (B,C,H,W) → `FiniteDifferenceLayer` stack

Norm inter-couches : `LayerNorm2d` (channel-first compatible). Activation : `mlp_act`.
Vérifié : forward + backward OK pour tous les configs (0/1/2/3 convs, filtres custom,
kernel 3/5, avec/sans condition).

> ⚠️ **Installation** : `pynop` est installé **non-editable** dans `site-packages` (copie
> **obsolète**). Les modifications dans l'arborescence source ne sont vues que si on lance
> depuis le répertoire PyNOP (CWD/notebook) — sinon la copie `site-packages` est importée.
> Pour que les fixes soient toujours effectifs : `pip install -e .` (recommandé) ou
> `pip install .` pour resynchroniser.

### 8. UNet classique (`pynop/models/unet.py`) — réécriture + bug corrigé dans les blocs

**Contexte** : l'ancienne classe `unet` (basée sur `MaxPoolConv`/`InterpolateConvUpSampleLayer`)
est remplacée par une **U-Net « traditionnelle »** (non opérateur neuronal), comme référence
pour `UNO` (voir §13).

**Nouvelle classe `UNet`** (alias backward-compat `unet = UNet`) :
- **Downsampling** : `Conv2d(kernel=3, stride=2, padding=1)` (l'ancien défaut sans padding
  perdait des pixels : 32→15→7→3 au lieu de 32→16→8→4 → crash de concat au décodage).
- **Upsampling** : `PixelShuffleUpsample` (`core/blocks.py`, sub-pixel conv, sans low-pass
  bilinaire), avec fusion skip intégrée (`skip_channels`).
- **Blocs acceptés** (les 4, vérifiés) : `ResBlock`, `BottleneckBlock`, `ConvNextv2Block`
  (étages conv) + **bottleneck transformer** (pile de `TransformerBlock` pre-LN, option
  `use_transformer_bottleneck=True/False`, style TransUNet). Le paramètre `block` est générique
  (classe ou `functools.partial`) ; les blocs sans `out_channels` (ConvNextv2Block) sont
  détectés via `inspect.signature`.
- **Stem** : `ConvLayer` 3×3 (défaut) ou **`PatchEmbed` optionnel** (`patch_size>=2`) — la
  head dé-patch via pixel-shuffle (`factor=patch_size`).
- **Fusion skip** : `fusion="concat"` (défaut) ou `"add"`.
- **Positional encoding du bottleneck** : soit **additif via `PEBlock`** (défaut,
  `method="adaptive"`, Nyquist adaptatif — exigence « PE via `PEBlock` »), soit **rotatif**
  (`rope=True`) via l'option RoPE du `TransformerBlock` unifié (voir §9) — les deux sont
  mutuellement exclusifs. `pe_method=None` **refusé** sans RoPE (l'attention serait
  invariante par permutation). Vérifié : le PE **et** le RoPE ont un effet mesurable.
- **Validation** : `filters`/`repeats` de même longueur, `factor>=2`, `fusion` valide,
  `filters[-1] % n_heads == 0`, et à l'exécution la divisibilité spatiale
  `H,W % (patch_size * 2**(n_stages-1)) == 0` (sinon `ValueError` explicite).
- **⚠️ Bug corrigé dans `ResBlock`/`BottleneckBlock` (`core/blocks.py`)** : le défaut
  `norm=partial(nn.GroupNorm, num_channels=32)` était cassé — appelé via `norm(C)`, il
  produisait `GroupNorm(num_groups=C, num_channels=32)` (mots-clés inversés) → crash pour
  tout `C > 32` et pour les canaux intermédiaires du bottleneck (ex. 8). Corrigé en
  `norm=LayerNorm2d` (ConvNeXt-style, channel-first, **agnostique du nb de canaux** —
  seul défaut robuste face aux mid-channels 8/16 de `BottleneckBlock`). `resnet.py`
  (toujours norm explicite = BatchNorm2d) non affecté.
- **Vérifié** (env. conda `pytorch`) : `pytest tests/test_unet.py` → **22/22 OK** :
  3 blocs × 2 fusions (forward/backward, forme exacte, fini, **aucun paramètre mort**),
  partial-kwargs style notebook, 3 étapes (res 16/64), bottleneck off, PatchEmbed 2/4,
  `TransformerBottleneck` seul, effet du PE, **bottleneck RoPE** (carré + non-carré,
  effet mesurable), `TransformerBlock(rope=...)` seul, alias `RoPETransformerBlock`
  déprécié, erreurs invalides (dont head-dim % 4 pour RoPE), alias.
  Smoke-test des consommateurs du bloc unifié : `SwinNO`, `LatentNO`, `LatentGalerkin`,
  `ITLNO`, `LatentTransolver`, `SpatioTemporalTransformer`, `LatentTemporalTransformerFullAttention`,
  `Transolver` → forward + gradients OK, aucun paramètre mort.
- **Trouvé puis résolu** : `Resnetb` (`models/resnet.py`) était cassé à l'instantiation
  (passe `norm_kwargs=` à `ConvLayer`/`BottleneckBlock`, paramètre inexistant, et
  `norm="bn"` string). → **Supprimé** et remplacé par `Resnet` unifiée (`block_type=
  "bottleneck"`), voir §11.

> ⚠️ `pynop` est installé non-editable (copie obsolète) : lancer depuis PyNOP ou `pip install -e .`.
> `tests/conftest.py` insère la racine du repo dans `sys.path` pour contourner le problème.

### 9. `TransformerBlock` unifié + option RoPE (`core/blocks.py`, `core/encoding.py`, `models/unet.py`)

**Contexte** : il existait **deux** implémentations de bloc transformer (`TransformerBlock` et
`RoPETransformerBlock`), et le bottleneck UNet composait ses propres blocs. Demande : **un seul
bloc `TransformerBlock` avec une option RoPE (`AdaptiveRoPE2D`)**, et mise à jour des classes qui
l'appellent — **périmètre limité aux modèles déjà validés dans AGENTS.md** (SwinNO, famille LNO)
+ `UNet` (les consommateurs hors périmètre — `SpatioTemporalTransformer`,
`LatentTemporalTransformerFullAttention`, `Transolver`, `backup.py` — ne sont **pas modifiés**).

- **`TransformerBlock`** (`core/blocks.py`) : nouveau paramètre `rope: bool = False` (+
  `rope_kwargs`). Si `rope=True`, le bloc possède un
  **`AdaptiveRoPE2D(dim // n_heads)`** appliqué à q/k quand `coords` est fourni (no-op sinon).
  `forward(x, coords=None, mask=None, rope=None, causal_attn=False)` : un `rope` **injecté**
  prend le **pas** sur le RoPE interne du bloc (préserve l'ancien pattern
  `RoPETransformerBlock` : RoPE 1-D temporel + coordonnées temps). `dim // n_heads % 4 != 0`
  avec `rope=True` → `ValueError` explicite.
- **`RoPETransformerBlock`** → **déprécié** : simple sous-classe de `TransformerBlock(rope=True)`
  gardant l'ancienne signature positionnelle `(dim, n_heads, mlp_dim, ...)`. Les callers
  historiques (injection RoPE 1-D) continuent de fonctionner (vérifié). Init aligné sur
  `TransformerBlock` (trunc-normal `std_ini`, pas xavier).
- **`AdaptiveRoPE2D`** (`core/encoding.py`) : ajout de `forward(t, coords, resolution=None)`
  déléguant à `rotate(...)` — rend le RoPE appelable via `rope(q, coords)` (protocole
  d'injection d'`Attention`). Le format 4-D `[B, H, W, 2]` désambiguise les grilles
  non-carrées (Nyquist correct par axe).
- **`UNet`/`TransformerBottleneck`** (`models/unet.py`) : nouvelle option `rope` (et
  `rope_kwargs`) — le RoPE du bloc **remplace** le PE additif `PEBlock` (mutuellement
  exclusifs). Coordonnées passees en 4-D pour les grilles non-carrées. `pe_method` ignoré si
  `rope=True`. Validation head-dim % 4.
- **Vérifié** (env. conda `pytorch`) : tests RoPE (bloc seul carré/non-carré, effet mesurable,
  alias déprécié + injection 1-D, UNet `rope=True` carré 32×32 + non-carré 64×32, erreurs
  head-dim) → tous OK. Smoke : les 8 consommateurs listés en §8 → forward + gradients OK,
  aucun paramètre mort.

### 10. Nettoyage `ops.py` — suppression des classes Shuffling/Unshuffling (`pynop/core/ops.py`, `pynop/core/blocks.py`, `models/resnet.py`)

**Objectif** : enlever de `ops.py` toutes les classes de *pixel-shuffle / pixel-unshuffle*,
conserver l'upsampler `PixelShuffleUpsample` déjà présent dans `blocks.py`, et déplacer dans
`blocks.py` les deux couches de redimensionnement non-shuffle.

- **Supprimées** (de `ops.py`, absentes du package entier) :
  - `ConvPixelUnshuffleDownSampleLayer`
  - `PixelUnshuffleChannelAveragingDownSampleLayer`
  - `ConvPixelShuffleUpSampleLayer`
  - `ChannelDuplicatingPixelUnshuffleUpSampleLayer`
  → `ops.py` ne référence plus `F.pixel_shuffle` / `F.pixel_unshuffle`.
- **Déplacées dans `blocks.py`** (+ docstrings + type hints) :
  - `StridedConvDownsamplingLayer` (conv stride = `factor`)
  - `InterpolateConvUpSampleLayer` (interpolation + conv)
  - toujours exportées au top-level (`pynop.*`) via `core/__init__.py`.
- **Conservée dans `ops.py`** : `MaxPoolConv` (avec docstring ajoutée).
- **Registres nettoyés** (`blocks.py`) :
  - `DOWNSAMPLING_LAYERS = {"stridedconv", "maxpooling"}` (retirés `convpixunshuffle`, `pixunshuffleaveraging`)
  - `UPSAMPLING_LAYERS = {"interpolate"}` (retirés `convpixshuffle`, `duplicatingpixshuffle`)
- **Nouveau helper `make_downsampler(...)`** (`blocks.py`) : instancie la couche de
  downsampling par clé du registre et lève `ValueError` avec une liste des méthodes valides
  si la clé est inconnue (remplace le fallback silencieux `.get(..., StridedConvDownsamplingLayer)`).
- **Blocs modifiés** (`blocks.py`) : `ResBlock` et `BottleneckBlock` — branche
  `pixunshuffleaveraging` supprimée, shortcut construit via `make_downsampler` ; docstrings +
  type hints ajoutés. `PixelShuffleUpsample` conservé (upsampler pixel-shuffle).
- **`models/resnet.py`** : docstrings de `Resnet`/`Resnetb` mises à jour (méthodes valides
  `"maxpooling"` / `"stridedconv"`). `Resnet` fonctionne avec les deux et rejette les méthodes
  supprimées. (`Resnetb`, cassé à l'instantiation, a depuis été **supprimé** et remplacé par
  `Resnet` unifiée — voir §11.)
- **Vérifié** (env. conda `pytorch`) : `pytest tests/` → **48/48 OK** (26 nouveaux tests
  `tests/test_resampling.py` + 22 `test_unet.py`) : registres, suppression des classes, formes +
  gradients de `StridedConvDownsamplingLayer`/`InterpolateConvUpSampleLayer`/`MaxPoolConv`,
  round-trip down/up, `ResBlock`/`BottleneckBlock`/`Resnet` (maxpooling & stridedconv),
  rejet des méthodes supprimées. Smoke : `UBlock`, `UFNOBlock`, `CoDABlock2D`, exports top-level.

### 11. `Resnet` unifiée — suppression de `Resnetb` (`pynop/models/resnet.py`)

**Contexte** : il existait deux classes, `Resnet` (blocs `ResBlock`) et `Resnetb`
(blocs `BottleneckBlock`). `Resnetb` était **cassé à l'instantiation** (passait
`norm_kwargs=` à `ConvLayer`/`BottleneckBlock`, paramètre inexistant, et `norm="bn"`
string). Demande : **supprimer `Resnetb`** et faire en sorte que `Resnet` utilise soit un
`ResBlock`, soit un `BottleneckBlock`, avec le paramètre dédié à la réduction du bottleneck.
Plus propre.

- **`Resnet` unifiée** : nouveau paramètre **`block_type`** (`"basic"` par défaut →
  `ResBlock` ; `"bottleneck"` → `BottleneckBlock`). Valeur inconnue → `ValueError` explicite.
  - **`bottleneck_ratio`** (défaut `4`) : réduction de la largeur mid du bottleneck
    (`mid = out // ratio`) — **uniquement** utilisé en `block_type="bottleneck"`.
  - **`se`** (défaut `False`) : porte Squeeze-and-Excitation dans le bottleneck
    (défaut `False` = ResNet standard, sans SE). Uniquement en `block_type="bottleneck"`.
  - Les scalaires `use_bias`/`norm`/`activation` sont passés **tels quels** à chaque bloc
    (chaque bloc construit son propre tuple via `make_tuple`) — évite le crash
    `make_tuple` (`.append` sur un tuple) quand `ResBlock` (n=2) et `BottleneckBlock` (n=3)
    attendent des longueurs différentes.
- **`Resnetb` supprimé** ; imports inutiles retirés (`make_tuple`, `LayerNorm2d`).
- **Factories** : `resnet18`/`resnet34` → `Resnet` (basic, inchangé) ; `resnet50` →
  `Resnet(block_type="bottleneck")` ; `resnext50` →
  `Resnet(block_type="bottleneck", groups=32, bottleneck_ratio=2)`. `resnet50`/`resnext50`
  fonctionnent désormais (étaient cassés via `Resnetb`).
- **Vérifié** (env. conda `pytorch`) : `pytest tests/` → **62/62 OK** (dont **14 nouveaux**
  dans `tests/test_resnet.py`) : forward+backward `basic`/`bottleneck` (maxpooling &
  stridedconv), `bottleneck_ratio` contrôle la capacité (ratio 2 > 8 en nb de params),
  `se` ajoute des params, routage vers `ResBlock`/`BottleneckBlock`, `resnet18`/`34` = basic,
  `resnet50`/`resnext50` = bottleneck (groups=32, ratio 2), `Resnetb` absent du module,
   `block_type` invalide rejeté. Imports top-level OK (`Resnetb` n'est plus exporté).

### 12. Blocs IT — `SepLITBlock` / `SepNLITBlock` / `LITBlock` (`pynop/core/blocks.py`)

**Contexte** : vérification des trois blocs *Learned Integral Transform* + test des
valeurs d'activation **après la transformée intégrale** (soupçon : la normalisation
constante est peu adaptée). `LITBlock` (utilisé par `ITNet`, supprimé) sert de référence.

**Mesures** (x ~ N(0,1), C=8, m=(8,8), résolutions 16/32/64) :
- La sortie brute de la transformée (avant normalisation) est **O(1–17)** — du bon
  ordre, comparable à l'entrée. Mais la division constante `1/(H·W)` (256/1024/4096)
  la réduisait à **O(2·10⁻³–5·10⁻³)**, soit **~1/50–1/400 du shortcut** (≈0.6) → la
  branche apprise était **morte** (sortie dominée par shortcut + biais du mixer).
  `LITBlock` fait mieux : échelle **apprenable** `sigmoid(IT_scale)` + `RMSNorm` →
  branche IT **O(0.35)** ≈ shortcut, invariante en résolution.
- ⚠️ La base apprise n'est **pas unit-énergie** (norme ~1–2.4, grandit sous-linéairement
  avec H) → l'hypothèse DFT de la constante `1/(H·W)` est fausse.

**Bugs corrigés** :
1. **Layout de la norm** — `SepLITBlock`/`SepNLITBlock` opèrent en channels-first
   `(B,C,H,W)` mais appliqueaient `AdaptiveLayerNorm` (channels-last) →
   `RuntimeError` dès que `out_channels ≠ W`. Fix : défaut `norm=LayerNorm2d`
   (channels-first, normalise sur C).
2. **Sur-normalisation** — remplacé la constante `1/(H·W)` par une échelle **apprenable**
   `(H·W)**sigmoid(IT_scale)` (nouveau paramètre `IT_scale`, init `-1.1` → sigmoid≈0.25,
   sortie IT O(1) à l'init). La `LayerNorm2d` (après le mixer) retire le biais DC et
   renormalise à O(1) → branche IT **O(1)** ≈ shortcut, invariante en résolution.
3. **Mixage de canaux `SepNLITBlock`** — `learned_weights_freq` était
   `(in, out, m1, m2)` alors que l'`einsum "bixy,oixy->boxy"` (et le commentaire)
   attendent `(out, in, m1, m2)` → crash si `in ≠ out` et mauvais mixage sinon.
   Fix : shape `(out_channels, in_channels, m1, m2)` (aligné sur `SepLITBlock`).

**Nettoyage imports** (classes retirées de `blocks.py`, réf. éliminées là où elles
cassaient) : `pynop/models/FNO.py` (retiré `UFNOBlock`/`ConvFNOBlock` de l'import +
`REGISTERED_FNO` + docstring) ; `tests/test_unet.py` (retiré `RoPETransformerBlock` +
son test d'alias + import `RoPE`). Le package `pynop` importe à nouveau.
(`pynop/models/backup.py` référence encore d'anciennes classes mais n'est pas importé.)

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **69/69 OK** (dont **8 nouveaux**
dans `tests/test_lit_blocks.py`) : forward+backward + aucun param mort pour les 3 blocs,
`in≠out` (mixage canaux), norm channels-first, **branche IT O(1) ≈ shortcut**
(ratio 0.1–10), `IT_scale` apprenable et effectif, invariance de résolution. Smoke
`in=4/out=12/m1=6/m2=8` → OK, aucun param mort.

> ⚠️ `pynop` est installé non-editable (copie obsolète) : lancer depuis PyNOP ou `pip install -e .`.

### 13. UNO — U-Net Neural Operator (`pynop/models/UNO.py`)

**Contexte** : la demande initiale était « une version traditionnelle **et** une version
NO_unet ». La version classique (`UNet`, §8) était déjà faite ; `UNO` est la version
**opérateur neuronal** : un U-Net multi-échelle construit sur les blocs Swin
(`AdaptiveSwinBlock`) avec **skips first-class** entre encodeur et décodeur.

**Architecture** :
```
input (B, Cin, H, W)
→ Stem          : PE (adaptive) + Linear → (B, H, W, D0)
→ Enc stage i   : repeats × (SwinBlock + ShiftSwinBlock)  → skip[i]
→ Down i        : WindowCrossAttention  → (B, H_i, W_i, D_i+1)
→ Bottleneck    : repeats × TransformerBlock (global, plus petite résolution)
→ Up i          : BlockCrossAttentionUpsampler (queries = skip[i], keys/values = z)
→ Dec stage i   : [skip fusion] + repeats × (SwinBlock + ShiftSwinBlock)
→ Head          : RMSNorm + Linear + GELU + Linear → (B, Cout, H, W)
```

**Différences vs `MultiscaleSWINNO`** :

| | MultiscaleSWINNO | UNO |
|---|---|---|
| Skips | Residual interne de l'upsampler (additif) | **First-class** : fusion explicite après self-attention |
| Fusion | Toujours `add` (residual) | Configurable : `"add"` ou `"concat"` (+ 1×1 proj) |
| Bottleneck | Non explicite (output_proj) | **Explicite** : `bottleneck_repeats` × TransformerBlock global |
| API | Encodeur/décodeur séparés | **Symétrique** : un seul objet, stages indépendamment configurables |
| Invariance rés. | Oui (fenêtres fixes, taille déduite) | Oui (idem) |

**Paramètres clés** :
- `grid_resolutions` : liste de tuples `(H, W)` par étape (plus grande → plus petite)
- `enc_num_windows` / `dec_num_windows` : nombre de fenêtres par étape (`None` = global)
- `model_dim` : int ou `List[int]` (dim par niveau)
- `enc_repeats` / `dec_repeats` : paires de SwinBlocks par étape
- `bottleneck_repeats` : nb de TransformerBlocks au bottleneck
- `skip_fusion` : `"add"` (défaut, ≈ MultiscaleSWINNO) ou `"concat"`
- `cond_dim` : optionnel (conditionnement)

**PE au stem** : l'analyse de la discussion révèle que le RoPE (positions relatives) dans
l'attention couvre déjà le besoin de position pour les poids d'attention. Le PE additif au
stem apporte en plus : (1) information positionnelle pour la **première MLP** (sinon
identique pour des points d'entrée identiques), (2) position **absolue** (utile pour
conditions aux limites positionnelles). Conclusion : **utile mais non strictement
nécessaire** — garder comme défaut, envisager un flag `use_pe` pour un design « pur NO ».

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **97/97 OK** (dont **28 nouveaux**
dans `tests/test_uno.py`) : construction (1/2/3 stages, fusion add/concat, fusion invalide
rejetée), forward (carré, non-carré 96×64/192×128), backward (aucun param mort),
invariance de résolution (64/128, 96×64/192×128), conditionnement, stages globaux
(`None`). Smoke : `SwinNO` et consommateurs des blocs partagés → OK.

> ⚠️ `pynop` est installé non-editable (copie obsolète) : lancer depuis PyNOP ou `pip install -e .`.

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
8. [x] **UNet** (`pynop/models/unet.py`) : version classique réécrite + 4 blocs vérifiés. ✅ Fait
9. [x] **UNO** (`pynop/models/UNO.py`) : version *neural operator* du U-Net — U-Net
       multi-échelle avec skips first-class, fusion `"add"`/`"concat"`, bottleneck
       transformer global, invariance de résolution (voir §13). ✅ Fait
 10. [ ] Vérifier les autres classes de `pynop/core/blocks.py` et `pynop/core/ops.py`.
 11. [x] **`Resnetb`** (`models/resnet.py`) : **supprimé** — `Resnet` unifiée avec
        `block_type="basic"/"bottleneck"` + `bottleneck_ratio` + `se` (voir §11). ✅ Fait
 12. [x] **Blocs IT** (`pynop/core/blocks.py`) : `SepLITBlock`/`SepNLITBlock`/`LITBlock`
        vérifiés — 3 bugs corrigés (norm channels-first, sur-normalisation → échelle
        apprenable, shape du mixage de canaux `SepNLITBlock`). ✅ Fait (voir §12)

## Conventions
- Environnement : conda `pytorch` (Python 3.12, Linux).
- Brancher sur `dev` ; committer chaque étape validée.
- Toujours vérifier : import du package, forward pass, circulation des gradients.
- Écrire docstrings + type hints ; tester dans `tests/`.
- **Demander confirmation** avant tout commit / opération git destructive.

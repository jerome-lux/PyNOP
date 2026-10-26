# PyNOP — Journal détaillé des sessions

Ce fichier contient l'**analyse complète** par modèle/bloc : contexte, bugs trouvés
(symptômes + causes), mesures, arbitrages et vérifications. C'est la référence de détail ;
`AGENTS.md` en est le résumé opérationnel (statut des modèles, décisions clés, prochaines étapes).

Les numéros `§N` sont **stables** et référencés depuis `AGENTS.md`.

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

> ✅ **Installation** : `pynop` est installé **en mode éditable** (`pip install -e .`) —
> les modifications dans l'arborescence source sont vues immédiatement, sans resynchronisation.

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

> ✅ `pynop` est installé en mode éditable (`pip install -e .`) — les modifications source sont vues immédiatement.

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

> ✅ `pynop` est installé en mode éditable (`pip install -e .`) — les modifications source sont vues immédiatement.

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
- `enc_num_windows` / `dec_num_windows` : nombre de fenêtres par étape (`None` = global).
  La longueur détermine `n_stages`. Pour garder une taille de fenêtre constante,
  diminuer le nb de fenêtres par `downsample_factor` à chaque stage
  (ex. `[(16,16),(8,8),(4,4)]` avec facteur 2).
- `downsample_factor` : facteur de downsampling par stage (défaut 2).
  Résolution au stage i = `input_res / downsample_factor**i`.
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

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **99/99 OK** (dont **30 nouveaux**
dans `tests/test_uno.py`) : construction (1/2/3 stages, fusion add/concat, fusion invalide
rejetée, downsample_factor invalide rejeté), forward (carré, non-carré 96×64),
backward (aucun param mort), invariance de résolution (32/64/128), conditionnement,
stages globaux (`None`), validation spatial size. Smoke : `SwinNO` et consommateurs
des blocs partagés → OK.

> ✅ `pynop` est installé en mode éditable (`pip install -e .`) — les modifications source sont vues immédiatement.

---

### 14. FNO — blocs spectraux + modèle (`pynop/core/ops.py`, `pynop/core/blocks.py`, `pynop/models/FNO.py`)

**Contexte** : vérification des blocs FNO (les 3 convolutions spectrales + `FNOBlock` +
modèle `FNO`). Objectif : rendre l'ordre des opérations du bloc cohérent avec les dépôts
de référence (neuralop / Li et al. 2021), tout en gardant la normalisation et
l'activation optionnelles et le choix parmi les **3** convolutions spectrales
(`standard`, `separable`, `tucker`).

**Bugs bloquants corrigés** :
1. **`FNO.forward` crashait à 100 %** — `AttributeError: 'FNO' object has no attribute
   'trainable_pos_encoding'` : le paramètre `trainable_pos_encoding` (et
   `pos_embedding_weights`) avait été retiré de `__init__` lors du cleanup §12 alors que
   le docstring et le `forward` (et le notebook `examples.ipynb`) y faisaient toujours
   référence. Fix : paramètres restaurés (`trainable_pos_encoding`,
   `trainable_pos_encoding_modes`, `trainable_pos_encoding_dims`) + PE spectral
   apprenable re-câblée (resampelée à la résolution d'entrée via `irfftn`).
2. **Tucker + `spectral_compression_factor=None` crashait au build** —
   `TypeError: unsupported operand for /: 'int' and 'NoneType'` (la ligne
   `ranks = (1,1,1)` était morte, immédiatement écrasée). Fix : `None` ⇒ pas de
   compression (facteur `1`), i.e. rangs = `(cin, cout, prod(modes))`.
3. **Clamping de modes qui corrompait l'état** (les 3 convolutions spectrales) — quand
   l'entrée était plus petite que les modes configurés, le `forward` **mutait**
   `self.modes_x/modes_y` (→ warning à chaque appel + module cassé pour *toutes* les
   entrées suivantes) et crashait l'`einsum` (shape des poids fixée à l'init). Fix :
   modes effectifs = `min(configuré, disponible)` appliqués à **l'entrée et aux
   poids** (sous-espace basse fréquence) — aucune mutation d'état, **invariance de
   résolution** (vérifié : même instance, entrée 8×8 puis 32×32 OK).

**Bugs/design corrigés** :
4. **Double activation de `FNOBlock`** — l'activation était appliquée 2× (garde morte
   `if self.activation is not None:` autour d'une 2ᵉ application). Fix : bloc aligné sur
   la référence — `[norm] → spectral_conv → [activation] → pointwise(1×1) → + shortcut`
   — avec **norm** et **activation optionnelles** (`None` pour désactiver). L'activation
   est appliquée **exactement une fois** (entre la conv spectrale et le pointwise).
5. **Fallback silencieux du registre** — `REGISTERED_FNO.get(name, FNOBlock)`
   construisait un `FNOBlock` pour n'importe quelle chaîne (ex. `"UFNO"` du notebook).
   Fix : `ValueError` explicite listant les types valides (`FNO`).
6. **API incohérente** — `modes` int, `hidden_channels` int et `blocks` scalaire
   désormais acceptés (broadcast propre) ; `block_kwargs` n'est plus une default
   mutable ; `dt` conservé (réservé, inutilisé) pour compatibilité d'API.

**Choix de design** :
- Les 3 convolutions spectrales restent disponibles et interchangeables via
  `spectral_layer_type` dans `FNOBlock` **et** le modèle `FNO`.
- `spectral_compression_factor` n'a d'effet que pour `tucker` (ignore pour
  `standard`/`separable`) ; `int` ou séquence de 3.
- `FNOBlockv2` (variante double-résidu, Kossaifi et al.) **supprimée** — retirée de
  `blocks.py`, `FNO.py` (import + registre + docstring), et `tests/test_fno.py`.

**Branche locale high-frequency (`local_conv`)** — nouveau paramètre de `FNOBlock`
et du modèle `FNO` :
- `local_conv: int = 0` — nombre de `FiniteDifferenceLayer` dans la branche locale
  (0 = désactivé). La branche est ajoutée **en parallèle du shortcut**, opérant sur
  l'entrée brute du bloc : `output = main_branch + shortcut(x) + local_branch(x)`.
- `local_kernel_size: int = 3` — taille du noyau (impair).
- `local_grid_width: float = 2.0` — largeur du domaine physique (pour `h = W/(N-1)`).
- Motivation : la branche spectrale est un **passe-bas structurel** (modes bas only) ;
  la conv différentielle (`K_diff = (K - mean(K)) / h`) est un **passe-haut**
  resolution-invariant (NeuralOperator-style, Liu-Schiaffini et al. ICML 2024).
  Ensemble, elles couvrent le spectre complet.
- Implémentation : `nn.Sequential` de `FiniteDifferenceLayer` (in→out pour la 1ʳᵉ,
  out→out pour les suivantes). Gradients vérifiés sur chaque couche.
- Vérifié : `local_conv` 0/1/2/3, `in≠out`, kernel 3/5, invariance de résolution,
  gradient flow, parallélisme au shortcut (zeroing change l'output).

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **140/140 OK** (dont **41
tests** dans `tests/test_fno.py`) : les 3 convolutions (entrée < et > modes, sans
mutation d'état, gradients), `FNOBlock` (3 types spectraux, int modes, norm/activation
optionnelles, `hidden_channels`, invariance de résolution, tucker sans ranks rejeté,
type spectral inconnu rejeté, ordre de référence, **branche locale** 0/1/2/3 convs,
`in≠out`, kernel 3/5, gradient flow, parallélisme shortcut), modèle `FNO` (forward
sans crash, PE apprenable + gradient, PE fixe off, tucker `None`/`2`, separable,
entrées scalaires, **local_conv 0/1/2**, `cond`, bloc inconnu rejeté, invariance de
résolution, grille non-carrée, longueurs incohérentes rejetées). Tous sans paramètre mort.

> ✅ `pynop` est installé en mode éditable (`pip install -e .`) — les modifications source sont vues immédiatement.

### 15. `lightning_model` — réécriture (`pynop/training/lightning_model.py`, `pynop/core/loss.py`)

**Contexte** : le fichier de travail était **tronqué** (55 lignes au lieu de 1188, coupé au
milieu d'une docstring — une réécriture partielle avait perdu le reste). L'objectif de la
session : (1) pouvoir préciser si la sortie du réseau est la **prédiction absolue** ou la
**dérivée**, (2) vérifier `derivative_loss_fn` par défaut, (3) clarifier le code,
(4) vérifier la compatibilité avec l'ensemble des modèles du dépôt, (5) décider, en cas
d'incompatibilité, de modifier le modèle ou de créer un nouveau lightning model.

**1. `predict_derivative` (dérivée vs absolue)** — nouveau champ `TrainingConfig`
(défaut `True` = comportement d'origine) :
- `True` : `out = du/dt` → `preds = u_t + dt·out`.
- `False` : `out = u_{t+dt}` → `preds = out`. La perte dérivée (si active) est alors
  évaluée sur la **dérivée implicite** `(preds − u_t)/dt` — restant un régulariseur
  cohérent dans les deux modes.
- Implémenté une seule fois dans `MultiStepLightningBase._rollout`, partagé par
  `MultiStepNOModel` et `DeepONetModel`.

**2. `derivative_loss_fn` par défaut — vérifié + rendu configurable** :
- Défaut conservé : `NormalizedTimeDerivativeMSE` (nMSE de la dérivée prédite vs
  dérivée finie `(target − previous)/dt`, par canal, invariant d'échelle) — le choix
  est sound. Nettoyé : docstring, paramètre `input` → `previous` (shadowing builtin).
- Rendu **configurable** : `TrainingConfig.derivative_loss_fn` (`None` → défaut) et
  `derivative_loss_weight = 0.0` **désactive** la perte (plus du tout calculée).
- **Cohérence train/val corrigée** : la perte dérivée était calculée seulement à
  l'entraînement ; elle l'est désormais dans les deux boucles (`val_nMSEdt`).
- **RMSE unifié** : la validation utilisait « moyenne des RMSE par sample »,
  l'entraînement « RMSE global » — le même RMSE global est désormais utilisé partout
  (cohérent avec la note du notebook : moyenne sur spatial/canal/batch).

**3. Clarté** :
- `configure_optimizers` était **duplicé 4×** → une seule implémentation dans la
  classe de base (avec `forward`, `_unpack_batch`, normalisation `loss_fn`/`loss_weights`).
- **Fusion des deux classes de base** (`BaseLightningNOModel` — plumbing — et
  `MultiStepLightningBase` — rollout) en une seule `MultiStepLightningBase` : après la
  suppression des wrappers autoencodeur/causal (non utilisés) tous les modules
  Lightning du dépôt sont des wrappers multi-step, donc la base contient à la fois le
  plumbing (modèle, optimizer, config, losses, introspection de signature,
  `configure_optimizers`) **et** la machine multi-step (rollout
  teacher-forcing/AR, curriculum, bruit, BPTT-truncation, logging). Les sous-classes
  (`MultiStepNOModel`, `DeepONetModel`) n'implémentent plus que `_forward_window`.
- **Wrappers supprimés** (non utilisés) : `CausalNOModel`,
  `AutoEncoderModel`/`AutoEncoderModelv2` (+ `AETrainingConfig`). La famille
  SpatioTemporal (`backup.py`) n'est plus couverte par un wrapper Lightning — à
  recréer si besoin.
- **Bugs corrigés** :
  - `default_scheduler_config` utilisait la clé `"name"` alors que
    `configure_optimizers` pop `"scheduler"` → `KeyError` à l'usage. Fix : clé
    `"scheduler"` + classe réelle + `monitor="avg_loss"`.
  - L'extraction des clés Lightning (`monitor`/`interval`/…) était faite **après** la
    construction du scheduler → `TypeError: ReduceLROnPlateau got unexpected kwarg
    'monitor'`. Fix : pop **avant** construction (comportement d'origine restauré).
  - `TrainingConfig.save`/`to_json_dict` intelligents (sér. des loss en noms de
    classe) — perdus dans la version tronquée, restaurés (le notebook appelle
    `train_config.save(...)`).
- Validation : `n_slices > 0`, `detach_grad_steps >= 0`, `dt > 0`.

**4. Compatibilité avec l'ensemble des modèles** (détection par introspection de
`model.forward` ; les mots-clés optionnels ne sont passés **que si acceptés**) :

| Modèle | signature | Verdict |
|---|---|---|
| FNO, LatentNO, LatentGalerkin, ITLNO, LatentTransolver, SwinNO, MultiscaleSWINNO, UNO, Transolver | `(x, cond=None)` | ✅ `MultiStepNOModel` |
| UNet, Resnet, ConvNext | `(x)` | ✅ `MultiStepNOModel` (cond non passé ; **erreur explicite** si le batch en fournit) |
| GalerkinTransformer | `(x, time, residual)` | ✅ `MultiStepNOModel` — `time` passé en `(B,1)` (contrat du modèle), `residual=True` = sortie **absolue** → `predict_derivative=False` |
| **DeepONet** | `(u, y)` — paradigme branch/trunk | ✅ **nouveau `DeepONetModel`** (voir §5) |
| Famille SpatioTemporal (backup.py) | `(x, time, history, ...)` | ⛔ non couvert (wrapper `CausalNO` supprimé, non utilisé) |

**5. Décision « modifier le modèle ou nouveau lightning model » — DeepONet** :
`DeepONet.forward(u, y)` est un paradigme différent (opérateur sur fonction : branch =
valeurs de la fonction, trunk = coordonnées de requête) → **nouvelle classe
`DeepONetModel`** (les modèles ne sont pas modifiés). Mapping :
- branch input = la fenêtre **aplatie** en vecteur fonction `(B, n·C·H·W)`
  (`branchnet[0] = n·C·H·W`) — le layout par-point `(B, H·W, n·C)` **cassait l'einsum**
  de DeepONet (`"ji, ki...->kj..."` exige dim-1 du branch = largeur du trunk),
- trunk input = coordonnées de la grille de sortie `(H·W, 2)` (2-D, sans batch, sinon
  l' MLP du trunk porte un batch et casse l'einsum),
- sortie `(B, H·W)` (ou `(B·H·W, C)` si decoder) → reshape `(B, 1, H, W)`.
`MultiStepNOModel` **refuse** un DeepONet avec un message pointant vers `DeepONetModel`.

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **213/213 OK** (dont **32**
dans `tests/test_lightning_model.py`) : FNO (2 modes + cond + gradients), SwinNO, UNet,
Resnet, GalerkinTransformer (time), UNO, Transolver, LatentNO, DeepONet (2 modes +
mapping branch/trunk + rejet cond), rejet DeepONet dans `MultiStepNOModel`,
formule `predict_derivative` (dérivée/absolue), derivative-loss (défaut / weight 0 /
custom), curriculum AR + feedback AR + bruit, optimizer/scheduler
(`default_scheduler_config` non muté, `ReduceLROnPlateau`), abstraction
`MultiStepLightningBase` (`_forward_window` non implémenté → `NotImplementedError`),
`fit()` end-to-end avec `pl.Trainer`, exports top-level.

> ✅ `pynop` est installé en mode éditable (`pip install -e .`) — les modifications source sont vues immédiatement.

### 16. `IAETBlock` — vérification contre l'implémentation officielle IAE-Net + fixes (`pynop/core/blocks.py`)

**Contexte** : `IAETBlock` (« Non Reversible Non Linear Integral Transform Block ») est le
bloc de la famille IT le plus proche d'**IAE-NET** (« Integral Autoencoders for
Discretization-Invariant Learning », Ong/Shen/Yang, arXiv:2203.05142 ; dépôt officiel
`github.com/IAE-Net/iae_net`, `libs/layers/iae_blocks.py`) : transformée intégrale 2-D
**non linéaire** (noyaux conditionnés sur les données) avec **transformée inverse
indépendamment apprise** (d'où « non réversible »). Comparaison faite contre le code
officiel (`Encoder`/`Decoder`/`TokenConv`, cas 2-D).

**Logique validée comme conforme à l'officiel** :
- Ordre des passes : encodeur W→H, décodeur H→W (ordre inverse) — identique.
- Conditionnement des noyaux d'encodeur : valeurs au point + coordonnée (passage 1),
  valeurs après le passage 1 + coordonnée (passage 2) — identique.
- Conditionnement du décodeur : **compresseur** (`Linear(m→1)`) sur les coefficients
  spectraux, en conservant l'indice de mode qui indexe la ligne utilisée dans le produit
  scalaire et en compressant l'index sommé (vérifié terme à terme contre `compressor_x`/
  `compressor_y`), répété le long de la dimension spatiale de sortie — identique.

**Bugs corrigés** :
1. **Crash entrée complexe** — `xc = … else x.float()` : `.float()` sur un tenseur
   complexe64 **jette la partie imaginaire** (UserWarning) et produit un tenseur *réel* →
   le premier matmul (réel × bases complexes de `ComplexMLPBlock`) lève
   `RuntimeError: expected scalar type Float but found ComplexFloat`. Le bloc était donc
   **inutilisable avec une entrée complexe**. Fix : le bloc opère sur la **partie réelle**
   de son entrée, uniformément pour la transformée et le shortcut (`x_real = x.real if
   x.is_complex() else x`, puis embedding complexe à partie imaginaire nulle) — l'intention
   d'origine (`x.float()` = « rendre réel »), sans le crash. Documenté dans le docstring ;
   sortie toujours réelle. (Le support *plein* complexe — shortcut complexe inclus — reste
   une extension future.)
2. **Shape transposée de `learned_weights_freq`** — construite `(in, out, m1, m2)` alors
   que l'einsum `"bixy,oixy->boxy"` attend `(out, in, m1, m2)` → crash si `in ≠ out`,
   **mixage de canaux erroné sinon** (même bug que `SepNLITBlock`, §12). Fix : shape
   `(out_channels, in_channels, m1, m2)`.
3. **Init morte + init tous-à-un** — `torch.randn(…)` immédiatement écrasée par
   `nn.init.constant_(…, 1.0)` ; des poids 1+0j couplent chaque canal de sortie à la somme
   de tous les canaux d'entrée (amplification O(√C_in) si `in ≠ out`). Fix : init
   Xavier-style complexe `std = (1/(in+out))^0.5` sur parties réelle et imaginaire
   (convention des blocs frères §12).
4. **Norm spectrale manquante → explosion de magnitude avec la résolution** — l'officiel
   applique un LayerNorm (`ln1`) après la 2ᵉ transformée de chaque encodeur/décodeur ; le
   port PyNOP l'avait omis. Mesuré (in=out=8, m=(8,8), x~N(0,1)) : sortie brute de la
   chaîne 0.18 → 0.36 → **5210** et conditionnements du décodeur 0.45 → 1.84 → **68** aux
   résolutions 16/32/64 (les sommes de Riemann non échelonnées divergent, et les noyaux
   inverse *conditionnés sur ces valeurs* étaient pilotés par la magnitude, pas le contenu).
   Fix : `ln_freq = ComplexLayerNorm(-1)` (RMS complexe-safe sur la dim canaux —
   `F.layer_norm` ne supporte pas les complexes ; bien comporté même pour C=1) appliqué à
   la représentation spectrale après l'encodeur, avant le mixage fréquentiel. Après fix :
   représentation spectrale O(0.92–0.93), conditionnements O(0.38–0.39), sortie brute
   bornée (0.07–0.09) et ratio branche-IT/shortcut **0.70 constant** aux résolutions
   16/32/64.
5. **Échelle finale fixe `1/(h·w)`** — remplacée par l'exposant **apprenable**
   `(H·W)**sigmoid(IT_scale)` (init `-1.1` → sigmoid≈0.25), convention de la famille §12 ;
   le facteur constant supposait une base unit-énergie (faux pour une base apprise) et,
   avec `norm=None`, faisait décroître la sortie en 1/N².

**Notes de design (écart assumés vs l'officiel, documentés dans le docstring)** : bases
complexes (`ComplexMLPBlock`) au lieu de réelles ; coordonnées `[-1,1]` au lieu de `[0,1]` ;
grilles **non carrées** supportées (l'officiel 2-D suppose des grilles carrées) ; branche
unique (le chemin Fourier parallèle et les connexions denses vivent dans le wrapper modèle
officiel `TokenDenseBlock`, pas dans le bloc).

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **219/219 OK** (dont **6 nouveaux**
dans `tests/test_lit_blocks.py`) : forward+backward sans param mort, `in≠out` (4→12) +
shape des poids, entrée complexe (pas de crash, pas de warning, `block(z) == block(Re z)`),
norm spectrale O(1) et invariante en résolution (hook sur `ln_freq`, 16/32/64), ratio
branche-IT/shortcut ∈ [0.1, 10] constant en résolution, `IT_scale` apprenable + effectif,
grille non carrée 32×48, chemin `norm=None`.

### 17. `Transolver` — analyse + fixes des blocs et du modèle (`pynop/models/transolver.py`, `pynop/core/blocks.py`)

**Contexte** : le wrapper `Transolver` expose trois familles de blocs via `mode` :
- `"linear"` → `LinearNOBlock` (attention linéaire canonique, « Transolver is a Linear
  Transformer », arXiv:2511.06294) ;
- `"t3"` → `TransolverBlockv3` (Transolver-3, arXiv:2602.04940 — voir **correction** ci-dessous) ;
- attention physique Gumbel-Softmax → `TransolverBlock` (Transolver / Transolver++ officiel).

Références vérifiées : `thuml/Transolver_plus@b2f656d` (`Physics_Attention_1D_Eidetic`,
`Transolver_plus_block`) ; `thuml/Transolver@main` (`Physics_Attention.py`,
`PDE-Solving-StandardBenchmark/model/Transolver_Structured_Mesh_2D.py`) ; `HiPRL/LinearNO@main`
(`Standard_PDE_Benchmark/model/LinearAttnNeuralOperator.py`) + Eq. (1)–(9) d'arXiv:2511.06294v3 ;
**`lbruand-db/distributed-transolver3@main`** (`transolver3/physics_attention_v3.py`,
`transolver3/transolver3_block.py`) + arXiv:2602.04940 pour le bloc v3.

> ⚠️ **Deux papiers distincts, initialement confondus** : *Transolver-3* (arXiv:2602.04940) et
> *LinearNO* (« Transolver is a Linear Transformer », arXiv:2511.06294). Le mécanisme de slicing
> `w` + normalisation diagonale appartient à la famille Transolver (arXiv:2402.02366) ; LinearNO le
> *reformule* en attention linéaire canonique (softmax asymétriques, pas de matrice `w` explicite).

**Bugs corrigés** :
1. **Dispatch des modes — `elif "t3":` toujours vrai** (`models/transolver.py`) : la condition
   testait la vérité d'une chaîne non vide → tout mode ≠ `"linear"` (y compris une faute de
   frappe) construisait silencieusement des `TransolverBlockv3`, et la branche attention physique
   était **inaccessible**. Fix : dispatch explicite `"linear"` / `"t3"` / `"physics"`, avec
   `ValueError` listant les modes valides sinon.
2. **`ln1` morte dans `TransolverBlock`** : le forward calculait `x_mid = self.ln1(x)` puis
   l'écrasait immédiatement par `self.in_project_x(x)` — la norme n'était jamais appliquée
   (grads poids/biais None). La référence officielle fait `fx = Attn(ln_1(fx)) + fx`. Fix :
   `in_project_x(lin1(x))` — pre-norm avant projection, comme l'officiel.
3. **Dropout non utilisé dans `TransolverBlock`** : `self.dropout` était créé mais jamais appelé
   (le paramètre `dropout` du modèle n'avait aucun effet). Fix : appliqué sur la sortie attention
   projetée, avant l'ajout résiduel (`dropout(proj(ln2(out))) + x`) — même position que le
   `to_out = Linear + Dropout` de l'officiel.
 4. **Ordre des opérations dans `SlicingBlock` (biais)** : le code faisait
    `linear1(bmm(wᵀ, x)) / d` (projection *après* la somme pondérée) alors que le slicing de la
    famille Transolver définit `s_j = Σ_i w_ij v_i / d_j` avec `v_i = Linear(V)(h_i)` calculé
    **avant** la moyenne. Avec un terme de biais, les deux ordres diffèrent : projeter après échelle
    le biais par `1/d_j` au lieu de l'ajouter une fois par slice (diff max mesurée 0.296). Fix :
    `bmm(wᵀ, linear1(x)) / d`. Affecte aussi `LatentTransolver` (`LNO.py`) qui réutilise ces blocs —
    re-vérifié après fix. *(Citation corrigée en §17b : le mécanisme est celui de la famille
    Transolver / Transolver-3, pas l'Eq. (3) d'arXiv:2511.06294.)*
5. **Paramètres du modèle ignorés en mode `"t3"`** : `dropout` (durci à 0.1) et `activation`
   (GELU durci) n'étaient pas transmis à `TransolverBlockv3`. Fix : paramètre `activation` ajouté
   au bloc ; les deux sont maintenant propagés depuis le modèle.

**Vérifié correct (inchangé)** :
- Math de `LinearNOBlock` : conforme aux Eq. (8)–(9) et au code officiel (`softmax(q, dim=M)`,
  `softmax(kᵀ, dim=N)`, forme factorisée `φ(Q)(ψ(K)ᵀV)`), diff max 0.0 ; sommes de lignes = 1.
- `DeslicingBlock` : projeter les slices avant deslice est **exactement** équivalent à deslice puis
  projection (lignes de `w` somment à 1 → le biais passe inchangé).
- Structure fusionnée de `TransolverBlock` (`proj = MLPBlock(inner→dim)` au lieu du
  `to_out + mlp` séparés de l'officiel) : simplification assumée, architecture conservée.
- `_init_weights` écrasant l'init orthogonale de `in_project_slice` : même comportement dans la
   référence officielle (re-init globale après construction des blocs) — conservé tel quel.

**Nettoyage** : suppression du code mort `MHSlicingBlock`/`MHDeslicingBlock` (variantes multi-têtes
nulle part utilisées, y compris dans les notebooks) ; imports inutilisés retirés de
`models/transolver.py` (`math`, `partial`, `F`, `Sequence`, `TransformerBlock`, normes et activations
ré-exportées, `print_stats`) ; commentaires corrompus réparés (`?`, `r�sidue`) ; docstrings ajoutés
aux classes modifiées.

#### 17b. Correction — `TransolverBlockv3` réécrit comme port fidèle de Transolver-3 (arXiv:2602.04940)

**Constat** : la première vérification (§17) avait comparé le bloc v3 à *LinearNO*
(arXiv:2511.06294) — **mauvais papier**. Contre la vraie référence
(`lbruand-db/distributed-transolver3` : `PhysicsAttentionV3` + `Transolver3Block`), l'ancien bloc
PyNOP déviateur sur 4 points :

| Aspect | Référence (papier) | Ancien bloc PyNOP |
|---|---|---|
| Poids de slicing | **par tête** : `w [B,H,N,M]` via `Linear(C → H·M)` init orthogonale | jeu unique `[B,N,M]` partagé entre têtes |
| Température | apprenable **par tête** `[1,H,1,1]`, init 0.5, clamp `[0.1, 5]` | scalaire `softplus(θ)+0.05` (init ≈ 1.05) |
| Attention en espace-slice | une seule couche SDPA sur les M slices/tête (`to_q/k/v: D→D`, partagés entre têtes), **sans FFN** dans le module | pile de `TransformerBlock` complets (MHA + FFN 4D) avec layout de têtes pynop |
| Deslice / composition du bloc | `slice_linear3` par slice puis **concaténation des têtes** (`C = H·D`, asserté) ; bloc = `Attn(ln1(fx))+fx` puis `mlp(ln2(fx))+fx` (MLP séparée, ratio 1 dans le modèle de réf.) | projection unique `D→C` + scatter poids partagés ; un seul pre-norm, résidu autour de tout, FFN dans les TransformerBlocks |

**Fix** : `TransolverBlockv3` réécrit (`core/blocks.py`) comme port fidèle — signature
`(dim, heads, dim_head=None, slice_num, mlp_ratio=1, dropout, activation)`, avec assert
`dim == heads * dim_head`. Tiling et state-caching **volontairement absents** (optimisations
mémoire/inférence mathématiquement identiques au chemin standard — confirmé par lecture du code de
référence : accumulation `s_raw`/`d` par tuile puis mêmes opérations en espace-slice). L'init
orthogonale de `in_project_slice` est écrasée par la re-init globale du wrapper (trunc-normal 0.02) —
**même comportement que le modèle de référence**, qui fait pareil.

**Non modifié** : `SlicingBlock`/`DeslicingBlock` (forme à jeu unique, utilisés par
`LatentTransolver`) — leur docstring cite désormais Transolver-3 / la famille Transolver au lieu de
LinearNO ; le docstring de `LinearNOBlock` est complété (titre complet + arXiv:2511.06294 +
mécanisme des softmax asymétriques Eq. 8–9).

**Vérifié** (env. conda `pytorch`) : `pytest tests/` → **247/247 OK** (dont **28** dans
`tests/test_transolver.py`, +4 nouveaux) : référence manuelle complète du bloc v3 (slicing par tête,
SDPA, concat des têtes, MLP séparée), poids par tête distincts entre têtes avec lignes sommant à 1 sur
M, température clampée `[0.1, 5]` vérifiée aux deux bornes, assert `dim == H·D`, dispatch par mode +
rejet des modes inconnus, shapes forward des 3 modes ± conditionnement, flux de gradient vers tous les
paramètres (aucun param mort), grad non nul sur `ln1` (`TransolverBlock`), dropout réellement sur le
chemin, `SlicingBlock` conforme et distinct de l'ordre buggé, équivalence deslice, attention linéaire
 manuelle pour `LinearNOBlock`, propagation dropout/activation dans les 3 modes, aucune mutation d'état
 entre résolutions 16×16 / 24×20.

 > ✅ `pynop` est installé en mode éditable (`pip install -e .`) — les modifications source sont vues immédiatement.

### 18. `train.py` — driver d'entraînement CLI, réécriture complète (`pynop/training/train.py`, `configs/`, `tests/test_train_cli.py`)

**Contexte** : l'ancien `pynop/training/train.py` (boucles torch brutes `train_step` /
`train_unrolled` / `CoDANO_training` + TensorBoard) ne reproduisait pas le protocole des
notebooks PDEBench (curriculum AR, losses multiples pondérées, régulariseur de dérivée,
checkpoints conditionnés). Rien n'importait ces fonctions (vérifié par grep) → remplacement
total. Le nouveau module est un **driver config-driven** qui réutilise `MultiStepNOModel` /
`DeepONetModel` (§15) — aucune logique de rollout dupliquée, pas de visualisation/évaluation :
entraînement uniquement.

**CLI** :
```bash
python -m pynop.training.train MODEL_CONFIG.json TRAINING_CONFIG.json [--pretrain] \
    [--device auto|cuda|cpu] [--output-root DIR] [--resume CKPT] [--init-from CKPT]
```
- `--pretrain` → section `"pretraining"` de TrainingConfig ; sans le flag → `"fulltraining"`.
- `MODEL_CONFIG.json` : `{name?, model, params}` — `model` = classe de `pynop.models` (ou
  chemin pointé) ; les chaînes d'activation (`"GELU"`, …) sont résolues en classes `nn.*`.
- `TRAINING_CONFIG.json` : `input_steps` + section **`dataset`** (type, path, batch_size,
  num_workers, load_in_ram, shift/scale, split_ratio, random_seed — pour l'instant
  `"pdebench"` uniquement) + `shared_params` (timestep, loss_weights, temporal_weighting,
  detach_grad_steps, derivative_loss_weight, predict_derivative, noise_level,
  time_normalization) + `losses` (liste `{name, params}` résolue dans `core.loss` puis
  `torch.nn`) + `optimizer` / `scheduler` + `trainer` (clip grad, log_every_n_steps) +
  `output` (root/experiment) + `loggers` (`csv`, `tensorboard`) + les deux sections de phase.
- Phases : `delta_ts`, `min_id`/`max_id`, `n_samples`, `lr`, `max_epochs`, `start_AR` /
  `final_AR`, `min_AR_steps` / `max_AR_steps` (+ override optionnel `scheduler`). La fenêtre est
  dérivée : `window_size = max_AR_steps + input_steps + 1`. Le split train/val se fait sur les
  **IDs de simulation** (shuffle seedé, troncature à `n_samples`, ratio) — identique aux
  notebooks → les deux phases partagent le même sous-ensemble.
- Scheduler : tokens `"$total_steps"` / `"$steps_per_epoch"` remplacés par
  `max_epochs * len(train_loader)` / `len(train_loader)` ; OneCycleLR sans `max_lr` → `lr` de la
  phase (comme les notebooks). Format dict transmis tel quel à `configure_optimizers`.

**Politique de checkpoints** (répertoire
`<root>/<experiment>/<logdir phase>/<name modèle>/<input_steps>steps_<timestamp>/`) :
- `last.ckpt` — sauvegardé **à chaque époque dès la première** (écrasé) : on a toujours les
  poids de la dernière époque (reprise après crash, warm-start phase 2 ← `last.ckpt` phase 1).
- `best_RMSE.ckpt` / `best_val_RMSE.ckpt` (top-k=2, fichiers secondaires `-vN`) — le monitoring
  ne démarre qu'**à l'époque où le rollout atteint le max de pas AR** (`epoch >= final_AR`) :
  pendant la rampe du curriculum les métriques d'époques successives ne sont pas comparables.
  Si `final_AR >= max_epochs`, aucun best n'est écrit (warning affiché au démarrage).

**Bugs corrigés en route** :
1. **`CurriculumCheckpoint` contourné sous Lightning 2.x** (`training/lightning_model.py`) :
   l'ancienne implé ne gérait que `on_validation_end`, mais avec PL ≥ 2 (ici 2.6) et
   `val_check_interval=1` (défaut), la sauvegarde se fait dans **`on_train_epoch_end`** → le gate
   était inopérant (best écrits dès l'époque 0). Fix : le gate est appliqué dans
   `_should_skip_saving_checkpoint`, le point de passage commun à tous les hooks de sauvegarde
   (+ `on_exception`), et les checkpoints du driver passent `save_on_train_epoch_end=False` pour
   épingler la sauvegarde en fin de validation (les deux métriques `RMSE`/`val_RMSE` y sont
   disponibles).
2. Ancien module : boucles non utilisées par le protocole NO multi-étapes — supprimées
   (`train_step`, `test_step`, `train`, `adversarial_train`, `train_unrolled`,
   `CoDANO_training`) ; récupérables dans l'historique git si besoin.

**Vérifié** (env. conda `pytorch`, torch 2.14 / PL 2.6) :
- `pytest tests/` → **262/262 OK**, dont 15 nouveaux (`tests/test_train_cli.py`) : parsing CLI,
  sélection de phase, builders (losses/optimizer/scheduler/tokens), split déterministe des IDs,
  et 4 runs end-to-end CPU sur un HDF5 PDEBench synthétique — politique de checkpoints vérifiée
  dans les deux sens (gate atteint → best présents ; gate non atteint → `last.ckpt` seul) +
  warm-start `--init-from`.
- Smoke test GPU sur le vrai dataset (`133017.hdf5`, FNOvanilla 1.64 M params, 8 sims,
  fenêtre 21) : losses DWMSE+Spectral+Sobolev (poids 1e4), dérivée `NormalizedTimeDerivativeMSE`,
  OneCycleLR (div_factor 1e3 → pic à lr phase), RMSE par pas AR loggés (`step_1..step_10`),
  `last.ckpt`/`best_RMSE.ckpt`/`best_val_RMSE.ckpt` + configs et `run_info.json` écrits.
- Reprise après crash : run 2 époques interrompu (SIGINT) à ~52 % de l'époque 1 →
  `--resume last.ckpt` avec la même config reprend exactement à l'époque 1 et termine.
  ⚠️ Limite inhérente au protocole : un resume OneCycleLR exige le **même nombre total de pas**
  (on ne peut pas augmenter `max_epochs` après coup — le schedule est figé).

**Exemples** : `configs/model_FNOvanilla.json` + `configs/training_PDEBench.json` reproduisent à
l'identique les paramètres des notebooks (`/run/media/jlux/SSD2/pyNOP`) — prétraining 40 sims /
t∈[0,50] / stride 2 / lr 2e-4 / 300 époques / AR 2→10 sur [0,200], fulltraining 400 sims / tout t /
stride 4 / lr 1e-4 / 100 époques / AR fixe 10.

 ### 19. `eval.py` — driver d'évaluation CLI + métriques de champ (`pynop/training/eval.py`, `pynop/training/metrics.py`, `tests/test_eval_cli.py`)

**Contexte** : demande utilisateur — un module compagnon de `train.py` (§18) qui charge un
checkpoint entraîné et évalue sur des **échantillons non utilisés pendant l'entraînement**, avec
des métriques par bande spectrale (style PDEBench), MSE/nMSE et autres. Point de vigilance soulevé :
la pertinence de la normalisation du nMSE quand le dataset est recentré à moyenne nulle → arbitrage
documenté ci-dessous.

**Choix des métriques** (`pynop/training/metrics.py`) — toutes calculées sur les champs en
**espace modèle** (après `(x + shift) * scale` du dataset), tenseurs channels-first `(..., C, H, W)` :
- **nRMSE = relative L2** (le « normalized RMSE » de PDEBench, `pdebench/models/metrics.py::metric_func`) :
  par échantillon ET canal `‖pred − target‖₂ / ‖target‖₂` (normes sur les dims spatiales), moyenné.
  **Invariante d'échelle** ; pour des champs normalisés à moyenne nulle `(x−μ)/σ`, le dénominateur est
  exactement l'écart-type du champ → métrique toujours significative (c'est ainsi que PDEBench l'utilise).
  Les normalisations par `mean(target)²` dégénèrent sur les champs à moyenne nulle → **délibérément non utilisées**.
- **R²** par échantillon/canal : `1 − ‖err‖² / ‖target − mean_spatial(target)‖²`, centrage **spatial
  uniquement** (pas global). Invariante à toute rénormalisation affine (translation ET échelle) → la plus
  sûre pour les datasets à moyenne nulle ; identité exacte par unité : `R²_{i,c} = 1 − nRMSE²_{i,c}` quand
  le champ est à moyenne nulle par canal.
- **Bande spectrale** : `rfftn` orthonormale sur `(H, W)`, énergie d'erreur groupée par coquilles radiales
  en **unités d'index de bin FFT** (`k = sqrt(iy² + ix²)`, `iy = min(k, h−k)` — l'ordre fft), bandes par
  défaut `[0,4) / [4,12) / [12,+∞)` = `iLow=4`/`iHigh=12` de PDEBench. Par défaut **relative**
  (`‖F(err)_bande‖₂ / ‖F(target)_bande‖₂` par unité, moyenné — généralisation spectrale du nRMSE, donc
  invariante d'échelle) ; mode absolu (énergie brute, plus proche de l'erreur Fourier raw de PDEBench) via
  `relative_spectral=False`. ⚠️ Les bornes sont des indices de coquille **dépendants de la résolution**
  (comme chez PDEBench), pas des fréquences physiques.
- **`FieldMetricAccumulator`** : accumulation incrémentale par batch (mémoire constante) avec
  **pooling exact** sur toutes les unités (fenêtre, pas[, canal]) — jamais une moyenne de moyennes ;
  courbe **par pas AR** (RMSE + nRMSE à chaque index de pas, poolés entre batches) = le diagnostic clé de
  croissance d'erreur du rollout. `compute_field_metrics` = wrapper one-shot équivalent.

**Driver CLI** (`pynop/training/eval.py`) :
```bash
python -m pynop.training.eval MODEL_CONFIG.json TRAINING_CONFIG.json --checkpoint PATH \
    [--split val|test] [--n-test N] [--pretrain] [--device auto|cuda|cpu] \
    [--batch-size B] [--max-windows W] [--output-dir DIR]
```
- Mêmes configs que l'entraînement ; la section de phase fournit le protocole de fenêtres (stride,
  plage t, `window_size = max_AR_steps + input_steps + 1`) — identique à celui de l'entraînement.
- **Sélection du jeu d'évaluation** (`select_eval_ids`) :
  - `--split val` (défaut) : le split validation du protocole — disjoint de l'entraînement par construction.
    Si un `run_info.json` (§18) est trouvé à côté du checkpoint, ses `train_ids`/`val_ids` stockés sont la
    **source de vérité** (robuste au drift de config après entraînement).
  - `--split test` : les premières `n_test` IDs du **pool inutilisé** — IDs jamais dans aucune phase
    (au-delà de `n_samples` dans l'ordre du shuffle seedé, même protocole RNG que `train.py`). Repli sur le
    split val (avec warning) si le pool est vide (toutes les sims du fichier ont été consommées).
  - Garde-fou : tout chevauchement eval/train lève une `RuntimeError`.
- **Inférence** : wrapper Lightning sans optimizer (`optimizer` rendu optionnel dans
  `MultiStepLightningBase.__init__`), `TrainingConfig` d'inférence (losses désactivées → liste vide,
  bruit 0, BPTT off, `predict_derivative`/`dt`/normalisation temporelle repris de `shared_params`).
  Rollout **AR complet** via `_rollout(..., training=False, store_fields=True)` — nouveau flag qui
  collecte les prédictions/cibles détachées par pas (liste `(B, C, H, W)`) sans rien changer au chemin
  d'entraînement. `DeepONetModel` vs `MultiStepNOModel` choisi via `_is_deeponet`.
- **Sortie** : table formatée sur stdout + `eval_results_<split>.json` (à côté du checkpoint ou
  `--output-dir`) auto-contenu : toutes les métriques scalaires, bandes spectrales, courbe par pas AR,
  bookkeeping complet (checkpoint, phase, split effectif, source des IDs, IDs évalués + train_ids en audit).
- Section optionnelle `"eval"` de TrainingConfig : `spectral_bands` (défaut PDEBench),
  `spectral_relative` (défaut true), `n_test` (taille du split test).

**Bugs corrigés en route** :
1. `radial_frequency_grid` initialement en **cycles/échantillon** (`fftfreq`) → les bornes de bandes
   PDEBench (4, 12) n'avaient plus le même sens ; remplacé par l'**index de bin FFT** (coquilles de
   PDEBench), docstrings alignés.
2. `r_squared` initialement centrée globalement → refaite **par (échantillon, canal)** avec centrage
   spatial uniquement (même unité que le nRMSE ; identité `R² = 1 − nRMSE²` vérifiée numériquement).
3. `pynop/training/__init__.py` : import mort `from .callbacks import *` supprimé (module retiré au §15) ;
   `from .eval import *` ajouté.

**Vérifié** (env. conda `pytorch`) :
- `pytest tests/` → **282/282 OK**, dont 20 nouveaux (`tests/test_eval_cli.py`) : invariances des
  métriques (nRMSE d'échelle, R² affine, bandes spectrales relatives), identité par unité
  `R² = 1 − nRMSE²` sur données à moyenne nulle, propriétés du grid radial (forme, symétrie fft, rayon max),
  équivalence one-shot ≡ accumulation incrémentale (pooling exact) + courbe par pas ; sélection des splits
  (config / `run_info.json` / repli test→val / pool inutilisé / split inconnu) ; dataloader d'évaluation
  (protocole de fenêtres, shapes, cap `max_windows`) ; et **3 runs end-to-end CPU** sur HDF5 synthétique :
  1 époque de prétraining → évaluation du split val (IDs pris dans le `run_info.json` du run), du split test
  (disjoint de train ET val) et via l'entrée CLI `main()`.

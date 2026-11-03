# Trained weights

All files are attached to the release of this repository (not stored in git). Run names: `orig_d8` and `orig_d18` are the unseeded runs of the first version (seed 1), `<variant>_s<seed>` the seeded runs, `tp_<matcher>_ft_s1` MatchFormer and LGFCTR retrained on the aerial stream (derived from Apache-2.0 weights).

| File | Run | Epoch | Note | Size (MB) | SHA-256 |
|---|---|---|---|---|---|
| `L0_s1_ep011.pth` | L0_s1 | ep011 | selected by the 15-pair rule | 53 | `e7e8901737ee9e3e12671e01535b1c4bcf9ac49d05aa1f68010c8fd8c93bee1b` |
| `L0_s2_ep012.pth` | L0_s2 | ep012 | selected by the 15-pair rule | 53 | `cac66fa3b5fd78b3dedbc0f74c515f201a74d192ee0ad30bf1637ee12c54debc` |
| `L4_s1_ep008.pth` | L4_s1 | ep008 | selected by the 15-pair rule | 55 | `42021e20c1c8eaf3bff084243a4e709d7ed96d91726e049baf5c3de8b887ce80` |
| `L4_s2_ep012.pth` | L4_s2 | ep012 | selected by the 15-pair rule | 55 | `1d0ea8f0b8c083e1240ba472069f9c7f6d35af1e3f7390ce1f855c1c3c07db2b` |
| `clean_s1_ep012.pth` | clean_s1 | ep012 | selected by the 15-pair rule | 58 | `00e06da07a6514d98d717556d62e8ff0b71461c8265ff687483ad575bf835a34` |
| `clean_s2_ep009.pth` | clean_s2 | ep009 | selected by the 15-pair rule | 58 | `4820cd51e1bebee8afaaae4f11d13465e8049b5825bde20d08951de3d399c6a8` |
| `d18_s2_ep008.pth` | d18_s2 | ep008 | selected by the 15-pair rule | 118 | `c1f50d0941e5c074146243e29be0ea6ef7b02dcea4972ab50edf901e30c0d33a` |
| `d18_s3_ep007.pth` | d18_s3 | ep007 | selected by the 15-pair rule | 118 | `d927e231198bce6f6c6d7e99d373c7961f54ff994a15fdd87cdbb729b8213515` |
| `d8_s2_ep012.pth` | d8_s2 | ep012 | selected by the 15-pair rule | 58 | `3b2a2c713cc21903169425d7d7c5338a5f122d45b7a66f97e72fb2b03c04f552` |
| `d8_s3_ep012.pth` | d8_s3 | ep012 | selected by the 15-pair rule | 58 | `1c11f40d2443ae9b37e38a59dbc57b6ec700c20d1d39000296a93a0f1e7d7e79` |
| `orig_d18_ep010.pth` | orig_d18 | ep010 | selected by the 15-pair rule | 118 | `19b4e10684888cbca46fbbfda162a477627a364fc7e404dedbba84df569651e3` |
| `orig_d8_ep008.pth` | orig_d8 | ep008 | selected by the 15-pair rule | 58 | `7366722d6bf019ca3b9b91401bb2ce746af762901636f940830ef74faf76f5f7` |
| `pub_s1_ep012.pth` | pub_s1 | ep012 | selected by the 15-pair rule | 58 | `5b25b10dd8bfd2d3506261eae24398382b87ce6ddd0b3c2db2b0c3557b4ea2a1` |
| `pub_s1_ep012.pth` | pub_s1 | ep012 | primary checkpoint of the public reproduction | 58 | `5b25b10dd8bfd2d3506261eae24398382b87ce6ddd0b3c2db2b0c3557b4ea2a1` |
| `r480_s1_ep001.pth` | r480_s1 | ep001 | selected by the 15-pair rule | 58 | `3c01b883f49fcc5c33c9d6e1b036ab3e53cc91dad5a90d310ce9a357c3908428` |
| `r480_s2_ep001.pth` | r480_s2 | ep001 | selected by the 15-pair rule | 58 | `82dd865ca36b21e9467225a8f8bfb02c62af74de4edda62e84be0365102c85cb` |
| `s32_s1_ep008.pth` | s32_s1 | ep008 | selected by the 15-pair rule | 58 | `baa36b1342664b193979c52c8838dfa80b4c079d6f09136930975b641b5d5453` |
| `s32_s2_ep012.pth` | s32_s2 | ep012 | selected by the 15-pair rule | 58 | `6ca72e51ddb431d9f3a03a9a095a15805c867bca00701c414476b7469a0c1391` |
| `tp_lgfctr_ft_s1_ep012.pth` | tp_lgfctr_ft_s1 | ep012 | selected by the 15-pair rule | 188 | `30597b69d6e66076aca270862efdd18b02bce3a8f3f9598d66c89b2869983d83` |
| `tp_matchformer_ft_s1_ep002.pth` | tp_matchformer_ft_s1 | ep002 | selected by the 15-pair rule | 77 | `74991c26342062406c954cde3ce4364497e4e350c37f1982c5e512a30ed3d7af` |
| `predecessor_s32.pth` | earlier /32 version | - | initialisation of the inherited variants | 170 | `237f7427511015cec0f71150549a826a27bf276cd03786277758e3ecc367e607` |

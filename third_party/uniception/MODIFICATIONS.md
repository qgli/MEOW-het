# Modifications in this copy of UniCeption

This directory contains UniCeption 0.1.7 (https://github.com/castacks/UniCeption, BSD 3-Clause
License, see LICENSE) with one change, made for MEOW:

- `uniception/models/info_sharing/alternating_attention_transformer.py`: an optional
  `variable_resolution` argument. When it is set, the IFR variant of the alternating-attention
  transformer accepts a list of per-view feature maps of different sizes (global attention over
  all tokens, frame attention per view). With the default (`False`) the module behaves exactly as
  the original.

The package version is marked `0.1.7+meow`; it still satisfies the `uniception==0.1.7`
requirement of MapAnything.

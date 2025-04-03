## Backbones quick introduction


### unett.py
- flat unet transformer
- structure same as in e2-tts & voicebox paper except using rotary pos emb
- update: allow possible abs pos emb & convnextv2 blocks for embedded text before concat

### dit.py
- adaln-zero dit
- embedded timestep as condition
- concatted noised_input + masked_cond + embedded_text, linear proj in
- possible abs pos emb & convnextv2 blocks for embedded text before concat
- possible long skip connection (first layer to last layer)

### mmdit.py
- sd3 structure
- timestep as condition
- left stream: text embedded and applied a abs pos emb
- right stream: masked_cond & noised_input concatted and with same conv pos emb as unett

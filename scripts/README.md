# Generate seismic waveforms

In order to generate synthetic seismic waveforms, call this following on the command line and follow the instructions:

```shell
python generate_waveforms.py
```

## Example calls

Create 32 samples using command line arguments:

```shell
 python generate_waveforms.py \
  --hypocentral_distance 531 \
  --is_shallow_crustal 1 \
  --magnitude 6 \
  --vs30 154 \
  --num_samples 32 \
  --output waveforms.h5 \
  --edm_checkpoint ../weights/edm.ckpt \
  --autoencoder_checkpoint ../weights/autoencoder.ckpt
```

Create samples using a CSV file:

```shell
 python generate_waveforms.py \
  --csv [japan.csv | little_japan.csv] \
  --output waveforms.h5 \
  --edm_checkpoint ../weights/edm.ckpt \
  --autoencoder_checkpoint ../weights/autoencoder.ckpt
```

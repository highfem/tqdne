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

## Convert `waveforms.h5` to seisbench framework:

  

To convert hdf5 file to seisbench framework, first prepare the station metadata consists of list of columns similar to the figure below in `.csv` format:
![Example waveform Plot](https://github.com/highfem/tqdne/tree/main/scripts/station_metadata.png)

Run the `write_to_seisbench.py` with the following command:
Inspecting the command:
```shell
python write_to_seisbench.py -h
```
Running the command:
```shell
python write_to_seisbench.py <path_to_station_metadata>/station_metadata.csv <path_to_gwm>/waveforms.h5 --origin_time "2024-02-01T00:00:00.0" --hypocenter 12 10 4 --magnitude 6 --num_realizations 10 --trace_sampling_rate 100 <path_to_seisbench>/events
```


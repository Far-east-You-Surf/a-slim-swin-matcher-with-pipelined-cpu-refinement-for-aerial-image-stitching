# Data

Set `AERIAL_DATA` to the folder that holds the datasets; the scripts expect the layout below (paths relative to it).

| Dataset | Use | Source | Expected location |
|---|---|---|---|
| UseGeo, flights 1–3 | accuracy with LiDAR ground truth, registration checked against ground truth, sequence test | https://usegeo.fbk.eu/ (Nex et al., ISPRS Open Journal of Photogrammetry and Remote Sensing, 2024) | `data_tambahan/usegeo/Dataset-{1,2,3}/` with the original structure, including `Depth_resized/undistorted_images/*_res.jpg` and `Depth_resized/depth_maps/*_depth_res.tiff` |
| sheffield_park_1 | throughput and energy on 4000×3000 photographs of another camera (its registration outputs are in the per-pair archive) | OpenDroneMap datasets, https://github.com/OpenDroneMap/ODMdata | `Data Uji Bersih/drone_dataset_sheffield_park_1-master/` |
| Seneca, Bellus | registration on little texture | OpenDroneMap datasets (odm_data_seneca, odm_data_bellus) | `odm_lowtex/odm_data_{seneca,bellus}/images/` |
| Public training data | reproduction without private data | OpenAerialMap UAV orthomosaics (CC-BY 4.0 and CC BY-NC 4.0), OpenDroneMap Monterey Road Bend (CC-BY-SA 4.0), DroneMapper Back 9 sample data; the exact list with licences is `revisi_d/results/public_sources.json` | `data_publik_arev/{orthos_train,orthos_val,flights}/`, assembled by `revisi_d/scripts/arev_public_assemble.py` and `arev_pseudo_public.py` |

The orthomosaics and survey flights used to train the reported models are private and cannot be distributed. The
public reproduction follows the same recipe and the same filters, with every source more than 100 km from every test
site; its decision rules and the deviations of its data sources are recorded in `docs/preregistration.md`.

Pairs are indexed by flight and frame distance `d`; `revisi_d/scripts/evaltool.py` builds the pair lists, and
`revisi_d/results/a4/overlap_usegeo_gt.json` gives the overlap of every UseGeo pair measured from the ground truth
poses and depth maps.

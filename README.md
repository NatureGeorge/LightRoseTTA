## LightRoseTTA - Pytorch

> forked from [psp3dcg/LightRoseTTA]((https://github.com/psp3dcg/LightRoseTTA)).

## Comments

* unideal stereochemical quality of both backbone and sidechain
    * the MLP, SE(3)-Transformer, as well as the full-atom energy potential help little
    * so the model is basically fitting the 2d map
* does not handle chirality and thus can sometimes yield mirrored structure
    * the mirrored stucture emerged before the SE(3)-Transformer
* the performance highly depends on the availability of the templates
    * verified by replacing `pdb100_2021Mar03` with `pdb100_2020Mar11` and tested on CASP14 targets
* can not handle large multi-domain proteins
    * since the model seems only to be trained on single domains

## NOTE

For more information, please turn to their original [README](https://github.com/psp3dcg/LightRoseTTA).

## References

~~X Wang, et al., LightRoseTTA: High-efficient and Accurate Protein Structure Prediction Using an Ultra-Lightweight Deep Graph Model, bioRxiv 10.1101/2023.11.20.566676 (2023).~~

X. Wang, T. Zhang, G. Liu, Z. Cui, Z. Zeng, C. Long, W. Zheng, J. Yang, LightRoseTTA: High-Efficient and Accurate Protein Structure Prediction Using a Light-Weight Deep Graph Model. Adv. Sci. 2024, 2309051. https://doi.org/10.1002/advs.202309051





	

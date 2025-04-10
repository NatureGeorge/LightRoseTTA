## LightRoseTTA - Pytorch

> forked from [psp3dcg/LightRoseTTA]((https://github.com/psp3dcg/LightRoseTTA)).

## Comments

* unideal stereochemical quality of both backbone and sidechain
    * the MLP, SE(3)-Transformer, as well as the full-atom energy potential help little
* does not handle chirality and thus can sometimes yield mirrored structure
    * the mirrored stucture emerged before the SE(3)-Transformer
    * so the model is basically fitting the 2d map
* the performance highly depends on the availability of the templates
    * verified by replacing `pdb100_2021Mar03` with `pdb100_2020Mar11` and tested on CASP14 targets
* can not handle large multi-domain proteins
    * since the model seems only to be trained on single domains

<table>
	<tr>
		<td>
			<img src="https://github.com/user-attachments/assets/b4770752-8b1c-4088-af34-cd0845f1fde5">
		</td>
		<td>
			<img src="https://github.com/user-attachments/assets/d096e1e4-eee3-4c4f-ba5f-e7bd9e4e7376">
		</td>
		<td>
			<img src="https://github.com/user-attachments/assets/94930f65-aff5-46dd-8263-f50a86626ca7">
		</td>
	</tr>
	<tr>
		<td>
			T1031 (all_atom, pdb100_2021Mar03)
		</td>
		<td>
			1UG6.A (backbone, pdb100_2021Mar03)
		</td>
		<td>
			mirrored 1UG6.A (backbone, pdb100_2021Mar03)
		</td>
	</tr>
	<tr>
		<td>
			<img src="https://github.com/user-attachments/assets/11090fdd-27cb-4fdd-af50-0afcb772b56d">
		</td>
		<td>
			<img src="https://github.com/user-attachments/assets/07464a04-9ce3-4fea-9b8e-3d946a470c46">
		</td>
		<td>
			<img src="https://github.com/user-attachments/assets/c9bbc97a-2852-473f-83ba-3bc67c3ef74b">
		</td>
	</tr>
	<tr>
		<td>
			T1049 (backbone, pdb100_2021Mar03)
		</td>
		<td>
			T1049 (backbone, <b>pdb100_2020Mar11</b>)
		</td>
		<td>
			4F92.A (backbone, pdb100_2021Mar03)
		</td>
	</tr>
</table>

## NOTE

For more information, please turn to their original [README](https://github.com/psp3dcg/LightRoseTTA).

## References

~~X Wang, et al., LightRoseTTA: High-efficient and Accurate Protein Structure Prediction Using an Ultra-Lightweight Deep Graph Model, bioRxiv 10.1101/2023.11.20.566676 (2023).~~

X. Wang, T. Zhang, G. Liu, Z. Cui, Z. Zeng, C. Long, W. Zheng, J. Yang, LightRoseTTA: High-Efficient and Accurate Protein Structure Prediction Using a Light-Weight Deep Graph Model. Adv. Sci. 2024, 2309051. https://doi.org/10.1002/advs.202309051





	

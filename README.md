<<<<<<< HEAD
# MDM: Human Motion Diffusion Model
=======
# Large-Scale 3D Representations for Continuous American Sign Language Understanding
>>>>>>> 7ab4e97809abe3a51ed7ad22ad020be86e57f094

This is the official Repo for "Large-Scale 3D Representations for Continuous American Sign Language Understanding". This sepcific codebase is for Sign Language Production.

## Dataset:
Prepare the data, we uploaded the data to HuggingFace. use  ```hugginface-cli``` to download the data to under ```data-bin``` under the main folder. Create one, if not present. The folder should looks like:

```
|--data-bin/
|  |--motions/
|  |  |--<file1.pt>
|  |  |--<file2.pt>
|  |-- texts/
```


### For OpenASL-35k
```
|--data-bin/
|  |--motions/
|  |  |--<file.train.pkl>
|  |  |--<file.test.pkl>
|  |-- texts/
```

> Sign Representations can be found here: [Data](https://huggingface.co/datasets/dongludeeplearning/OpenASL3D_Dataset/), ```texts``` folder can be found here: [texts](https://buffalo.box.com/s/5fbomz6rr007dcrvc15ek7uzur13wypl). For OpenASl-35k, [download](https://buffalo.box.com/s/15nanqj5bmw0vnab6ekk0qrmyrge602r). Please ensure to unzip the ```texts``` and ```openasl-35k pickle``` for ruuning



## Train
```
bash run_train.sh
```
<<<<<<< HEAD
</details>

### 3. Download the pretrained models

Download the model(s) you wish to use, then unzip and place them in `./save/`. 

<details>
  <summary><b>Text to Motion</b></summary>

**You need only the first one.** 

**HumanML3D**

[NEW!] [humanml_trans_dec_512_bert-50steps](https://drive.google.com/file/d/1z5IW5Qa9u9UdkckKylkcSXCwIYgLPhIC/view?usp=sharing) - Runs 20X faster with improved precision!

[NEW!] [humanml-encoder-512-50steps](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing) - Runs 20X faster with comparable performance!

[humanml-encoder-512](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?usp=sharing) (best model used in the paper)

[humanml-decoder-512](https://drive.google.com/file/d/1q3soLadvVh7kJuJPd2cegMNY2xVuVudj/view?usp=sharing)

[humanml-decoder-with-emb-512](https://drive.google.com/file/d/1GnsW0K3UjuOkNkAWmjrGIUmeDDZrmPE5/view?usp=sharing)

**KIT**

[kit-encoder-512](https://drive.google.com/file/d/1SHCRcE0es31vkJMLGf9dyLe7YsWj7pNL/view?usp=sharing)

</details>

<details>
  <summary><b>Action to Motion</b></summary>

**UESTC**

[uestc](https://drive.google.com/file/d/1goB2DJK4B-fLu2QmqGWKAqWGMTAO6wQ6/view?usp=sharing)

[uestc_no_fc](https://drive.google.com/file/d/1fpv3mR-qP9CYCsi9CrQhFqlLavcSQky6/view?usp=sharing)

**HumanAct12**

[humanact12](https://drive.google.com/file/d/154X8_Lgpec6Xj0glEGql7FVKqPYCdBFO/view?usp=sharing)

[humanact12_no_fc](https://drive.google.com/file/d/1frKVMBYNiN5Mlq7zsnhDBzs9vGJvFeiQ/view?usp=sharing)

</details>

<details>
  <summary><b>Unconstrained</b></summary>

**HumanAct12**

[humanact12_unconstrained](https://drive.google.com/file/d/1uG68m200pZK3pD-zTmPXu5XkgNpx_mEx/view?usp=share_link)

</details>


## Motion Synthesis
<details>
  <summary><b>Text to Motion</b></summary>

### Generate from test set prompts

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your text file

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/example_text_prompts.txt
```

### Generate a single prompt

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --text_prompt "the person walked forward and is picking up his toolbox."
```
</details>

<details>
  <summary><b>Action to Motion</b></summary>

### Generate from test set actions

```shell
python -m sample.generate --model_path ./save/humanact12/model000350000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your actions file

```shell
python -m sample.generate --model_path ./save/humanact12/model000350000.pt --action_file ./assets/example_action_names_humanact12.txt
```

### Generate a single action

```shell
python -m sample.generate --model_path ./save/humanact12/model000350000.pt --action_name "drink"
```
</details>

<details>
  <summary><b>Unconstrained</b></summary>

```shell
python -m sample.generate --model_path ./save/unconstrained/model000450000.pt --num_samples 10 --num_repetitions 3
```
# Continuous Sign Production

### Generate from test set prompts

```shell
python -m sample.generate --model_path /home/exx/Akib/motion-diffusion-model/save/openasl.test.05-13/model000050000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your text file

```shell
python -m sample.generate --model_path ./save/wlasl100_ckp_02.03_text/model000400000.pt --input_text ./assets/sign_words.txt
```

### Generate a single prompt

```shell
python -m sample.generate --model_path ./save/wlasl100_ckp_02.03_text/model000400000.pt --text_prompt "paper"
```




By abuse of notation, (num_samples * num_repetitions) samples are created, and are visually organized in a display of num_samples rows and num_repetitions columns.

</details>

**You may also define:**
* `--device` id.
* `--seed` to sample different prompts.
* `--motion_length` (text-to-motion only) in seconds (maximum is 9.8[sec]).

**Running those will get you:**

* `results.npy` file with text prompts and xyz positions of the generated animation
* `sample##_rep##.mp4` - a stick figure animation for each generated motion.

It will look something like this:

![example](assets/example_stick_fig.gif)

You can stop here, or render the SMPL mesh using the following script.

### Render SMPL mesh

To create SMPL mesh per frame run:

```shell
python -m visualize.render_mesh --input_path /path/to/mp4/stick/figure/file
```

```bash
python -m visualize.render_mesh --input_path /home/exx/Akib/motion-diffusion-model/save/openasl.test.05-13/samples_openasl.test.05-13_000050000_seed10/sample00_rep00.mp4
```

**This script outputs:**
* `sample##_rep##_smpl_params.npy` - SMPL parameters (thetas, root translations, vertices and faces)
* `sample##_rep##_obj` - Mesh per frame in `.obj` format.

**Notes:**
* The `.obj` can be integrated into Blender/Maya/3DS-MAX and rendered using them.
* This script is running [SMPLify](https://smplify.is.tue.mpg.de/) and needs GPU as well (can be specified with the `--device` flag).
* **Important** - Do not change the original `.mp4` path before running the script.

**Notes for 3d makers:**
* You have two ways to animate the sequence:
  1. Use the [SMPL add-on](https://smpl.is.tue.mpg.de/index.html) and the theta parameters saved to `sample##_rep##_smpl_params.npy` (we always use beta=0 and the gender-neutral model).
  1. A more straightforward way is using the mesh data itself. All meshes have the same topology (SMPL), so you just need to keyframe vertex locations. 
     Since the OBJs are not preserving vertices order, we also save this data to the `sample##_rep##_smpl_params.npy` file for your convenience.

## Motion Editing

* This feature is available for text-to-motion datasets (HumanML3D and KIT).
* In order to use it, you need to acquire the full data (not just the texts).
* We support the two modes presented in the paper: `in_between` and `upper_body`.

### Unconditioned editing

```shell
python -m sample.edit --model_path ./save/humanml_trans_enc_512/model000200000.pt --edit_mode in_between
```

**You may also define:**
* `--num_samples` (default is 10) / `--num_repetitions` (default is 3).
* `--device` id.
* `--seed` to sample different prompts.
* `--edit_mode upper_body` For upper body editing (lower body is fixed).


The output will look like this (blue frames are from the input motion; orange were generated by the model):

![example](assets/in_between_edit.gif)

* As in *Motion Synthesis*, you may follow the **Render SMPL mesh** section to obtain meshes for your edited motions.

### Text conditioned editing

Just add the text conditioning using `--text_condition`. For example:

```shell
python -m sample.edit --model_path ./save/humanml_trans_enc_512/model000200000.pt --edit_mode upper_body --text_condition "A person throws a ball"
```

The output will look like this (blue joints are from the input motion; orange were generated by the model):

![example](assets/upper_body_edit.gif)

## Train your own MDM

<details>
  <summary><b>Text to Motion</b></summary>

**HumanML3D**

To reproduce the original paper model, run:

```shell
python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset humanml
```

**OpenASL**
```shell
python3 -m train.train_mdm --save_dir save/my_openasl_trans_enc_512 --dataset openasl --overwrite --diffusion_steps 50 --arch gru --text_encoder_type clip --mask_frames --cond_mode  text
```

To reproduce MDM-50 steps, Run:

```shell
python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512_50steps --dataset humanml --diffusion_steps 50 --mask_frames --use_ema
```

To reproduce MDM+DistilBERT, Run:

```shell
python -m train.train_mdm --save_dir save/my_humanml_trans_dec_bert_512 --dataset humanml --diffusion_steps 50 --arch trans_dec --text_encoder_type bert --mask_frames --use_ema
```


**KIT**
```shell
python -m train.train_mdm --save_dir save/my_kit_trans_enc_512 --dataset kit
```
</details>
<details>
  <summary><b>Action to Motion</b></summary>

```shell
python -m train.train_mdm --save_dir save/my_name --dataset {humanact12,uestc} --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1
```
</details>

<details>
  <summary><b>Unconstrained</b></summary>

```shell
python -m train.train_mdm --save_dir save/my_name --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1  --unconstrained
```
</details>


* **Recommended:** Add `--eval_during_training` and `--gen_during_training` to evaluate and generate motions for each saved checkpoint. 
  This will slow down training but will give you better monitoring.
* **Recommended:** Add `--use_ema` for Exponential Moving Average, and `--mask_frames` to fix a masking bug. Both improve performance.
* Use `--diffusion_steps 50` to train the faster model with less diffusion steps.
* Use `--device` to define GPU id.
* Use `--arch` to choose one of the architectures reported in the paper `{trans_enc, trans_dec, gru}` (`trans_enc` is default).
* Use `--text_encoder_type` to choose the text encoder `{clip, bert}` (`clip` is default).
* Add `--train_platform_type {WandBPlatform, TensorboardPlatform}` to track results with either [WandB](https://wandb.ai/site/) or [Tensorboard](https://www.tensorflow.org/tensorboard).


## Evaluate

<details>
  <summary><b>Text to Motion</b></summary>

<!-- * Takes about 20 hours (on a single GPU) -->
* The output of this script for the pre-trained models (as was reported in the paper) is provided in the checkpoints zip file.

**HumanML3D**
```shell
python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512/model000475000.pt
```

**KIT**
```shell
python -m eval.eval_humanml --model_path ./save/kit_trans_enc_512/model000400000.pt
```
</details>

<details>
  <summary><b>Action to Motion</b></summary>

* Takes about 7 hours for UESTC and 2 hours for HumanAct12 (on a single GPU)
* The output of this script for the pre-trained models (as was reported in the paper) is provided in the checkpoints zip file.

```shell
python -m eval.eval_humanact12_uestc --model <path-to-model-ckpt> --eval_mode full
```
where `path-to-model-ckpt` can be a path to any of the pretrained action-to-motion models listed above, or to a checkpoint trained by the user.

</details>


<details>
  <summary><b>Unconstrained</b></summary>

* Takes about 3 hours (on a single GPU)

```shell
python -m eval.eval_humanact12_uestc --model ./save/unconstrained/model000450000.pt --eval_mode full
```

Precision and recall are not computed to save computing time. If you wish to compute them, edit the file eval/a2m/gru_eval.py and change the string `fast=True` to `fast=False`.

</details>

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
=======

> For Sign Language Translation pipeline, please [refer](https://github.com/neccam/slt)
>>>>>>> 7ab4e97809abe3a51ed7ad22ad020be86e57f094

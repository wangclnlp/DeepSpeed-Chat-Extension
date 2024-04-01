# Prior Constraints-based Reward Model Training for Aligning Large Language Models

This is the code edited for paper [Prior Constraints-based Reward Model Training for Aligning Large Language Models]() based on [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples) project.

- Abstract

  Reinforcement learning with human feedback for aligning large language models (LLMs) trains a reward model typically using ranking loss with comparison pairs. 
  However, the training procedure suffers from an inherent problem: the uncontrolled scaling of reward scores during reinforcement learning due to the lack of constraints while training the reward model.
  This paper proposes a **P**rior **C**onstraints-based **R**eward **M**odel (**PCRM**) training method to mitigate this problem.
  PCRM incorporates prior constraints—specifically, length ratio and cosine similarity between outputs of each comparison pair—during reward model training to regulate optimization magnitude and control score margins.
  We comprehensively evaluate PCRM by examining its rank correlation with human preferences and its effectiveness in aligning LLMs via RL. Experimental results demonstrate that PCRM significantly improves alignment performance by effectively constraining reward score scaling.
  As another bonus, our method is easily integrated into arbitrary rank-based alignment methods, such as direct preference optimization, and can yield consistent improvement.

------

## Installation

Make sure your Python>=3.9.

```bash
pip install -r requirements.txt
pip install -e .
```

## Preparations before Training

- Prepare Datasets

  Datasets are created based on [tatsu-lab/alpaca_farm](https://github.com/tatsu-lab/alpaca_farm) and [openai/summarize-from-feedback](https://github.com/openai/summarize-from-feedback).

  ```bash
  cd training
  tar -zxvf ./data.tar.gz
  ```

- Prepare Models

  ```bash
  # Download meta-llama/Llama-2-7b-hf
  # Replace hf_*** with your huggingface token 
  huggingface-cli download --token hf_*** --resume-download meta-llama/Llama-2-7b-hf --local-dir models/meta-llama/Llama-2-7b-hf --local-dir-use-symlinks False
  # Download google-bert/bert-base-uncased
  huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir models/google-bert/bert-base-uncased --local-dir-use-symlinks False
  ```
## Steps to Replicate Our Results

- Step1 SFT

  ```bash
  cd step1_supervised_finetuning/
  # dialogue task
  bash training_scripts/llama2/run_llama2_7b.sh
  # summarization task
  bash training_scripts/llama2/run_llama2_7b_tldr.sh
  ```

- Step2 Reward Modeling

  ```bash
  cd ../step2_reward_model_finetuning
  # dialogue task
  ## baseline
  bash training_scripts/llama2/run_llama2_7b_alpaca.sh
  ## PCRM cos sim
  bash training_scripts/llama2/run_llama2_7b_alpaca_PCRM_cos_sim.sh
  ## PCRM len rat
  bash training_scripts/llama2/run_llama2_7b_alpaca_PCRM_len_rat.sh
  ## PCRM random
  bash training_scripts/llama2/run_llama2_7b_alpaca_PCRM_random.sh
  ## PCRM fixed
  bash training_scripts/llama2/run_llama2_7b_alpaca_PCRM_fixed.sh

  # summarization task
  ## baseline
  bash training_scripts/llama2/run_llama2_7b_tldr.sh
  ## PCRM cos sim
  bash training_scripts/llama2/run_llama2_7b_tldr_PCRM_cos_sim.sh
  ## PCRM len rat
  bash training_scripts/llama2/run_llama2_7b_tldr_PCRM_len_rat.sh
  ## PCRM random
  bash training_scripts/llama2/run_llama2_7b_tldr_PCRM_random.sh
  ## PCRM fixed
  bash training_scripts/llama2/run_llama2_7b_tldr_PCRM_fixed.sh
  ```

- Step2 DPO

  ```bash
  cd ../step2_dpo_finetuning
  # dialogue task
  ## baseline
  bash training_scripts/llama2/run_llama2_7b.sh
  ## PCRM cos sim
  bash training_scripts/llama2/run_llama2_7b_PCRM_cos_sim.sh
  ## PCRM len rat
  bash training_scripts/llama2/run_llama2_7b_PCRM_len_rat.sh
  ```

- Step3 RLHF

  ```bash
  cd ../step3_rlhf_finetuning
  # dialogue task
  ## baseline
  bash training_scripts/llama2/run_llama2_7b_alpaca.sh
  ## PCRM cos sim
  bash training_scripts/llama2/run_llama2_7b_alpaca_PCRM_cos_sim.sh
  ## PCRM len rat
  bash training_scripts/llama2/run_llama2_7b_alpaca_PCRM_len_rat.sh

  # summarization task
  ## baseline
  bash training_scripts/llama2/run_llama2_7b_tldr.sh
  ## PCRM cos sim
  bash training_scripts/llama2/run_llama2_7b_tldr_PCRM_cos_sim.sh
  ## PCRM len rat
  bash training_scripts/llama2/run_llama2_7b_tldr_PCRM_len_rat.sh
  ```

---
Thanks to the [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples), [tatsu-lab/alpaca_farm](https://github.com/tatsu-lab/alpaca_farm), [openai/summarize-from-feedback](https://github.com/openai/summarize-from-feedback) project and their contributors❤️❤️❤️!
---
title: Schedule
nav: true
---

# Accepted Papers

### [Analyzing Representational Shifts in Multimodal Models: A Study of Feature Dynamics in Gemma and PaliGemma](https://openreview.net/forum?id=FiVBgTrsHJ)
**Authors:** Aaron C Friedman, Trinabh Gupta, Raine Ma, Sean O'Brien, Kevin Zhu, Cole Blondin

<details>
<summary>Abstract</summary>

Understanding internal representational shifts that occur from the adaptation of large language models (LLMs) to vision-language models (VLMs) provides insight into trade-offs in model interpretability, feature reuse, and task specialization. This paper presents an empirical study on representational shifts that occur when extending the LLM Gemma2-2B into its multimodal successor, PaliGemma2-3B. Our initial performance analysis reveals that sparse autoencoders (SAEs) trained on Gemma struggle to reconstruct PaliGemma’s activations, motivating a deeper investigation into its activation patterns. Across 26 layers, 37% of SAE features show reduced activation in PaliGemma relative to Gemma. Further experiments on CIFAR-100 and TruthfulQA reveal that PaliGemma relies heavily on visual inputs, activating substantially fewer features for text alone. Additional analyses—including Residual Stream SAE Performance Analysis, Activation Frequency and Dead Feature Quantification, Cross-Modal Feature Activity Patterns, and Semantic Robustness under Label Perturbations—provide consistent evidence that PaliGemma’s internal representations are more visually grounded and less aligned with purely textual features. Our findings suggest key representational trade-offs in feature dynamics when transitioning from unimodal to multimodal models.
</details>

---

### [Angular Steering: Behavior Control via Rotation in Activation Space](https://openreview.net/forum?id=scqQxchEyM)
**Authors:** Hieu M. Vu, Tan Minh Nguyen

<details>
<summary>Abstract</summary>

Controlling specific behaviors in large language models while preserving their general capabilities is a central challenge for safe and reliable artificial intelligence (AI) deployment. Current steering methods, such as vector addition and directional ablation, are constrained within a two-dimensional subspace defined by the activation and feature direction, making them sensitive to chosen parameters and potentially affecting unrelated features due to unintended interactions in activation space. We introduce Angular Steering, a novel and flexible method for behavior modulation that operates by rotating activations within a fixed two-dimensional subspace. By formulating steering as a geometric rotation toward or away from a target behavior direction, Angular Steering provides continuous, fine-grained control over behaviors such as refusal and compliance. We demonstrate this method using refusal steering as a use case. Additionally, we propose Adaptive Angular Steering, a selective variant that rotates only activations aligned with the target feature, further enhancing stability and coherence. Angular Steering generalizes existing addition and orthogonalization techniques under a unified geometric rotation framework, simplifying parameter selection and maintaining model stability across a broader range of adjustments. Experiments across multiple model families and sizes show that Angular Steering achieves robust behavioral control while maintaining general language modeling performance, underscoring its flexibility, generalization, and robustness compared to prior approaches.
</details>

---

### [Attend or Perish: Benchmarking Attention in Algorithmic Reasoning](https://arxiv.org/abs/2503.01909)
**Authors:** Michal Spiegel, Michal Štefánik, Marek Kadlčík, Josef Kuchař

<details>
<summary>Abstract</summary>

Can transformers learn to perform algorithmic tasks reliably across previously unseen input/output domains? While pre-trained language models show solid accuracy on benchmarks incorporating algorithmic reasoning, assessing the reliability of these results necessitates an ability to distinguish genuine algorithmic understanding from rote memorization. In this paper, we propose an algorithmic benchmark comprising five tasks of infinite input domains where we can also disentangle and trace the correct, robust algorithm necessary for the task. This allows us to assess (i) models' ability to extrapolate to unseen types of inputs, including new lengths, value ranges or input domains, but also (ii) to assess the robustness of their learned mechanisms. By analyzing attention maps and performing targeted interventions, we causally demonstrate that the attention mechanism is a key bottleneck, directly contributing to failures in extrapolation. We make the implementation of all our tasks and interpretability methods publicly available.
</details>

---

### [Attributing Response to Context: A Jensen–Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](https://openreview.net/forum?id=QC1Pd6MamX)
**Authors:** Ruizhe Li, Chen Chen, Yuchen Hu, Yanjun Gao, Xi Wang, Emine Yilmaz

<details>
<summary>Abstract</summary>

Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen–Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models.
</details>

---

### [BERTology in the Modern World](https://openreview.net/forum?id=FqahMD2wY8)
**Authors:** Michael Li, Nishant Subramani

<details>
<summary>Abstract</summary>

Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information is rooted in studies of early models like BERT and GPT-2. To better understand today's language models, we investigate how both classical architectures (BERT, DeBERTa, GPT-2) and contemporary large language models (Pythia, OLMo-2, Gemma-2, Qwen2.5, Llama-3.1) represent lexical identity and inflectional morphology. We train linear and nonlinear classifiers on layer-wise activations to predict word lemmas and inflectional features. We discover that models concentrate lexical information linearly in early layers and increasingly nonlinearly in later layers, while keeping inflectional information uniformly accessible and linearly separable throughout the layers. Further analysis reveals that these models encode inflectional morphology through generalizable abstractions, but rely predominantly on memorization to encode lexical identity. Remarkably, these patterns emerge across all 16 models we test, despite differences in architecture, size, and training regime (including pretrained and instruction-tuned variants). This consistency suggests that, despite substantial advances in LLM technologies, transformer models organize linguistic information in similar ways, indicating that these properties could be fundamental for next token prediction and are learned early during pretraining. Our code is available at https://github.com/ml5885/model_internal_sleuthing
</details>

---

### [Causal Interventions Reveal Shared Structure Across English Filler–Gap Constructions](https://openreview.net/forum?id=zfqVbPKE2z)
**Authors:** Sasha Boguraev, Christopher Potts, Kyle Mahowald

<details>
<summary>Abstract</summary>

Large Language Models (LLMs) have emerged as powerful sources of evidence for linguists seeking to develop theories of syntax. In this paper, we argue that causal interpretability methods, applied to LLMs, can greatly enhance the value of such evidence by helping us characterize the abstract mechanisms that LLMs learn to use. Our empirical focus is a set of English filler–gap dependency constructions (e.g., questions, relative clauses). Linguistic theories largely agree that these constructions share many properties. Using experiments based in Distributed Interchange Interventions, we show that LLMs converge on similar abstract analyses of these constructions. These analyses also reveal previously overlooked factors – relating to frequency, filler type, and surrounding context – that could motivate changes to standard linguistic theory. Overall, these results suggest that mechanistic, internal analyses of LLMs can push linguistic theory forward.
</details>

---

### [Comparing Prompt and Representation Engineering for Personality Control in Language Models: A Case Study](https://openreview.net/forum?id=vn9TiICAui)
**Authors:** Pengrui Han

<details>
<summary>Abstract</summary>

Language models can exhibit different personalities through methods like prompt engineering and representation engineering, but how these approaches differ in modeling personality traits remains unclear. In this case study, we conduct a systematic comparison of these methods across two tasks: moral decision-making and narrative generation. In moral dilemmas, we examine how personalities (logical, empathetic, conservative, and risk-taking) influence choices between progressive and conservative options, finding that prompt engineering better aligns with intuitive personality traits while control vectors show more consistent but sometimes unexpected behaviors. In narrative generation, we analyze how different personalities (extroverted, introspective, angry, and whimsical) affect story characteristics, revealing that control vectors enable wider emotional range but lower lexical diversity compared to prompting. Our results demonstrate complementary strengths: prompt engineering excels in maintaining personality-aligned behaviors and vocabulary richness, while representation engineering offers more precise control over emotional expression and linguistic complexity. These findings provide insights into choosing and combining personality control methods for different applications.
</details>

---

### [Death by a Thousand Directions: Exploring the Geometry of Harmfulness in LLMs through Subconcept Probing](https://openreview.net/forum?id=CMrHN5YnfY)
**Authors:** McNair Shah, Saleena Angeline Sartawita, Adhitya Rajendra Kumar, Naitik Chheda, Kevin Zhu, Vasu Sharma, Sean O'Brien, Will Cai

<details>
<summary>Abstract</summary>

Recent advances in large language models (LLMs) have intensified the need to understand and reliably curb their harmful behaviours. We introduce a multidimensional framework for probing and steering harmful content in model internals. For each of 55 distinct harmfulness subconcepts (e.g., racial hate, employment scams, weapons), we learn a linear probe, yielding 55 interpretable directions in activation space. Collectively, these directions span a harmfulness subspace that we show is strikingly low-rank. We then test ablation of the entire subspace from model internals, as well as steering and ablation in the subspace's dominant direction. We find that dominant direction steering allows for near elimination of harmfulness with a low decrease in utility.  
Our findings advance the emerging view that concept subspaces provide a scalable lens on LLM behaviour and offer practical tools for the community to audit and harden future generations of language models.
</details>

---

### [Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models](https://openreview.net/forum?id=N8zX1XFTfn)
**Authors:** Benjamin Reichman, Adar Avsian, Larry Heck

<details>
<summary>Abstract</summary>

This work investigates how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden-state space. Using a synthetic dataset of emotionally rewritten sentences, we identify a low-dimensional emotional manifold via singular value decomposition and show that emotional representations are directionally encoded, distributed across layers, and aligned with interpretable dimensions. These structures are stable across depth and generalize to eight real-world emotion datasets spanning five languages. Cross-domain alignment yields low error and strong linear probe performance, indicating a universal emotional subspace. Within this space, internal emotion perception can be steered while preserving semantics using a learned intervention module, with especially strong control for basic emotions across languages. These findings reveal a consistent and manipulable affective geometry in LLMs and offer insight into how they internalize and process emotion.
</details>

---

### [Evaluating Contrast Localizer for Identifying Causal Units in Social & Mathematical Tasks in Language Models](https://openreview.net/forum?id=2jq1xa93rn)
**Authors:** Yassine Jamaa, Badr AlKhamissi, Satrajit S Ghosh, Martin Schrimpf

<details>
<summary>Abstract</summary>

This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive task pairs and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the specificity of contrast-based methods and highlight the need for broader stimulus sets and more accurately capture task-specific units.
</details>

---

### [From Indirect Object Identification to Syllogisms: Exploring Binary Mechanisms in Transformer Circuits](https://openreview.net/forum?id=P8rwvwB3qH)
**Authors:** Karim Saraipour, Shichang Zhang

<details>
<summary>Abstract</summary>

Transformer-based large language models (LLMs) can perform a wide range of tasks, and mechanistic interpretability aims to reverse engineer the components responsible for task completion to understand their behavior. Previous mechanistic interpretability research has primarily focused on linguistic tasks like Indirect Object Identification (IOI). In this paper, we investigate the ability of GPT-2 small to handle binary truth values by analyzing its behavior with syllogistic prompts, such as "Statement A is true. Statement B matches statement A. Statement B is," which requires more complex logical reasoning compared to IOI. Through our analysis of several syllogism tasks of varying difficulty, we identify multiple circuits that explain GPT-2’s logical-reasoning capabilities and uncover binary mechanisms that facilitate task completion, including the ability to produce a negated token that does not appear in the input prompt through negative heads. Our evaluation using a faithfulness metric shows that a circuit comprising five attention heads achieves over 90% of the original model’s performance. By relating our findings to IOI analysis, we provide new insights into the roles of attention heads and MLPs in LLMs. We believe these insights contribute to a broader understanding of model reasoning and benefit future research in mechanistic interpretability.
</details>

---

### [How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence](https://openreview.net/forum?id=yg2tat6A3d)
**Authors:** Hongzhe Du, Weikai Li, Min Cai, Karim Saraipour, Zimin Zhang, Yizhou Sun, Himabindu Lakkaraju, Shichang Zhang

<details>
<summary>Abstract</summary>

Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by linear vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward-transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training.
</details>

---

### [Interpreting the Latent Structure of Operator Precedence in Language Models](https://openreview.net/forum?id=NFQ9C7lh6F)
**Authors:** Dharunish Yugeswardeenoo, Harshil Nukala, Cole Blondin, Sean O'Brien, Vasu Sharma, Kevin Zhu

<details>
<summary>Abstract</summary>

Large Language Models (LLMs) have demonstrated impressive reasoning capabilities but continue to struggle with arithmetic tasks. Prior works largely focus on outputs or prompting strategies, leaving the open question of the internal structure through which models do arithmetic computation. In this work, we investigate whether LLMs encode operator precedence in their internal representations via the open-source instruction-tuned LLaMA 3.2-3B model. We constructed a dataset of arithmetic expressions with three operands and two operators, varying the order and placement of parentheses. Using this dataset, we trace whether intermediate results appear in the residual stream of the instruction-tuned LLaMA 3.2-3B model. We apply interpretability techniques such as logit lens, linear classification probes, and UMAP geometric visualization. Our results show that intermediate computations are present in the residual stream, particularly after MLP blocks. We also find that the model linearly encodes precedence in each operator's embeddings post attention layer. We introduce partial embedding swap, a technique that modifies operator precedence by exchanging high-impact embedding dimensions between operators.
</details>

---

### [LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization](https://openreview.net/forum?id=8jkQGNkRzP)
**Authors:** Jiarui Liu, Jivitesh Jain, Mona T. Diab, Nishant Subramani

<details>
<summary>Abstract</summary>

Although large language models (LLMs) have tremendous utility, trustworthiness is still a chief concern: models often generate incorrect information with high confidence. While contextual information can help guide generation, identifying when a query would benefit from retrieved context and assessing the effectiveness of that context remains challenging. In this work, we operationalize interpretability methods to ascertain whether we can predict the correctness of model outputs from the model’s activations alone. We also explore whether model internals contain signals about the efficacy of external context. We consider correct, incorrect, and irrelevant context and introduce metrics to distinguish amongst them. Experiments on six different models reveal that a simple classifier trained on intermediate layer activations of the first output token can predict output correctness with nearly 80% accuracy, enabling early auditing. Our model-internals-based metric significantly outperforms prompting baselines at distinguishing between correct and incorrect context, guarding against inaccuracies introduced by polluted context. These findings offer a lens to better understand the underlying decision-making processes of LLMs.
</details>

---

### [Localizing Persona Representations in LLMs](https://openreview.net/forum?id=QnayDNXZNO)
**Authors:** Celia Cintas, Miriam Rateike, Erik Miehling, Elizabeth M. Daly, Skyler Speakman

<details>
<summary>Abstract</summary>

We present a study on how and where personas – defined by distinct sets of human characteristics, values, and beliefs – are encoded in the representation space of large language models (LLMs). Such insights can improve the model's interpretability and enable more precise control over the generative process.  
Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces.  
We find that, across multiple pre-trained decoder-only LLMs,  
the analyzed personas show large differences in representation space only within the final third of the decoder layers.  
When we look at some of the later layers, we observe overlapping activations for specific ethical perspectives – such as moral nihilism and utilitarianism – suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions.  
These findings improve our understanding of how LLMs represent information internally and allow for greater control over how specific human traits are expressed in their responses.
</details>

---

### [Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps](https://arxiv.org/abs/2502.14829)
**Authors:** Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, Yonatan Belinkov

<details>
<summary>Abstract</summary>

When prompted to think step-by-step, language models (LMs) produce a chain of thought (CoT), a sequence of reasoning steps that the model supposedly used to produce its prediction. Despite much work on CoT prompting, it is unclear if reasoning verbalized in a CoT is faithful to the models’ parametric beliefs. We introduce a framework for measuring parametric faithfulness of generated reasoning and propose Faithfulness by Unlearning Reasoning steps (FUR), an instance of this framework. FUR erases information contained in reasoning steps from model parameters and measures faithfulness as the resulting effect on the model’s prediction. Our experiments with four LMs and five multi-choice question answering (MCQA) datasets show that FUR is frequently able to precisely change the underlying models’ prediction for a given instance by unlearning key steps, indicating when a CoT is parametrically faithful. Further analysis shows that CoTs generated by models post-unlearning support different answers, hinting at a deeper effect of unlearning.
</details>

---

### [MICE for CATs: Model-Internal Confidence Estimation for Calibrating Agents with Tools](https://aclanthology.org/2025.naacl-long.615/)
**Authors:** Nishant Subramani, Jason Eisner, Justin Svegliato, Benjamin Van Durme, Yu Su, Sam Thomson

<details>
<summary>Abstract</summary>

Tool-using agents that act in the world need to be both useful and safe. Well-calibrated model confidences can be used to weigh the risk versus reward of potential actions, but prior work shows that many models are poorly calibrated. Inspired by interpretability literature exploring the internals of models, we propose a novel class of model-internal confidence estimators (MICE) to better assess confidence when calling tools. MICE first decodes from each intermediate layer of the language model using logit lens and then computes similarity scores between each layer’s generation and the final output. These features are fed into a learned probabilistic classifier to assess confidence in the decoded output. On the simulated trial and error (STE) tool-calling dataset using Llama3 models, we find that MICE beats or matches the baselines on smoothed expected calibration error. Using MICE confidences to determine whether to call a tool significantly improves over strong baselines on a new metric, expected tool-calling utility. Further experiments show that MICE is sample-efficient, can generalize zero-shot to unseen APIs, and results in higher tool-calling utility in scenarios with varying risk levels. Our code is open source, available at https://github.com/microsoft/mice_for_cats.
</details>

---

### [Number Embeddings of Pre-trained LMs are Remarkably Accurate](https://arxiv.org/abs/2502.14829)
**Authors:** Marek Kadlčík, Michal Štefánik, Timothee Mickus, Josef Kuchař, Michal Spiegel

<details>
<summary>Abstract</summary>

While language models show excellent capacity to model coherent text, it is commonly believed that their limitations reside in tasks requiring exact representations, such as numeric values. This work shows that representations of numbers that encode their nominal numeric values naturally emerge in text-only causal language models. Contrary to previous work assuming linearity of models' representations, we find that different pre-trained models consistently learn highly precise sinusoidal representations already within the input embedding, and can be accurately decoded with an appropriate probing method. These findings undermine existing assumptions about the inherent inability of language models to represent numeric information accurately and, consequently, point to the real limitation of robust arithmetic proficiency in language models in their limited capacity to combine accurate input representations.
</details>

---

### [On the Geometry of Semantics in Next-token Prediction](https://openreview.net/forum?id=LQ3I5OoroE)
**Authors:** Yize Zhao, Christos Thrampoulidis

<details>
<summary>Abstract</summary>

Modern language models demonstrate a remarkable ability to capture linguistic meaning despite being trained solely through next-token prediction (NTP). We investigate how this conceptually simple training objective leads models to extract and encode latent semantic and grammatical concepts.  Our analysis reveals that NTP optimization implicitly guides models to encode concepts via singular value decomposition (SVD) factors of a centered data-sparsity matrix that captures next-word co-occurrence patterns. While the model never explicitly constructs this matrix, learned word and context embeddings effectively factor it to capture linguistic structure. We find that the most important SVD factors are learned first during training, motivating using spectral clustering of embeddings to identify human-interpretable semantics, including both classical k-means and a new orthant-based method directly motivated by our interpretation of concepts. Overall, our work bridges distributional semantics, neural collapse geometry, and neural network training dynamics, providing insights into how NTP's implicit biases shape the emergence of meaning representations in language models.
</details>

---

### [One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs](https://openreview.net/forum?id=bfnqgL06FK)
**Authors:** Jacob Dunefsky, Arman Cohan

<details>
<summary>Abstract</summary>

Steering vectors (SVs) have emerged as a promising approach for interpreting and controlling LLMs, but current methods typically require large contrastive datasets that are often impractical to construct and may capture spurious correlations.  
We propose directly optimizing SVs through gradient descent on a single training example, and systematically investigate how these SVs generalize.  
We consider several SV optimization techniques and find that the resulting SVs effectively mediate safety-relevant behaviors in multiple models.  
Indeed, in experiments on an alignment-faking model, we are able to optimize one-shot SVs that induce harmful behavior on benign examples and whose negations suppress harmful behavior on malign examples.  
And in experiments on refusal suppression, we demonstrate that one-shot optimized SVs can transfer across inputs, yielding a Harmbench attack success rate of 96.9%.  
Furthermore, we extend work on "emergent misalignment" and show that SVs optimized to induce a model to write vulnerable code cause the model to respond harmfully on unrelated open-ended prompts.  
Finally, we use one-shot SV optimization to investigate how an instruction-tuned LLM recovers from outputting false information, and find that this ability is independent of the model's explicit verbalization that the information was false.  
Overall, our findings suggest that optimizing SVs on a single example can mediate a wide array of misaligned behaviors in LLMs.
</details>

---

### [Predicting Success of Model Editing via Intrinsic Features](https://openreview.net/forum?id=qQNiXK0U0J)
**Authors:** Yanay Soker, Martin Tutek, Yonatan Belinkov

<details>
<summary>Abstract</summary>

Due to the ever-changing nature of information in the world, the ability to update factual knowledge of LLMs is important both for maintaining their veracity and for reducing the costs of retraining. Model editing has emerged as a research area that aims to perform surgical updates to model parameters with the goal of updating factually incorrect or outdated information. However, the components underpinning success of an edit applied to an LLM are unknown. In this work, we propose two metrics and show empirically that they can serve as indicators of editing outcomes: (1) the location where the knowledge is stored in the parameters, as reflected by the logit-lens technique; and (2) the probability that the model assigns to the original output. We find a correlation between the location of the knowledge and the optimal layer for editing, as well as between the output probability and the edit success, as measured by efficacy and specificity. We also demonstrate the potential use of output probability for setting the regularization of the editing process.
</details>

---

### [Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking](https://openreview.net/forum?id=madN8XeB7s)
**Authors:** Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen, Xi Ye

<details>
<summary>Abstract</summary>

Recent work has identified retrieval heads (Wu et al., 2025), a subset of attention heads responsible for retrieving salient information in long-context language models (LMs), as measured by their copy-paste behavior in Needle-in-a-Haystack tasks. In this paper, we introduce QRHead (Query-Focused Retrieval Head), an improved set of attention heads that significantly enhance retrieval from long contexts. We identify QRHead by aggregating attention scores with respect to the input query, using a handful of examples from real-world tasks (e.g. long-context QA). We further introduce QRRetriever, an efficient and effective retriever that uses the accumulated attention mass of QRHead as retrieval scores. We use QRRetriever for long-context reasoning by selecting the most relevant parts with the highest retrieval scores. On multi-hop reasoning tasks LongMemEval and CLIPPER, this yields over 10% performance gains over full context and outperforms strong dense retrievers. We also evaluate QRRetriever as a re-ranker on the BEIR benchmark and find that it achieves strong zero-shot performance, outperforming other LLM-based re-rankers such as RankGPT. Further analysis shows that both the query-context attention scoring and task selection are crucial for identifying QRHead with strong downstream utility. Overall, our work contributes a general-purpose retriever and offers interpretability insights into the long-context capabilities of LMs.
</details>

---

### [Safety Subspaces are Not Distinct: A Fine-Tuning Case Study](https://openreview.net/forum?id=2uLBkfMyX5)
**Authors:** Shaan Shah, Kaustubh Ponkshe, Raghav Singhal, Praneeth Vepakomma

<details>
<summary>Abstract</summary>

Large Language Models (LLMs) rely on safety alignment to produce socially acceptable responses. This is typically achieved through instruction tuning and reinforcement learning from human feedback. However, this alignment is known to be brittle: further fine-tuning, even on benign or lightly contaminated data, can degrade safety and reintroduce harmful behaviors. A growing body of work suggests that alignment may correspond to identifiable geometric directions in weight space, forming subspaces that could, in principle, be isolated or preserved to defend against misalignment. In this work, we conduct a comprehensive empirical study of this geometric perspective. We examine whether safety-relevant behavior is concentrated in specific subspaces, whether it can be separated from general-purpose learning, and whether harmfulness arises from distinguishable patterns in internal representations. Across both parameter and activation space, our findings are consistent: subspaces that amplify safe behaviors also amplify unsafe ones, and prompts with different safety implications activate overlapping representations. We find no evidence of a subspace that selectively governs safety. These results challenge the assumption that alignment is geometrically localized. Rather than residing in distinct directions, safety appears to emerge from entangled, high-impact components of the model’s broader learning dynamics. This suggests that subspace-based defenses may face fundamental limitations and underscores the need for alternative strategies to preserve alignment under continued training. We corroborate these findings through multiple experiments on five open-source LLMs. Our code is available anonymously at: https://github.com/CERT-Lab/safety-subspaces
</details>

---

### [Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs](https://openreview.net/forum?id=EPxv1f71Ka)
**Authors:** Ziling Cheng, Meng Cao, Marc-Antoine Rondeau, Jackie CK Cheung

<details>
<summary>Abstract</summary>

The widespread success of LLMs on NLP benchmarks has been accompanied by concerns that LLMs function primarily as stochastic parrots that reproduce texts similar to what they saw during pre-training, often erroneously. But what is the nature of their errors, and do these errors exhibit any regularities? In this work, we examine irrelevant context hallucinations, in which models integrate misleading contextual cues into their predictions. Through behavioral analysis, we show that these errors result from a structured yet flawed mechanism that we term _class-based (mis)generalization_, in which models combine abstract class cues with features extracted from the query or context to derive answers. Furthermore, mechanistic interpretability experiments on Llama-3, Mistral, and Pythia across 39 factual recall relation types reveal that this behavior is reflected in the model's internal computations: (i) abstract class representations are constructed in lower layers before being refined into specific answers in higher layers, (ii) feature selection is governed by two competing circuits  –  one prioritizing direct query-based reasoning, the other incorporating contextual cues  –  whose relative influences determine the final output. Our findings provide a more nuanced perspective on the stochastic parrot argument: through form-based training, LLMs can exhibit generalization leveraging abstractions, albeit in unreliable ways based on contextual cues — what we term _stochastic chameleons_.
</details>

---

### [Understanding In-context Learning of Addition via Activation Subspaces](https://openreview.net/forum?id=78NuRFVQHU)
**Authors:** Xinyan Hu, Kayo Yin, Michael I. Jordan, Jacob Steinhardt, Lijie Chen

<details>
<summary>Abstract